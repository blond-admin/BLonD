# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Analytic single-bunch matching routines (BLonD 2 style).

Assembles the analytic building blocks — RF potential well, separatrix
cut, 2D Hamiltonian, distribution families/Abel transform and
bunch-length/emittance targeting — into
:class:`~blond.beam_preparation.base.MatchingRoutine` classes usable
with :meth:`~blond.core.simulation.simulation.Simulation.prepare_beam`:

* :class:`AnalyticDistributionMatcher` — the BLonD 2
  ``matched_from_distribution_function`` workflow: a stationary
  distribution family :math:`g(H)` sized to a bunch length or
  emittance target.
* :class:`LineDensityMatcher` — the BLonD 2
  ``matched_from_line_density`` workflow: a *line density* (typically a
  measured bunch profile) Abel-inverted to the distribution function
  :math:`F(H)` that reproduces it.

Both include the intensity-effect iteration: if the ring contains
:class:`~blond.physics.impedances.base.WakeField` elements, the induced
potential of the smooth candidate line density is added to the RF well
and the matching is iterated to self-consistency. Unlike BLonD 2, the
correction can be under-relaxed (``relaxation_factor``) to stabilise
the iteration when collective effects are strong.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from blond.beam_preparation.base import MatchingRoutine
from blond.beam_preparation.bigaussian import get_main_harmonic_attributes
from blond.beam_preparation.helpers import populate_beam
from blond.core.helpers import int_from_float_with_warning
from blond.experimental.beam_preparation.analytic_abel import (
    distribution_from_line_density,
)
from blond.experimental.beam_preparation.analytic_action import (
    action_from_potential_well,
    hamiltonian_from_emittance,
)
from blond.experimental.beam_preparation.analytic_distributions import (
    distribution_function,
    line_density,
    x0_from_bunch_length,
)
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)
from blond.experimental.beam_preparation.analytic_induced_potential import (
    clone_wakefields_on_smooth_profile,
    induced_voltage_from_line_density,
)
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    rf_potential_well,
)
from blond.experimental.beam_preparation.analytic_well_cut import (
    cut_potential_well,
)
from blond.generals.cupy.no_cupy_import import AllowPlotting, copy_to_cpu
from blond.generals.iterables_ import all_equal

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def _machine_parameters(simulation: Simulation, beam: BeamBaseClass) -> dict:
    """
    Longitudinal machine parameters shared by the analytic matchers.

    Reuses ``get_main_harmonic_attributes`` and the drifts' slippage
    factor (required identical across drifts) — no third
    parameter-extraction variant.
    """
    from blond.physics.drifts import DriftSimple

    _, omega_rf, _, _ = get_main_harmonic_attributes(
        beam=beam, simulation=simulation
    )
    drifts = simulation.ring.elements.get_elements(
        DriftSimple, recursive=False
    )
    eta_0_values = [
        drift.eta_0(gamma=beam.reference.gamma) for drift in drifts
    ]
    assert all_equal(eta_0_values), (
        f"Expected all `eta_0` to be the same, got {eta_0_values}."
    )
    eta_0 = eta_0_values[0]
    energy_gain_per_turn = (
        simulation.magnetic_cycle.get_target_total_energy(
            turn_i=0,
            section_i=0,
            reference_time=0,
            particle_type=beam.particle_type,
        )
        - beam.reference.total_energy
    )
    return dict(
        omega_rf=omega_rf,
        eta_0=eta_0,
        charge=beam.particle_type.charge,
        t_rev=simulation.get_t_rev_init(),
        energy_gain_per_turn=energy_gain_per_turn,
        eom_factor_dE=calc_eom_factor_dE(
            eta_0=eta_0,
            beta=beam.reference.beta,
            total_energy=beam.reference.total_energy,
        ),
    )


def _ring_has_wakefields(simulation: Simulation) -> bool:
    """Whether the ring contains any WakeField element."""
    from blond.physics.impedances.base import WakeField

    return (
        len(simulation.ring.elements.get_elements(WakeField, recursive=False))
        > 0
    )


def _resolve_dt_margin_fraction(
    dt_margin_fraction: float | None, ring_has_wakefields: bool
) -> float:
    """
    Frame margin default: 0.4 with wakefields (BLonD 2), else 0.

    The induced potential shifts and tilts the bucket; the margin keeps
    it inside the frame.
    """
    if dt_margin_fraction is not None:
        return dt_margin_fraction
    return 0.4 if ring_has_wakefields else 0.0


def _total_rf_voltage(
    simulation: Simulation, time_array: NumpyArray
) -> NumpyArray:
    """Total RF voltage waveform summed over all RF stations, in [V]."""
    from blond.physics.cavities import (
        MultiHarmonicRFStation,
        SingleHarmonicRFStation,
    )

    rf_stations = simulation.ring.elements.get_elements(
        SingleHarmonicRFStation, recursive=False
    ) + simulation.ring.elements.get_elements(
        MultiHarmonicRFStation, recursive=False
    )
    total_voltage = np.zeros_like(time_array)
    for rf_station in rf_stations:
        total_voltage += copy_to_cpu(
            rf_station.calc_gap_voltage_without_feedbacks(ts=time_array)
        )
    return total_voltage


class AnalyticDistributionMatcher(MatchingRoutine):
    r"""
    Matched single-bunch generation from an analytic distribution.

    The BLonD 2 ``matched_from_distribution_function`` workflow: the RF
    potential well is reconstructed analytically from the RF stations,
    cut at the separatrix, the 2D Hamiltonian is built and a stationary
    distribution :math:`g(H)` of the requested family is sized to the
    requested bunch length *or* emittance; the beam is then sampled from
    the resulting density grid.

    If the ring contains wakefields, the matching is iterated with
    intensity effects: the induced potential of the *smooth* candidate
    line density (computed by deep copies of the ring's wakefields on a
    dedicated profile — no macroparticles involved) is added to the RF
    well and the distribution re-matched until the potential well is
    self-consistent.

    Parameters
    ----------
    n_macroparticles
        Number of macroparticles to generate.
    distribution_type
        ``"waterbag"``, ``"parabolic_amplitude"``, ``"parabolic_line"``,
        ``"binomial"`` or ``"gaussian"`` (Laclare families).
    exponent
        Binomial exponent :math:`\mu`; required for ``"binomial"``.
    bunch_length
        Target bunch length, in [s] (its meaning set by
        ``bunch_length_fit``; default 4-sigma rms). Exactly one of
        ``bunch_length`` / ``emittance`` must be given.
    bunch_length_fit
        ``"rms"`` (4-sigma rms, default), ``"fwhm"``
        (gaussian-equivalent 4 sigma) or ``"full"`` (full extent).
    emittance
        Target longitudinal emittance, in [eV.s] (area of the matched
        iso-Hamiltonian contour, :math:`2\pi J`).
    seed
        Random seed for the macroparticle sampling.
    n_points_grid
        Resolution of the internal time and energy grids (also the
        resolution of the smooth line-density profile driving the
        induced-voltage computation).
    dt_margin_fraction
        Frame margin as a fraction of the main RF period, so that a
        bucket shifted/tilted by the induced potential stays inside
        the frame. Default (None): ``0.4`` when the ring has
        wakefields (the BLonD 2 ``dt_margin_percent`` default), ``0``
        otherwise.
    maxiter_intensity_effects
        Maximum number of intensity-effect iterations.
    tolerance_potential_well
        Convergence threshold on the fixed-point residual of the
        induced potential, relative to the RF potential-well amplitude.
    relaxation_factor
        Fraction :math:`\alpha \in (0, 1]` of the induced-potential
        correction applied per iteration:
        :math:`\Phi_{i+1} = (1-\alpha)\,\Phi_i + \alpha\,\Phi_{new}`.
        ``1.0`` reproduces the BLonD 2 full-correction behaviour;
        smaller values slow the iteration down (as in tomography /
        gradient-descent schemes) and stabilise cases where strong
        collective effects make the full-step iteration oscillate
        without converging.
    allow_inner_buckets
        If True, wells split by the induced potential (inner buckets)
        are accepted with a warning instead of raising — see
        :func:`~blond.experimental.beam_preparation.analytic_potential_well.check_single_bucket_well`.
    verbose
        If True, print matching diagnostics.
    plot
        If True, draw the requested (matched density) line density
        against the generated macroparticle profile after sampling
        (and the RF vs distorted potential well when intensity effects
        are active).

    Attributes
    ----------
    fitted_x_0
        Fitted distribution size parameter X0, in [eV] (after run).
    matched_bunch_length
        4-sigma rms bunch length of the matched density, in [s].
    matched_emittance
        Longitudinal emittance of the matched :math:`H = X_0` contour,
        :math:`2\pi J(X_0)`, in [eV.s], evaluated in the final
        (intensity-distorted, when wakefields are present) potential
        well. For the binomial families this is the full-bunch
        emittance (X0 is the support edge); for the gaussian it is the
        area of the fitted contour.
    n_intensity_iterations
        Number of induced-potential updates performed (0 when the ring
        has no wakefields).
    final_potential_well_error
        Last fixed-point residual of the induced potential, relative to
        the RF potential-well amplitude (None when the ring has no
        wakefields).

    Examples
    --------
    >>> from blond import Simulation
    >>> from blond.experimental.beam_preparation.analytic_matcher import (
    ...     AnalyticDistributionMatcher,
    ... )
    >>> simulation = Simulation( ... )
    >>> simulation.prepare_beam(
    ...     beam= ... ,
    ...     preparation_routine=AnalyticDistributionMatcher(
    ...         n_macroparticles=1e6,
    ...         distribution_type="parabolic_amplitude",
    ...         bunch_length=1.2e-9,
    ...     ),
    ... )
    """

    def __init__(
        self,
        n_macroparticles: int | float,
        distribution_type: str,
        exponent: float | None = None,
        bunch_length: float | None = None,
        bunch_length_fit: Literal["rms", "fwhm", "full"] = "rms",
        emittance: float | None = None,
        seed: int | None = 0,
        n_points_grid: int = 1000,
        dt_margin_fraction: float | None = None,
        maxiter_intensity_effects: int = 100,
        tolerance_potential_well: float = 1e-6,
        relaxation_factor: float = 1.0,
        allow_inner_buckets: bool = False,
        verbose: bool = False,
        plot: bool = False,
    ) -> None:
        super().__init__()
        if (bunch_length is None) == (emittance is None):
            raise ValueError(
                "Specify exactly one of `bunch_length` or `emittance`."
            )
        if not 0.0 < relaxation_factor <= 1.0:
            raise ValueError(
                f"relaxation_factor must be in (0, 1], "
                f"got {relaxation_factor}."
            )
        self._n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self._distribution_type = distribution_type
        self._exponent = exponent
        self._bunch_length = bunch_length
        self._bunch_length_fit = bunch_length_fit
        self._emittance = emittance
        self._seed = seed
        self._n_points_grid = int(n_points_grid)
        self._dt_margin_fraction = dt_margin_fraction
        self._maxiter_intensity_effects = int(maxiter_intensity_effects)
        self._tolerance_potential_well = tolerance_potential_well
        self._relaxation_factor = relaxation_factor
        self._allow_inner_buckets = allow_inner_buckets
        self._verbose = verbose
        self._plot = plot

        #: Fitted distribution size parameter X0, in [eV] (after run).
        self.fitted_x_0: float | None = None
        #: 4-sigma rms bunch length of the matched density, in [s].
        self.matched_bunch_length: float | None = None
        #: Emittance of the matched H = X0 contour, 2*pi*J(X0) [eV.s].
        self.matched_emittance: float | None = None
        #: Number of induced-potential updates performed.
        self.n_intensity_iterations: int = 0
        #: Last fixed-point residual of the induced potential.
        self.final_potential_well_error: float | None = None
        #: Fixed-point residual per intensity iteration (after run).
        self.intensity_residuals: list[float] = []

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with matched macroparticles.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        """
        super().prepare_beam(simulation=simulation, beam=beam)

        # --- machine parameters (shared helpers, no third variant) ----
        params = _machine_parameters(simulation, beam)
        charge = params["charge"]
        t_rev = params["t_rev"]
        eta_0 = params["eta_0"]
        eom_factor_dE = params["eom_factor_dE"]

        # --- RF potential well from the actual RF waveform ------------
        dt_margin_fraction = _resolve_dt_margin_fraction(
            self._dt_margin_fraction, _ring_has_wakefields(simulation)
        )
        time_array = bucket_time_array(
            params["omega_rf"],
            n_points=self._n_points_grid,
            dt_margin_fraction=dt_margin_fraction,
        )
        rf_potential_raw = rf_potential_well(
            time_array,
            _total_rf_voltage(simulation, time_array),
            charge=charge,
            t_rev=t_rev,
            eta_0=eta_0,
            energy_gain_per_turn=params["energy_gain_per_turn"],
            subtract_min=False,
        )

        # --- intensity-effect iteration (BLonD 2 + under-relaxation) --
        # The smooth line-density profile spans the full frame with the
        # matcher's own resolution: the induced voltage (including the
        # wake tail behind the bunch) and its potential live on the
        # frame directly, with no cut-edge interpolation artefacts.
        induced_potential = np.zeros_like(time_array)
        wakefield_clones, smooth_profile = clone_wakefields_on_smooth_profile(
            simulation, time_array
        )
        rf_amplitude = None
        rf_well_cut_for_plot = None
        residual = None
        self.n_intensity_iterations = 0
        self.final_potential_well_error = None
        self.intensity_residuals = []
        converged = False

        for iteration in range(self._maxiter_intensity_effects + 1):
            total_potential_raw = rf_potential_raw + induced_potential
            time_cut, well_cut = cut_potential_well(
                time_array,
                total_potential_raw,
                allow_inner_buckets=self._allow_inner_buckets,
            )
            if rf_amplitude is None:
                # Iteration 0 cuts the bare RF well: reference for the
                # convergence normalization and the diagnostic plot.
                rf_amplitude = float(well_cut.max() - well_cut.min())
                rf_well_cut_for_plot = (time_cut, well_cut)

            time_grid, deltaE_grid, hamilton_2D = hamiltonian_grid(
                time_cut,
                well_cut,
                eom_factor_dE=eom_factor_dE,
                n_points_deltaE=self._n_points_grid,
                allow_inner_buckets=self._allow_inner_buckets,
            )
            separatrix_level = float(well_cut.max())
            inside_bucket_mask = hamilton_2D <= separatrix_level

            if self._bunch_length is not None:
                x_0 = x0_from_bunch_length(
                    time_cut,
                    hamilton_2D,
                    target_bunch_length=self._bunch_length,
                    distribution_type=self._distribution_type,
                    exponent=self._exponent,
                    bunch_length_fit=self._bunch_length_fit,
                    inside_bucket_mask=inside_bucket_mask,
                )
            else:
                sorted_hamiltonian, sorted_action = action_from_potential_well(
                    time_cut,
                    well_cut,
                    eom_factor_dE=eom_factor_dE,
                    allow_inner_buckets=self._allow_inner_buckets,
                )
                x_0 = hamiltonian_from_emittance(
                    self._emittance, sorted_hamiltonian, sorted_action
                )

            density = distribution_function(
                hamilton_2D, self._distribution_type, x_0, self._exponent
            )
            density = np.where(inside_bucket_mask, density, 0.0)
            density /= density.sum()
            line_density_values = density.sum(axis=0)

            if len(wakefield_clones) == 0:
                break

            if converged or iteration == self._maxiter_intensity_effects:
                if not converged:
                    warnings.warn(
                        "Intensity-effect matching did not converge in "
                        f"{self._maxiter_intensity_effects} iterations "
                        f"(residual {residual:.2e}, tolerance "
                        f"{self._tolerance_potential_well:.2e}). "
                        "Consider a smaller `relaxation_factor`.",
                        UserWarning,
                        stacklevel=2,
                    )
                break

            # Induced potential of the smooth candidate line density,
            # computed on the full frame (the line density vanishes at
            # the separatrix edges, so the frame extension is smooth).
            line_density_frame = np.interp(
                time_array,
                time_cut,
                line_density_values,
                left=0.0,
                right=0.0,
            )
            induced_voltage = induced_voltage_from_line_density(
                wakefield_clones,
                smooth_profile,
                line_density_frame,
                beam,
            )
            induced_potential_new = rf_potential_well(
                time_array,
                induced_voltage,
                charge=charge,
                t_rev=t_rev,
                eta_0=eta_0,
                subtract_min=False,
            )

            # Fixed-point residual (independent of the relaxation).
            residual = float(
                np.sqrt(
                    np.mean((induced_potential_new - induced_potential) ** 2)
                )
                / rf_amplitude
            )
            converged = residual < self._tolerance_potential_well

            # Under-relaxed update (relaxation_factor = 1 -> BLonD 2).
            induced_potential = (
                1.0 - self._relaxation_factor
            ) * induced_potential + (
                self._relaxation_factor * induced_potential_new
            )
            self.n_intensity_iterations = iteration + 1
            self.final_potential_well_error = residual
            self.intensity_residuals.append(residual)
            if self._verbose:
                print(
                    "[AnalyticDistributionMatcher] intensity iteration "
                    f"{iteration + 1:3d}: residual {residual:.3e} "
                    f"(tolerance {self._tolerance_potential_well:.1e}, "
                    f"relaxation {self._relaxation_factor})"
                )

        # --- diagnostics ----------------------------------------------
        # Emittance of the matched contour, 2*pi*J(X0), in the final
        # (possibly intensity-distorted) well. The emittance-target
        # path already tabulated J(H) for this well; the bunch-length
        # path computes it here.
        if self._bunch_length is not None:
            sorted_hamiltonian, sorted_action = action_from_potential_well(
                time_cut,
                well_cut,
                eom_factor_dE=eom_factor_dE,
                allow_inner_buckets=self._allow_inner_buckets,
            )
        self.matched_emittance = float(
            2.0 * np.pi * np.interp(x_0, sorted_hamiltonian, sorted_action)
        )
        total = line_density_values.sum()
        mean_time = (line_density_values * time_cut).sum() / total
        self.matched_bunch_length = float(
            4.0
            * np.sqrt(
                (line_density_values * (time_cut - mean_time) ** 2).sum()
                / total
            )
        )
        self.fitted_x_0 = float(x_0)
        if self._verbose:
            target = (
                f"bunch length {self._bunch_length:.4e} s "
                f"({self._bunch_length_fit})"
                if self._bunch_length is not None
                else f"emittance {self._emittance:.4e} eV.s"
            )
            print(
                "[AnalyticDistributionMatcher] "
                f"{self._distribution_type}, target {target}: "
                f"x_0={self.fitted_x_0:.4e} eV, matched 4-sigma rms "
                f"bunch length {self.matched_bunch_length:.4e} s, "
                f"contour emittance {self.matched_emittance:.4f} eV.s, "
                f"{self.n_intensity_iterations} intensity iteration(s)"
            )

        # --- sampling -------------------------------------------------
        populate_beam(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=density,
            n_macroparticles=self._n_macroparticles,
            seed=self._seed,
        )

        if self._plot:
            self._plot_matched_profile(
                time_cut,
                line_density_values,
                beam,
                rf_well_cut=rf_well_cut_for_plot
                if len(wakefield_clones) > 0
                else None,
                total_well_cut=(time_cut, well_cut)
                if len(wakefield_clones) > 0
                else None,
            )

    def _plot_matched_profile(
        self,
        time_array,
        line_density_values,
        beam: BeamBaseClass,
        rf_well_cut=None,
        total_well_cut=None,
    ) -> None:
        """Requested (matched density) vs generated beam profile."""
        import matplotlib.pyplot as plt

        time_step = float(time_array[1] - time_array[0])
        requested_density = line_density_values / (
            line_density_values.sum() * time_step
        )  # probability density, in [1/s]
        with_wells = rf_well_cut is not None
        with AllowPlotting():
            if with_wells:
                fig, (ax, ax_well) = plt.subplots(
                    2, 1, num="AnalyticDistributionMatcher", sharex=True
                )
            else:
                fig, ax = plt.subplots(num="AnalyticDistributionMatcher")
            ax.hist(
                copy_to_cpu(beam.read_partial_dt()),
                bins=min(200, self._n_points_grid),
                density=True,
                alpha=0.5,
                color="C0",
                label="generated beam",
            )
            ax.plot(
                time_array,
                requested_density,
                color="C1",
                lw=2.0,
                label="requested (matched density)",
            )
            ax.set_ylabel("Line density [1/s]")
            ax.set_title(f"{self._distribution_type}: requested vs generated")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
            if with_wells:
                rf_time, rf_well = rf_well_cut
                total_time, total_well = total_well_cut
                ax_well.plot(
                    rf_time,
                    rf_well,
                    color="grey",
                    label="RF potential well",
                )
                ax_well.plot(
                    total_time,
                    total_well,
                    color="C3",
                    label="with induced potential",
                )
                ax_well.set_ylabel("Potential well [eV]")
                ax_well.legend(loc="upper right")
                ax_well.grid(alpha=0.3)
                ax_well.set_xlabel("Time [s]")
            else:
                ax.set_xlabel("Time [s]")
            fig.tight_layout()


def _profile_position(
    time_line_den: NumpyArray,
    line_density_values: NumpyArray,
    profile_centering: str,
) -> float:
    """
    Reference position of a line density for centering, in [s].

    ``"peak"`` is the BLonD 2 behaviour (position of the maximum);
    ``"barycenter"`` averages the samples above 60 % of the peak
    (operational measurement practice — robust for flat-topped or
    slightly noisy profiles).
    """
    if profile_centering == "peak":
        return float(time_line_den[np.argmax(line_density_values)])
    above = line_density_values >= 0.6 * float(line_density_values.max())
    return float(
        np.sum(time_line_den[above] * line_density_values[above])
        / np.sum(line_density_values[above])
    )


def _well_minimum_time(time_cut: NumpyArray, well_cut: NumpyArray) -> float:
    """
    Sub-sample time of the potential-well minimum, in [s].

    Three-point quadratic interpolation around the grid minimum. A
    grid-quantized minimum position would make the centering shift flip
    between adjacent samples each intensity iteration — a limit cycle
    that plateaus the fixed-point residual at the grid resolution
    instead of converging.
    """
    minimum_index = int(np.argmin(well_cut))
    if minimum_index == 0 or minimum_index == len(well_cut) - 1:
        return float(time_cut[minimum_index])
    value_left, value_center, value_right = well_cut[
        minimum_index - 1 : minimum_index + 2
    ]
    curvature = value_left - 2.0 * value_center + value_right
    if curvature <= 0.0:
        return float(time_cut[minimum_index])
    # Vertex offset in units of the local grid step, in [-1/2, 1/2]
    # (1/2 exactly when the minimum sample is duplicated).
    offset = 0.5 * (value_left - value_right) / curvature
    local_step = 0.5 * float(
        time_cut[minimum_index + 1] - time_cut[minimum_index - 1]
    )
    return float(time_cut[minimum_index]) + offset * local_step


class LineDensityMatcher(MatchingRoutine):
    r"""
    Matched single-bunch generation from a line density (Abel route).

    The BLonD 2 ``matched_from_line_density`` workflow: a line density
    — typically a **measured bunch profile**, or an analytic family —
    is centred in the analytic RF bucket and Abel-inverted over the
    potential well into the stationary distribution function
    :math:`F(H)` that reproduces it; the beam is then sampled from the
    resulting phase-space density.

    The input profile is always recentred onto the potential-well
    minimum (measured profiles are arbitrarily positioned). It is used
    as given otherwise: measured profiles are expected clean
    (baseline-subtracted up to a constant — the minimum is removed as
    in BLonD 2 — and low-noise; the Abel transform differentiates the
    profile, so noise is amplified and filtering is deliberately left
    to the caller).

    If the ring contains wakefields, the centering and the induced
    potential of the profile are iterated to self-consistency, with the
    same under-relaxation stabiliser as
    :class:`AnalyticDistributionMatcher` (absent in BLonD 2's
    line-density path).

    Parameters
    ----------
    n_macroparticles
        Number of macroparticles to generate.
    time_array
        Time coordinates of the input line density, in [s]
        (measured-profile mode; give together with
        ``line_density_values``).
    line_density_values
        Input line density at ``time_array`` (arbitrary normalization).
    line_density_type
        Analytic family mode (alternative to the measured arrays):
        ``"waterbag"``, ``"parabolic_amplitude"``, ``"parabolic_line"``,
        ``"binomial"``, ``"gaussian"`` or ``"cosine_squared"`` — see
        :func:`~blond.experimental.beam_preparation.analytic_distributions.line_density`.
    bunch_length
        Full bunch length of the analytic family, in [s]
        (:math:`4\sigma` for the gaussian); required with
        ``line_density_type``.
    exponent
        Binomial phase-space exponent :math:`\mu` for
        ``line_density_type="binomial"`` (the +1/2 line-density shift
        is applied internally).
    half_option
        Branch of the well used for the Abel inversion: ``"first"``
        (BLonD 2 default), ``"second"`` or ``"both"`` (average — robust
        for asymmetric profiles).
    n_points_abel
        Resolution of the Abel inversion per branch (BLonD 2 default
        1e4).
    profile_centering
        ``"peak"`` (BLonD 2: profile maximum onto the well minimum) or
        ``"barycenter"`` (mean of samples above 60 % of the peak —
        robust for flat-topped measured profiles).
    seed
        Random seed for the macroparticle sampling.
    n_points_grid
        Resolution of the internal time and energy grids (also the
        resolution of the smooth line-density profile driving the
        induced-voltage computation).
    dt_margin_fraction
        Frame margin as a fraction of the main RF period. Default
        (None): ``0.4`` when the ring has wakefields, ``0`` otherwise.
    maxiter_intensity_effects
        Maximum number of intensity-effect iterations.
    tolerance_potential_well
        Convergence threshold on the fixed-point residual of the
        induced potential, relative to the RF potential-well amplitude.
    relaxation_factor
        Fraction :math:`\alpha \in (0, 1]` of the induced-potential
        correction applied per iteration (see
        :class:`AnalyticDistributionMatcher`).
    allow_inner_buckets
        If True, wells split by the induced potential are accepted with
        a warning instead of raising.
    verbose
        If True, print matching diagnostics.
    plot
        If True, draw the input line density against the reconstructed
        density and the generated macroparticle profile (and the RF vs
        distorted well when intensity effects are active).

    Attributes
    ----------
    hamiltonian_coord
        Hamiltonian coordinates of the reconstruction, in [eV] (after
        run; ascending, 0 at the well minimum).
    distribution_values
        Abel-reconstructed :math:`F(H)` at ``hamiltonian_coord``.
    matched_bunch_length
        4-sigma rms bunch length of the reconstructed density, in [s].
    matched_bunch_position
        Mean position of the reconstructed density, in [s] (the
        profile is recentred onto the potential-well minimum).
    profile_reconstruction_error
        Maximum deviation between the normalized input and
        reconstructed line densities, relative to the input peak: how
        well the Abel closure reproduces the requested profile.
    n_intensity_iterations
        Number of induced-potential updates performed (0 when the ring
        has no wakefields).
    final_potential_well_error
        Last fixed-point residual of the induced potential, relative to
        the RF potential-well amplitude (None when the ring has no
        wakefields).

    Examples
    --------
    >>> from blond import Simulation
    >>> from blond.experimental.beam_preparation.analytic_matcher import (
    ...     LineDensityMatcher,
    ... )
    >>> simulation = Simulation( ... )
    >>> simulation.prepare_beam(
    ...     beam= ... ,
    ...     preparation_routine=LineDensityMatcher(
    ...         n_macroparticles=1e6,
    ...         time_array=measured_time,
    ...         line_density_values=measured_profile,
    ...         half_option="both",
    ...     ),
    ... )
    """

    def __init__(
        self,
        n_macroparticles: int | float,
        time_array: NumpyArray | None = None,
        line_density_values: NumpyArray | None = None,
        line_density_type: str | None = None,
        bunch_length: float | None = None,
        exponent: float | None = None,
        half_option: Literal["first", "second", "both"] = "first",
        n_points_abel: int = 10_000,
        profile_centering: Literal["peak", "barycenter"] = "peak",
        seed: int | None = 0,
        n_points_grid: int = 1000,
        dt_margin_fraction: float | None = None,
        maxiter_intensity_effects: int = 100,
        tolerance_potential_well: float = 1e-6,
        relaxation_factor: float = 1.0,
        allow_inner_buckets: bool = False,
        verbose: bool = False,
        plot: bool = False,
    ) -> None:
        super().__init__()
        measured_mode = (time_array is not None) and (
            line_density_values is not None
        )
        family_mode = (line_density_type is not None) and (
            bunch_length is not None
        )
        if measured_mode == family_mode:
            raise ValueError(
                "Specify exactly one input: measured arrays "
                "(`time_array` + `line_density_values`) or an analytic "
                "family (`line_density_type` + `bunch_length`)."
            )
        if half_option not in ("first", "second", "both"):
            raise ValueError(
                f"Unknown {half_option=}; use 'first', 'second' or 'both'."
            )
        if profile_centering not in ("peak", "barycenter"):
            raise ValueError(
                f"Unknown {profile_centering=}; use 'peak' or 'barycenter'."
            )
        if not 0.0 < relaxation_factor <= 1.0:
            raise ValueError(
                f"relaxation_factor must be in (0, 1], "
                f"got {relaxation_factor}."
            )
        if measured_mode:
            input_time = np.asarray(time_array, dtype=float).copy()
            input_line_density = np.asarray(
                line_density_values, dtype=float
            ).copy()
            assert input_time.shape == input_line_density.shape, (
                f"{input_time.shape=} must match {input_line_density.shape=}"
            )
            assert np.all(np.diff(input_time) > 0.0), (
                "`time_array` must be strictly increasing."
            )
            self._input_time = input_time
            self._input_line_density = input_line_density
        else:
            self._input_time = None
            self._input_line_density = None
        self._n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self._line_density_type = line_density_type
        self._bunch_length = bunch_length
        self._exponent = exponent
        self._half_option: Literal["first", "second", "both"] = half_option
        self._n_points_abel = int(n_points_abel)
        self._profile_centering = profile_centering
        self._seed = seed
        self._n_points_grid = int(n_points_grid)
        self._dt_margin_fraction = dt_margin_fraction
        self._maxiter_intensity_effects = int(maxiter_intensity_effects)
        self._tolerance_potential_well = tolerance_potential_well
        self._relaxation_factor = relaxation_factor
        self._allow_inner_buckets = allow_inner_buckets
        self._verbose = verbose
        self._plot = plot

        #: Hamiltonian coordinates of the reconstruction, in [eV].
        self.hamiltonian_coord: NumpyArray | None = None
        #: Abel-reconstructed F(H) at ``hamiltonian_coord``.
        self.distribution_values: NumpyArray | None = None
        #: 4-sigma rms bunch length of the reconstructed density, [s].
        self.matched_bunch_length: float | None = None
        #: Mean position of the reconstructed density, in [s].
        self.matched_bunch_position: float | None = None
        #: Max input-vs-reconstructed deviation, rel. to input peak.
        self.profile_reconstruction_error: float | None = None
        #: Number of induced-potential updates performed.
        self.n_intensity_iterations: int = 0
        #: Last fixed-point residual of the induced potential.
        self.final_potential_well_error: float | None = None
        #: Fixed-point residual per intensity iteration (after run).
        self.intensity_residuals: list[float] = []

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with matched macroparticles.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beams.Beam` object.
        """
        super().prepare_beam(simulation=simulation, beam=beam)

        # --- machine parameters and RF potential well -----------------
        params = _machine_parameters(simulation, beam)
        eom_factor_dE = params["eom_factor_dE"]
        dt_margin_fraction = _resolve_dt_margin_fraction(
            self._dt_margin_fraction, _ring_has_wakefields(simulation)
        )
        time_array = bucket_time_array(
            params["omega_rf"],
            n_points=self._n_points_grid,
            dt_margin_fraction=dt_margin_fraction,
        )
        rf_potential_raw = rf_potential_well(
            time_array,
            _total_rf_voltage(simulation, time_array),
            charge=params["charge"],
            t_rev=params["t_rev"],
            eta_0=params["eta_0"],
            energy_gain_per_turn=params["energy_gain_per_turn"],
            subtract_min=False,
        )

        # --- input line density on its own time axis ------------------
        if self._input_time is not None:
            assert self._input_line_density is not None
            time_line_den = self._input_time.copy()
            line_density_values = self._input_line_density.copy()
            # Constant-baseline removal, as in BLonD 2.
            line_density_values -= line_density_values.min()
        else:
            assert self._line_density_type is not None
            assert self._bunch_length is not None
            time_line_den = np.linspace(
                float(time_array[0]),
                float(time_array[-1]),
                self._n_points_abel,
            )
            line_density_values = line_density(
                time_line_den,
                self._line_density_type,
                self._bunch_length,
                bunch_position=0.5 * float(time_array[0] + time_array[-1]),
                exponent=self._exponent,
            )

        # --- centering + intensity-effect iteration -------------------
        # The profile is ALWAYS recentred onto the well minimum
        # (measured profiles are arbitrarily positioned); with
        # wakefields the well moves with the induced potential, so
        # centering and induced potential iterate together (BLonD 2)
        # with the under-relaxation stabiliser.
        induced_potential = np.zeros_like(time_array)
        wakefield_clones, smooth_profile = clone_wakefields_on_smooth_profile(
            simulation, time_array
        )
        rf_amplitude = None
        rf_well_cut_for_plot = None
        residual = None
        self.n_intensity_iterations = 0
        self.final_potential_well_error = None
        self.intensity_residuals = []
        converged = False

        for iteration in range(self._maxiter_intensity_effects + 1):
            total_potential_raw = rf_potential_raw + induced_potential
            time_cut, well_cut = cut_potential_well(
                time_array,
                total_potential_raw,
                allow_inner_buckets=self._allow_inner_buckets,
            )
            if rf_amplitude is None:
                rf_amplitude = float(well_cut.max() - well_cut.min())
                rf_well_cut_for_plot = (time_cut, well_cut)

            time_line_den = time_line_den - (
                _profile_position(
                    time_line_den,
                    line_density_values,
                    self._profile_centering,
                )
                - _well_minimum_time(time_cut, well_cut)
            )

            if len(wakefield_clones) == 0:
                break

            if converged or iteration == self._maxiter_intensity_effects:
                if not converged:
                    warnings.warn(
                        "Intensity-effect matching did not converge in "
                        f"{self._maxiter_intensity_effects} iterations "
                        f"(residual {residual:.2e}, tolerance "
                        f"{self._tolerance_potential_well:.2e}). "
                        "Consider a smaller `relaxation_factor`.",
                        UserWarning,
                        stacklevel=2,
                    )
                break

            line_density_frame = np.interp(
                time_array,
                time_line_den,
                line_density_values,
                left=0.0,
                right=0.0,
            )
            induced_voltage = induced_voltage_from_line_density(
                wakefield_clones,
                smooth_profile,
                line_density_frame,
                beam,
            )
            induced_potential_new = rf_potential_well(
                time_array,
                induced_voltage,
                charge=params["charge"],
                t_rev=params["t_rev"],
                eta_0=params["eta_0"],
                subtract_min=False,
            )
            residual = float(
                np.sqrt(
                    np.mean((induced_potential_new - induced_potential) ** 2)
                )
                / rf_amplitude
            )
            converged = residual < self._tolerance_potential_well
            induced_potential = (
                1.0 - self._relaxation_factor
            ) * induced_potential + (
                self._relaxation_factor * induced_potential_new
            )
            self.n_intensity_iterations = iteration + 1
            self.final_potential_well_error = residual
            self.intensity_residuals.append(residual)
            if self._verbose:
                print(
                    "[LineDensityMatcher] intensity iteration "
                    f"{iteration + 1:3d}: residual {residual:.3e} "
                    f"(tolerance {self._tolerance_potential_well:.1e}, "
                    f"relaxation {self._relaxation_factor})"
                )

        # --- Abel inversion on the profile's own sampling -------------
        # The inversion runs on the input time axis restricted to the
        # cut well (BLonD 2 kept the profile resolution and
        # interpolated the potential onto it); `n_points_abel` then
        # refines each branch inside the transform.
        abel_support = (time_line_den >= time_cut[0]) & (
            time_line_den <= time_cut[-1]
        )
        abel_time = time_line_den[abel_support]
        if len(abel_time) < 8:
            raise ValueError(
                "The input line density has fewer than 8 samples inside "
                "the RF bucket — check its time axis (units: seconds)."
            )
        self.hamiltonian_coord, self.distribution_values = (
            distribution_from_line_density(
                abel_time,
                line_density_values[abel_support],
                np.interp(abel_time, time_cut, well_cut),
                eom_factor_dE=eom_factor_dE,
                half_option=self._half_option,
                n_points_abel=self._n_points_abel,
            )
        )

        # --- phase-space density on the 2D grid -----------------------
        time_grid, deltaE_grid, hamilton_2D = hamiltonian_grid(
            time_cut,
            well_cut,
            eom_factor_dE=eom_factor_dE,
            n_points_deltaE=self._n_points_grid,
            allow_inner_buckets=self._allow_inner_buckets,
        )
        density = np.interp(
            hamilton_2D - float(well_cut.min()),
            self.hamiltonian_coord,
            self.distribution_values,
        )
        # Outside the tabulated range and the separatrix the density
        # is unknown/unphysical: zero it (np.interp would extend the
        # edge value as a constant).
        density[
            hamilton_2D - float(well_cut.min())
            > float(self.hamiltonian_coord[-1])
        ] = 0.0
        density = np.where(hamilton_2D <= float(well_cut.max()), density, 0.0)
        density /= density.sum()
        reconstructed_line_density = density.sum(axis=0)

        # --- diagnostics ----------------------------------------------
        total = reconstructed_line_density.sum()
        mean_time = (reconstructed_line_density * time_cut).sum() / total
        self.matched_bunch_position = float(mean_time)
        self.matched_bunch_length = float(
            4.0
            * np.sqrt(
                (
                    reconstructed_line_density * (time_cut - mean_time) ** 2
                ).sum()
                / total
            )
        )
        input_on_cut = np.interp(
            time_cut,
            time_line_den,
            line_density_values,
            left=0.0,
            right=0.0,
        )
        input_normalized = input_on_cut / input_on_cut.sum()
        self.profile_reconstruction_error = float(
            np.max(np.abs(reconstructed_line_density - input_normalized))
            / np.max(input_normalized)
        )
        if self._verbose:
            source = (
                "measured profile"
                if self._input_time is not None
                else f"'{self._line_density_type}' family"
            )
            print(
                "[LineDensityMatcher] "
                f"{source}, half_option='{self._half_option}': "
                f"matched 4-sigma rms bunch length "
                f"{self.matched_bunch_length:.4e} s at "
                f"{self.matched_bunch_position:.4e} s, profile "
                "reconstruction error "
                f"{self.profile_reconstruction_error:.2%}, "
                f"{self.n_intensity_iterations} intensity iteration(s)"
            )

        # --- sampling -------------------------------------------------
        populate_beam(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=density,
            n_macroparticles=self._n_macroparticles,
            seed=self._seed,
        )

        if self._plot:
            self._plot_matched_profile(
                time_cut,
                input_normalized,
                reconstructed_line_density,
                beam,
                rf_well_cut=rf_well_cut_for_plot
                if len(wakefield_clones) > 0
                else None,
                total_well_cut=(time_cut, well_cut)
                if len(wakefield_clones) > 0
                else None,
            )

    def _plot_matched_profile(
        self,
        time_array,
        input_normalized,
        reconstructed_line_density,
        beam: BeamBaseClass,
        rf_well_cut=None,
        total_well_cut=None,
    ) -> None:
        """Input vs reconstructed line density vs generated beam."""
        import matplotlib.pyplot as plt

        # Two different scalings on purpose: the beam histogram and
        # the reconstructed density are both integral-normalized
        # (probability densities, [1/s]) so the sampling quality is
        # judged fairly — peak normalization would bias the histogram
        # low, its maximum being inflated by binning noise. The input
        # profile is instead anchored to the PEAK of the reconstructed
        # density: the Abel reconstruction reproduces the input exactly
        # on the inverted branch up to an amplitude scale and both peak
        # at the well minimum, so this overlays the matched branch
        # exactly (e.g. the full first half for half_option="first") —
        # which integral normalization would hide whenever the other
        # branch differs (distorted wells).
        time_step = float(time_array[1] - time_array[0])
        reconstructed_density = reconstructed_line_density / (
            reconstructed_line_density.sum() * time_step
        )  # probability density, in [1/s]
        input_density = input_normalized * (
            reconstructed_density.max() / input_normalized.max()
        )
        with_wells = rf_well_cut is not None
        with AllowPlotting():
            if with_wells:
                fig, (ax, ax_well) = plt.subplots(
                    2, 1, num="LineDensityMatcher", sharex=True
                )
            else:
                fig, ax = plt.subplots(num="LineDensityMatcher")
            ax.hist(
                copy_to_cpu(beam.read_partial_dt()),
                bins=min(200, self._n_points_grid),
                density=True,
                alpha=0.5,
                color="C0",
                label="generated beam",
            )
            ax.plot(
                time_array,
                input_density,
                color="k",
                lw=2.0,
                label="input line density (peak-anchored)",
            )
            ax.plot(
                time_array,
                reconstructed_density,
                color="C1",
                ls="--",
                lw=1.5,
                label="reconstructed density",
            )
            ax.set_ylabel("Line density [1/s]")
            ax.set_title("LineDensityMatcher: input vs reconstructed")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
            if with_wells:
                rf_time, rf_well = rf_well_cut
                total_time, total_well = total_well_cut
                ax_well.plot(
                    rf_time,
                    rf_well,
                    color="grey",
                    label="RF potential well",
                )
                ax_well.plot(
                    total_time,
                    total_well,
                    color="C3",
                    label="with induced potential",
                )
                ax_well.set_ylabel("Potential well [eV]")
                ax_well.legend(loc="upper right")
                ax_well.grid(alpha=0.3)
                ax_well.set_xlabel("Time [s]")
            else:
                ax.set_xlabel("Time [s]")
            fig.tight_layout()
