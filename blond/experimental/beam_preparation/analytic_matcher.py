# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Analytic single-bunch matching routine (BLonD 2 style).

Assembles the analytic building blocks — RF potential well, separatrix
cut, 2D Hamiltonian, distribution families and bunch-length/emittance
targeting — into a :class:`~blond.beam_preparation.base.MatchingRoutine`
usable with :meth:`~blond.core.simulation.simulation.Simulation.prepare_beam`,
reproducing the BLonD 2 ``matched_from_distribution_function`` workflow,
including the intensity-effect iteration: if the ring contains
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
from blond.experimental.beam_preparation.analytic_action import (
    action_from_potential_well,
    hamiltonian_from_emittance,
)
from blond.experimental.beam_preparation.analytic_distributions import (
    distribution_function,
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
        from blond import MultiHarmonicRFStation
        from blond.physics.cavities import SingleHarmonicRFStation
        from blond.physics.drifts import DriftSimple

        super().prepare_beam(simulation=simulation, beam=beam)

        # --- machine parameters (shared helpers, no third variant) ----
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
        charge = beam.particle_type.charge
        t_rev = simulation.get_t_rev_init()
        energy_gain_per_turn = (
            simulation.magnetic_cycle.get_target_total_energy(
                turn_i=0,
                section_i=0,
                reference_time=0,
                particle_type=beam.particle_type,
            )
            - beam.reference.total_energy
        )
        eom_factor_dE = calc_eom_factor_dE(
            eta_0=eta_0,
            beta=beam.reference.beta,
            total_energy=beam.reference.total_energy,
        )

        # --- RF potential well from the actual RF waveform ------------
        from blond.physics.impedances.base import WakeField

        ring_has_wakefields = (
            len(
                simulation.ring.elements.get_elements(
                    WakeField, recursive=False
                )
            )
            > 0
        )
        if self._dt_margin_fraction is not None:
            dt_margin_fraction = self._dt_margin_fraction
        elif ring_has_wakefields:
            # Legacy BLonD 2 default: the induced potential shifts and
            # tilts the bucket, the margin keeps it inside the frame.
            dt_margin_fraction = 0.4
        else:
            dt_margin_fraction = 0.0
        time_array = bucket_time_array(
            omega_rf,
            n_points=self._n_points_grid,
            dt_margin_fraction=dt_margin_fraction,
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
        rf_potential_raw = rf_potential_well(
            time_array,
            total_voltage,
            charge=charge,
            t_rev=t_rev,
            eta_0=eta_0,
            energy_gain_per_turn=energy_gain_per_turn,
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
