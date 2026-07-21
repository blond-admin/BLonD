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
reproducing the BLonD 2 ``matched_from_distribution_function`` for the
single-bunch case without intensity effects (the intensity iteration is
an upcoming step).
"""

from __future__ import annotations

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
from blond.experimental.beam_preparation.analytic_potential_well import (
    bucket_time_array,
    rf_potential_well,
)
from blond.experimental.beam_preparation.analytic_well_cut import (
    cut_potential_well,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.iterables_ import all_equal

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

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
        Resolution of the internal time and energy grids.
    verbose
        If True, print matching diagnostics.

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
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if (bunch_length is None) == (emittance is None):
            raise ValueError(
                "Specify exactly one of `bunch_length` or `emittance`."
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
        self._verbose = verbose

        #: Fitted distribution size parameter X0, in [eV] (after run).
        self.fitted_x_0: float | None = None
        #: 4-sigma rms bunch length of the matched density, in [s].
        self.matched_bunch_length: float | None = None

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
        energy_gain_per_turn = (
            simulation.magnetic_cycle.get_target_total_energy(
                turn_i=0,
                section_i=0,
                reference_time=0,
                particle_type=beam.particle_type,
            )
            - beam.reference.total_energy
        )

        # --- potential well from the actual RF waveform ---------------
        time_array = bucket_time_array(omega_rf, n_points=self._n_points_grid)
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
        potential_well = rf_potential_well(
            time_array,
            total_voltage,
            charge=beam.particle_type.charge,
            t_rev=simulation.get_t_rev_init(),
            eta_0=eta_0,
            energy_gain_per_turn=energy_gain_per_turn,
        )
        time_cut, well_cut = cut_potential_well(time_array, potential_well)

        # --- Hamiltonian grid and distribution sizing -----------------
        eom_factor_dE = calc_eom_factor_dE(
            eta_0=eta_0,
            beta=beam.reference.beta,
            total_energy=beam.reference.total_energy,
        )
        time_grid, deltaE_grid, hamilton_2D = hamiltonian_grid(
            time_cut,
            well_cut,
            eom_factor_dE=eom_factor_dE,
            n_points_deltaE=self._n_points_grid,
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
                verbose=self._verbose,
            )
        else:
            sorted_hamiltonian, sorted_action = action_from_potential_well(
                time_cut, well_cut, eom_factor_dE=eom_factor_dE
            )
            x_0 = hamiltonian_from_emittance(
                self._emittance, sorted_hamiltonian, sorted_action
            )

        density = distribution_function(
            hamilton_2D, self._distribution_type, x_0, self._exponent
        )
        density = np.where(inside_bucket_mask, density, 0.0)
        density /= density.sum()

        # --- diagnostics ----------------------------------------------
        line_density_values = density.sum(axis=0)
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
                f"bunch length {self.matched_bunch_length:.4e} s"
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
