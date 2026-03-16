# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.beam_preparation.base import MatchingRoutine
from blond.core.helpers import int_from_float_with_warning
from blond.generals.distributed.helpers import mpi_local_size
from blond.physics.cavities import SingleHarmonicRFStation
from blond.physics.drifts import DriftSimple
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    _SynchrotronRadiationTracker,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
        SynchrotronRadiationMaster,
    )


class SynchrotronRadiationMatcher(MatchingRoutine):
    """
    Beam matching routine to generate a matched distribution with synchrotron radiation.

    The expected layout for the ring is
    [SingleHarmonicRFStation, SynchrotronRadiationTracker, DriftSimple]

    Parameters
    ----------
    synchrotron_radiation_master
        The :class:`~blond.physics.synchrotron_radiation.synchrotron_radiation_master.SynchrotronRadiationMaster`
        object handling the synchrotron radiation parameters.
    n_macroparticles
        Number of macroparticles to be generated.
    seed
        Random seed parameter.

    Examples
    --------
    >>> from blond import Simulation
    >>> from blond.experimental.beam_preparation.synchrotron_radiation_matcher import SynchrotronRadiationMatcher
    >>> simulation = Simulation( ... )
    >>> simulation.prepare_beam(
    ...     beam= ... ,
    ...     preparation_routine=SynchrotronRadiationMatcher(
    ...         synchrotron_radiation_master= ... ,
    ...         n_macroparticles=100000,
    ...     ),
    ... )
    """

    def __init__(
        self,
        synchrotron_radiation_master: SynchrotronRadiationMaster,
        n_macroparticles: int | float,
        seed: int | None = 0,
    ) -> None:
        super().__init__()

        self._sr_master = synchrotron_radiation_master

        self._n_macroparticles_local = mpi_local_size(
            int_from_float_with_warning(
                n_macroparticles, warning_stacklevel=2
            ),
            warning_hint="n_macroparticles",
        )

        self._seed = seed

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with macro-particles.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beam.Beam` object.
        """

        # Check if the lattice is comparable to expectation
        # To be extented for many SR+Drift

        expected_elements = [
            SingleHarmonicRFStation,
            _SynchrotronRadiationTracker,
            DriftSimple,
        ]

        element_error_message = (
            "The SynchrotronRadiationMatcher function "
            + "is presently only implemented for the lattice [Kick, SR, Drift]"
        )

        # TODO: consider many SR+Drift sections or Drift+SR
        n_sections = 1  # Hard coded for now to be taken from the Ring layout

        if len(simulation.ring.elements.elements) != len(expected_elements):
            raise ValueError(element_error_message)
        for idx_element, element in enumerate(
            simulation.ring.elements.elements
        ):
            if not isinstance(element, expected_elements[idx_element]):
                raise ValueError(element_error_message)

        # Prepare the beam and other objects to get base parameters
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        all_base_params = self.get_all_base_params(
            simulation=simulation,
            beam=beam,
        )

        covariance_matrix_scaled, scaling_factor = (
            self.compute_covariance_matrix(all_base_params=all_base_params)
        )

        # Generate the random distribution
        # TODO: assess usage of mpi_aware_random_generator_cpu
        dt_distrib, dE_distrib = (
            np.random.default_rng(seed=self._seed)
            .multivariate_normal(
                [0, 0],
                covariance_matrix_scaled,
                size=self._n_macroparticles_local,
            )
            .T
        )

        # Scale the distribution
        dt_distrib *= np.sqrt(scaling_factor)
        dE_distrib *= np.sqrt(1 / scaling_factor)

        # Compute the expected stable phase offset
        dt_center = all_base_params["phi_s"] / all_base_params["omega_rf"]
        dE_center = -all_base_params["energy"] * sawtooth_factor(n_sections)

        # Position the beam in the stable point in (time, energy)
        dt_distrib += dt_center
        dE_distrib += dE_center

        beam.setup_beam(
            dt=dt_distrib,
            dE=dE_distrib,
            mpi_mode="all-ranks",  # because the random generator above is MPI aware
        )

    def get_all_base_params(
        self, simulation: Simulation, beam: BeamBaseClass
    ) -> dict[str, float]:
        """
        Get the parameters to compute the covariance matrix.

        This includes: energy, charge, rf_voltage, energy_loss_per_turn,
        sigma_dE, beta, eta_0, t_rev, t_rf, omega_rf, phi_s.

        Parameters
        ----------
            simulation (Simulation)
                `Simulation` context manager.
            beam (BeamBaseClass)
                Simulation :class:`~blond.core.beam.beam.Beam` object.

        Returns
        -------
            dict[str, float]
                All relevant parameters for the `compute_covariance_matrix` function.
        """

        ring = simulation.ring

        rf_system = ring.elements.elements[0]
        drift = ring.elements.elements[-1]

        # Get the parameters from the simulation
        self._sr_master.compute_synchrotron_radiation_parameters(
            ring,
            beam,
        )

        energy_loss_per_turn = self._sr_master.energy_loss_per_turn
        sigma_dE = self._sr_master.natural_energy_spread

        beta = beam.reference.beta
        eta_0 = drift.eta_0(beam.reference.gamma)
        t_rev = simulation.get_t_rev_init()
        t_rf = rf_system.calc_main_harmonic_t_rf(beta, ring.circumference)
        omega_rf = 2 * np.pi / t_rf

        # NB: already factors in the synchrotron radiation loss!
        phi_s = rf_system.calc_phi_s_main_harmonic(beam)

        return {
            "energy": beam.reference.total_energy,
            "charge": beam.particle_type.charge,
            "rf_voltage": rf_system.voltage,
            "energy_loss_per_turn": energy_loss_per_turn,
            "sigma_dE": sigma_dE,
            "beta": beta,
            "eta_0": eta_0,
            "t_rev": t_rev,
            "t_rf": t_rf,
            "omega_rf": omega_rf,
            "phi_s": phi_s,
        }

    def compute_covariance_matrix(
        self, all_base_params: dict
    ) -> tuple[np.ndarray, float]:
        """
        Compute the covariance matrix (Courant-Snyder parameters) representing the
        expected tilted trajectories of the particles in phase space.

        The input dict should contain : energy, charge, rf_voltage,
        energy_loss_per_turn, sigma_dE, beta, eta_0, t_rev, t_rf, omega_rf, phi_s.

        Parameters
        ----------
            all_base_params (dict)
                All relevant parameters for the `get_all_base_params` function.

        Returns
        -------
            covariance_matrix_scaled
                The Courant-Snyder parameters for the kick drift
        """

        # Define the Kick Drift parameters
        kick_param = (
            -all_base_params["charge"]
            * all_base_params["rf_voltage"]
            * all_base_params["omega_rf"]
            * np.cos(all_base_params["phi_s"])
        )
        drift_param = (
            -all_base_params["t_rev"]
            * all_base_params["eta_0"]
            / (all_base_params["beta"] ** 2.0 * all_base_params["energy"])
        )

        # Compute the Courant-Snyder parameters for the kick drift
        synchrotron_tune = (
            np.arcsin(np.sqrt(-(drift_param * kick_param) / 4)) / np.pi
        )
        mu = np.sign(drift_param) * 2 * np.pi * synchrotron_tune
        beta_cs = drift_param / np.sin(mu)
        gamma_cs = -kick_param / np.sin(mu)
        alpha_cs = np.sign(drift_param) * np.tan(np.pi * synchrotron_tune)

        # Get the longitudinal emittance
        epsilon_rms_tilted = (
            all_base_params["sigma_dE"] * all_base_params["energy"]
        ) ** 2.0 / gamma_cs

        # Get the covariance matrix
        covariance_matrix = epsilon_rms_tilted * np.array(
            [[beta_cs, -alpha_cs], [-alpha_cs, gamma_cs]]
        )

        # Get the "scaled" covariance matrix
        # (NB: multivariate_normal doesn't like big order of magnitude values)
        scaling_factor = 10 ** np.floor(np.log10(np.abs(beta_cs)))
        covariance_matrix_scaled = np.array(covariance_matrix)
        covariance_matrix_scaled[0, 0] /= scaling_factor
        covariance_matrix_scaled[1, 1] *= scaling_factor

        return covariance_matrix_scaled, scaling_factor


def sawtooth_factor(n_sections, order="sr+drift") -> float:
    """The sawtooth factor is the fraction of the total energy loss due to
    synchrotron radiation at which the synchronous energy is sitting right
    before the RF cavity with a single RF station (for the one-turn map
    being (RF + [Drift + SR] * n_sections))

    This will depend on the layout and needs to be generalized.
    """

    if order == "sr+drift":
        return (n_sections - 1) / (2 * n_sections)

    if order == "drift+sr":
        return (n_sections + 1) / (2 * n_sections)

    raise ValueError("The order should either be sr+drift or drift+sr")
