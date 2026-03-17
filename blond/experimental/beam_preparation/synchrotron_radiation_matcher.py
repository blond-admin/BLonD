# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# References: Alexandre Lasheen

from __future__ import annotations

from dataclasses import dataclass
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
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
        SynchrotronRadiationMaster,
    )


@dataclass
class _MatcherAcceleratorParameters:
    energy: float
    charge: float
    rf_voltage: float
    energy_loss_per_turn: float
    sigma_dE: float
    beta: float
    eta_0: float
    t_rev: float
    t_rf: float
    omega_rf: float
    phi_s: float
    phi_rf: float


class SynchrotronRadiationMatcher(MatchingRoutine):
    """
    Beam matching routine to generate a matched distribution with synchrotron radiation.

    The expected layout for the ring is
    [`SingleHarmonicRFStation`, `_SynchrotronRadiationTracker`, `DriftSimple`]..

    The case with multiple RF stations is not covered.

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
        check_ring_layout: bool = True,
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
        if check_ring_layout:
            self.check_ring_layout(simulation)

        n_sections = int((simulation.ring.elements.n_elements - 1) / 2)

        # Prepare the beam and other objects to get base parameters
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        matcher_parameters = self.get_matcher_parameters(
            simulation=simulation,
            beam=beam,
        )

        covariance_matrix = self.compute_covariance_matrix(
            matcher_parameters=matcher_parameters
        )

        self.generate_distribution(
            beam=beam,
            matcher_parameters=matcher_parameters,
            covariance_matrix=covariance_matrix,
            n_sections=n_sections,
            order="sr+drift",  # to be extended when SR allows for drift+sr
        )

    def check_ring_layout(self, simulation: Simulation) -> None:
        """
        Check if the lattice is comparable to expectation.
        """

        assert simulation.ring.n_rf_stations == 1, (
            "The case with multiple RF stations is not covered."
        )

        n_sections = int((simulation.ring.elements.n_elements - 1) / 2)
        expected_elements = [SingleHarmonicRFStation] + [
            _SynchrotronRadiationTracker,
            DriftSimple,
        ] * n_sections

        element_error_message = (
            "The `SynchrotronRadiationMatcher` function "
            + "is presently only implemented for the lattice "
            + "[`SingleHarmonicRFStation`] "
            + "+ [`_SynchrotronRadiationTracker`, `DriftSimple`] * n_sections"
        )

        if len(simulation.ring.elements.elements) != len(expected_elements):
            raise ValueError(element_error_message)
        for expected, actual in zip(
            expected_elements, simulation.ring.elements.elements
        ):
            if not isinstance(actual, expected):
                raise ValueError(element_error_message)

    def get_matcher_parameters(
        self, simulation: Simulation, beam: BeamBaseClass
    ) -> _MatcherAcceleratorParameters:
        """
        Get the parameters to compute the covariance matrix.

        This includes: energy, charge, rf_voltage, energy_loss_per_turn,
        sigma_dE, beta, eta_0, t_rev, t_rf, omega_rf, phi_s.

        Parameters
        ----------
            simulation
                `Simulation` context manager.
            beam
                Simulation :class:`~blond.core.beam.beam.Beam` object.

        Returns
        -------
            matcher_parameters
                All relevant parameters for the `compute_covariance_matrix` function.
        """

        ring = simulation.ring

        rf_system = simulation.ring.elements.get_element(
            SingleHarmonicRFStation
        )
        drift = simulation.ring.elements.get_elements(DriftSimple)[0]

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
        omega_rf = rf_system.calc_omega_rf_design(beta, ring.circumference)
        phi_rf = rf_system.phi_rf_design

        # NB: already factors in the synchrotron radiation loss!
        phi_s = rf_system.calc_phi_s_main_harmonic(beam)

        return _MatcherAcceleratorParameters(
            energy=beam.reference.total_energy,
            charge=beam.particle_type.charge,
            rf_voltage=rf_system.voltage,
            energy_loss_per_turn=energy_loss_per_turn,
            sigma_dE=sigma_dE,
            beta=beta,
            eta_0=eta_0,
            t_rev=t_rev,
            t_rf=t_rf,
            omega_rf=omega_rf,
            phi_s=phi_s,
            phi_rf=phi_rf,
        )

    def compute_covariance_matrix(
        self, matcher_parameters: _MatcherAcceleratorParameters
    ) -> NumpyArray:
        """
        Compute the covariance matrix for tilted phase space trajectories.

        The covariance matrix (Courant-Snyder parameters) is obtained assuming
        linear longitudinal maps.

        Parameters
        ----------
            matcher_parameters
                All relevant parameters from the `get_matcher_parameters` function.

        Returns
        -------
            covariance_matrix
                The Courant-Snyder parameters for the kick drift
        """

        # Define the Kick Drift parameters
        kick_param = (
            -matcher_parameters.charge
            * matcher_parameters.rf_voltage
            * matcher_parameters.omega_rf
            * np.cos(matcher_parameters.phi_s)
        )
        drift_param = (
            -matcher_parameters.t_rev
            * matcher_parameters.eta_0
            / (matcher_parameters.beta**2.0 * matcher_parameters.energy)
        )

        # Compute the Courant-Snyder parameters for the kick drift
        synchrotron_tune = (
            np.arcsin(np.sqrt(-(drift_param * kick_param) / 4)) / np.pi
        )
        mu = np.sign(drift_param) * 2 * np.pi * synchrotron_tune
        beta_cs = drift_param / np.sin(mu)
        gamma_cs = -kick_param / np.sin(mu)
        alpha_cs = np.sign(drift_param) * np.tan(np.pi * synchrotron_tune)

        # Get the covariance matrix
        covariance_matrix = np.array(
            [[beta_cs, -alpha_cs], [-alpha_cs, gamma_cs]]
        )

        return covariance_matrix

    def generate_distribution(
        self,
        beam: BeamBaseClass,
        matcher_parameters: _MatcherAcceleratorParameters,
        covariance_matrix: NumpyArray,
        n_sections: int,
        order: Literal["sr+drift", "drift+sr"],
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Generate a random multivariate normal particle distribution following the
        covariance matrix.

        Parameters
        ----------
            beam
                Simulation :class:`~blond.core.beam.beam.Beam` object.
            matcher_parameters
                All relevant parameters from the `get_matcher_parameters` function.
            covariance_matrix
                The Courant-Snyder parameters for the kick drift as output from `compute_covariance_matrix`.
            n_sections
                Number of [Drift, SR] or [SR, Drift] sections in the ring.
            order
                The order of the [Drift, SR] or [SR, Drift] sections in the ring.
                The expected input is "sr+drift" or "drift+sr".

        Returns
        -------
            dt_distrib, dE_distrib
                The generated particle distribution in (dt, dE)
                NB: the beam distribution is already passed to the `Beam` object
                at that stage.
        """

        # Get the "scaled" covariance matrix
        # (NB: multivariate_normal doesn't like big order of magnitude values)
        scaling_factor = 10 ** np.floor(
            np.log10(np.abs(covariance_matrix[0, 0]))
        )
        covariance_matrix_scaled = np.array(covariance_matrix)
        covariance_matrix_scaled[0, 0] /= scaling_factor
        covariance_matrix_scaled[1, 1] *= scaling_factor

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

        # Get the longitudinal emittance
        epsilon_rms_tilted = (
            matcher_parameters.sigma_dE * matcher_parameters.energy
        ) ** 2.0 / covariance_matrix[1, 1]

        # Scale the distribution
        dt_distrib *= np.sqrt(epsilon_rms_tilted * scaling_factor)
        dE_distrib *= np.sqrt(epsilon_rms_tilted / scaling_factor)

        # Compute the expected stable phase offset
        dt_center = (
            matcher_parameters.phi_s - matcher_parameters.phi_rf
        ) / matcher_parameters.omega_rf
        dE_center = -matcher_parameters.energy_loss_per_turn * sawtooth_factor(
            n_sections, order
        )

        # Position the beam in the stable point in (time, energy)
        dt_distrib += dt_center - np.mean(dt_distrib)
        dE_distrib += dE_center - np.mean(dE_distrib)

        # Setup the beam
        beam.setup_beam(
            dt=dt_distrib,
            dE=dE_distrib,
            mpi_mode="all-ranks",  # To be checked
        )

        return dt_distrib, dE_distrib


def sawtooth_factor(
    n_sections: int, order: Literal["sr+drift", "drift+sr"]
) -> float:
    """The sawtooth factor is the fraction of the total energy loss due to
    synchrotron radiation at which the synchronous energy is sitting right
    before the RF cavity with a single RF station (for the one-turn map
    being (RF + [Drift + SR] * n_sections))

    This will depend on the layout and needs to be generalized.
    """

    if order == "sr+drift":
        factor = (n_sections - 1) / (2 * n_sections)

    elif order == "drift+sr":
        factor = (n_sections + 1) / (2 * n_sections)

    else:
        raise ValueError(
            f"The order should either be 'sr+drift' or 'drift+sr', not {order}."
        )

    return factor
