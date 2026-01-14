# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**, **Elleanor Lamb**
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from scipy.constants import c

from blond.core.beam.beams import BeamBaseClass
import numpy as np

if TYPE_CHECKING:
    from blond.core.simulation.simulation import Simulation
    from blond.cycles.magnetic_cycle import MagneticCycleBase

from xtrack import Line, Particles, ReferenceEnergyIncrease

from blond.core.base import BeamPhysicsRelevant
from blond.physics.drifts import DriftBaseClass  # import the base drift class


class DriftXsuite(DriftBaseClass):
    """
    BLonD–Xsuite interface element that has drift with Xsuite Line or sub-element.

    The energy update of the coordinates is done in BLonD, but the drift is done
    via Xsuite tracking. Hence, the 'DriftXsuite'.

    Parameters
    ----------
    beam : BeamBaseClass
        BLonD beam to track.
    line : xtrack.Line
        The Xsuite line (or sub-line) to be used for drift transport.
    phi_s : float, optional
        Synchronous phase [rad].
    orbit_length : int, optional
        Length of orbit [rad].
    element_name : str, optional
        Name of the specific Xsuite element to track (e.g. "drift_1").
        If None, the entire line will be tracked each call.
    section_index : int, optional
        Section index to group elements (passed to DriftBaseClass).
    **kwargs : Any
        Additional keyword arguments passed to DriftBaseClass.
    """

    skip_find_instances_attributes = ["_xsuite_element", "_line_internal"]

    def __init__(
        self,
        beam: BeamBaseClass,
        line: Line,
        phi_s: float = 0.0,  # can this be removed
        orbit_length: float = 0.0,  # is this needed?
        element_name: str = None,
        section_index: int = 0,
        momentum_compaction_factor: float = None,
        **kwargs: Any,  # for MRO and future compatibility
    ) -> None:
        super().__init__(
            orbit_length=orbit_length, section_index=section_index, **kwargs
        )
        self.beam = beam
        self._line_internal = line
        self.element_name = element_name
        self.phi_s = phi_s
        self._xsuite_element: Any | None = (
            None  # Will be set in on_init_simulation
        )
        self._momentum_compaction_factor = momentum_compaction_factor

        self._transition_gamma: float | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Init simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance to be initiated.
        """
        super().on_init_simulation(simulation)
        warnings.warn(
            "DriftXsuite is only valid for flat energy cycles. ",
            UserWarning,
            stacklevel=2,
        )

        self.omega_rf = 2.0 * np.pi * self.beam.reference_beta * c / simulation.ring.circumference

        if self.element_name is not None:
            try:
                self._xsuite_element = self._line_internal[self.element_name]
            except KeyError as exc:
                raise ValueError(
                    f"Xsuite element '{self.element_name}' not found in the line."
                ) from exc

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        #turn_i_init: int,
        **kwargs: Any,
    ) -> None:
        """
        Hook executed during simulation runtime.

        Parameters
        ----------
        simulation : Simulation
            Active BLonD simulation instance.
        beam : BeamBaseClass
            Beam being tracked.
        n_turns : int
            Number of turns to track.
        turn_i_init : int
            Initial turn index.
        **kwargs : Any
            Additional runtime arguments.
        """
        pass

    def track(self, beam: BeamBaseClass | None = None) -> None:
        """
        Track the beam through the Xsuite drift element or line.

        Parameters
        ----------
        beam : BeamBaseClass | None
            Beam to track; if None, the internally stored beam is used.
        """
        # --- Convert BLonD → Xsuite coordinates ---
        zeta = (
            -(beam._dt - self.phi_s / self.omega_rf)
            * self.beam.reference_beta
            * c
        )
        ptau = beam._dE / (
            self.beam.reference_beta * self.beam.reference_total_energy
        )

        # Create a temporary Xsuite Particles object
        particles = Particles(
            zeta=zeta,
            ptau=ptau,
            beta0=self.beam.reference_beta, #0.99999978,#self.beam.reference_beta, # is this correct
            energy0=self.beam.reference_total_energy, # is this correct
        )

        # --- Perform tracking ---
        if self._xsuite_element is not None:
            # Track only this drift (sub-element)
            self._xsuite_element.track(particles)
        else:
            # Track through the full line
            self._line_internal.track(particles)

        # --- Convert back to BLonD coordinates ---
        beam.dt = (
            self.alpha_0
            - particles.zeta / (self.beam.reference_beta * c)
            + self.phi_s / self.omega_rf
        )
        beam.dE = (
            particles.ptau
            * self.beam.reference_beta
            * self.beam.reference_total_energy
        )  # should this have been updated?

    @property
    def momentum_compaction_factor(self) -> float | None:
        """
        Return the momentum compaction factor.

        Returns
        -------
        float | None
            Momentum compaction factor if defined.
        """
        return self._momentum_compaction_factor

    @property
    def transition_gamma(self) -> float | None:
        """
        Return the transition gamma.

        Returns
        -------
        float | None
            Transition gamma if defined.
        """
        return self._transition_gamma

    @transition_gamma.setter
    def transition_gamma(self, transition_gamma: float) -> None:
        """
        Set the transition gamma and update momentum compaction.

        Parameters
        ----------
        transition_gamma : float
            Relativistic gamma at transition.
        """
        self._momentum_compaction_factor = float(
            1.0 / (transition_gamma * transition_gamma)
        )
        self._transition_gamma = transition_gamma

    def eta_0(self, gamma: float) -> float:
        """
        Compute the phase slip factor eta_0.

        Parameters
        ----------
        gamma : float
            Relativistic gamma.

        Returns
        -------
        float
            Phase slip factor eta_0.
        """
        return float(self.alpha_0 - (1 / (gamma * gamma)))

    @property
    def alpha_0(self) -> float | None:
        """
        Return the momentum compaction factor alpha_0.

        Returns
        -------
        float | None
            Momentum compaction factor.
        """
        if self.momentum_compaction_factor is None:
            raise ValueError(
                "Momentum compaction factor is not defined. "
            )
        return self.momentum_compaction_factor

    def get_line(self):
        """
        Get back the Xsuite line.

        Returns
        -------
        xsuite.interfaces.xsuite.Line
            Xsuite line.
        """
        return self._line_internal

    def compute_momentum_compaction(self):
        twiss = self._line_internal.twiss()
        momentum_compaction = twiss['momentum_compaction_factor']
        print(momentum_compaction)


class EnergyUpdateXsuite(BeamPhysicsRelevant):
    """
    Class to update the synchronous energy from the momentum program in BLonD.

    Both Xsuite and BLonD track a relative to a reference energy, this needs
    to be updated in both Xsuite and BlonD.

    Parameters
    ----------
    momentum : sequence
        Momentum program [eV/c] from BLonD.

    Attributes
    ----------
    momentum : numpy-array
        Momentum program [eV/c] from BLonD.
    xsuite_energy_update : xtrack.ReferenceEnergyIncrease class
        Class to update the momentum in xsuite.
    """

    def __init__(self, momentum: MagneticCycleBase):
        # Load momentum program
        self.momentum = momentum

        # Find initial momentum update
        init_p0c = self.momentum[1] - self.momentum[0]

        # Enter the initial momentum update in the ReferenceEnergyIncrease class in xsuite
        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Initialize simulation start.

        Parameters
        ----------
        simulation : Simulation
            Active BLonD simulation instance.
        """
        pass

    def track(self, particles: Particles):
        """
        Update the synchronous energy of particles.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles to which the energy update is applied.
        """
        # Check for particles which are still alive
        mask_alive = particles.state > 0  # todo

        # Use the still alive particles to find the current turn momentum
        p0c_before = particles.p0c[mask_alive]  # todo

        # Find the momentum for the next turn
        p0c_after = self.momentum[particles.at_turn[mask_alive][0]]  # todo

        # Update the energy increment
        self.xsuite_energy_update.Delta_p0c = p0c_after - p0c_before[0]  # todo

        # Apply the energy increment to the particles
        self.xsuite_energy_update.track(particles)  # todo
