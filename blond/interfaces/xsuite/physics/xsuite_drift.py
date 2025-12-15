"""
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**, **Elleanor Lamb**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from scipy.constants import c as clight

from blond._core.backends import backend
from blond._core.beam.beams import BeamBaseClass

if TYPE_CHECKING:
    from xtrack import Line, Particles
    from blond._core.simulation.simulation import Simulation
    from blond._core.base import BeamPhysicsRelevant
    from blond.cycles.magnetic_cycle import MagneticCycleBase

from blond.physics.drifts import DriftBaseClass  # import the base drift class


class DriftXSuite(DriftBaseClass):
    """
    BLonD–Xsuite interface element that performs drift tracking using an
    Xsuite Line or sub-element.

    Parameters
    ----------
    beam : BeamBaseClass
        BLonD beam to track.
    line : xtrack.Line
        The Xsuite line (or sub-line) to be used for drift transport.
    element_name : str, optional
        Name of the specific Xsuite element to track (e.g. "drift_1").
        If None, the entire line will be tracked each call.
    beta0 : float
        Reference beta (usually from synchronous particle).
    energy0 : float
        Reference total energy [eV].
    omega_rf : float
        RF angular frequency [rad/s].
    phi_s : float, optional
        Synchronous phase [rad].
    section_index : int, optional
        Section index to group elements (passed to DriftBaseClass).
    """

    skip_find_instances_attributes = ["_xsuite_element", "_line_internal"]

    def __init__(
        self,
        beam: BeamBaseClass,
        line: Line,
        beta0: float,
        energy0: float,
        omega_rf: float,
        phi_s: float = 0.0,
        orbit_length=0.0,
        element_name: Optional[str] = None,
        section_index: int = 0,
        **kwargs: Any,  # for MRO and future compatibility
    ) -> None:
        super().__init__(
            orbit_length=orbit_length, section_index=section_index, **kwargs
        )
        self.beam = beam
        self._line_internal = line
        self.element_name = element_name
        self.beta0 = beta0
        self.energy0 = energy0
        self.omega_rf = omega_rf
        self.phi_s = phi_s
        self._xsuite_element: Optional[Any] = (
            None  # Will be set in on_init_simulation
        )

        # Initialize drift-specific internal state
        self._transition_gamma: Optional[backend.float] = None
        self._momentum_compaction_factor: Optional[backend.float] = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        super().on_init_simulation(simulation)
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
        turn_i_init: int,
        **kwargs: Any,
    ) -> None:
        pass

    def track(self, beam: Optional[BeamBaseClass] = None) -> None:
        """
        Perform drift tracking via the Xsuite line.

        Converts BLonD beam coordinates to Xsuite, runs the drift, and converts back.
        """
        from xtrack import Particles

        if beam is None:
            beam = self.beam

        # --- Convert BLonD → Xsuite coordinates ---
        zeta = -(beam.dt - self.phi_s / self.omega_rf) * self.beta0 * clight
        ptau = beam.dE / (self.beta0 * self.energy0)

        # Create a temporary Xsuite Particles object
        particles = Particles(
            zeta=zeta,
            ptau=ptau,
            beta0=self.beta0,
            energy0=self.energy0,
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
            -particles.zeta / (self.beta0 * clight)
            + self.phi_s / self.omega_rf
        )
        beam.dE = particles.ptau * self.beta0 * self.energy0

    @property
    def momentum_compaction_factor(self) -> Optional[backend.float]:
        """Momentum compaction factor."""
        return self._momentum_compaction_factor

    @property
    def transition_gamma(self) -> Optional[backend.float]:
        """Gamma of transition crossing."""
        return self._transition_gamma

    @transition_gamma.setter
    def transition_gamma(self, transition_gamma: float) -> None:
        """Set gamma of transition crossing and compute momentum compaction factor."""
        self._momentum_compaction_factor = backend.float(
            1.0 / (transition_gamma * transition_gamma)
        )
        self._transition_gamma = backend.float(transition_gamma)

    def eta_0(self, gamma: float) -> backend.float:
        """Calculate eta_0 for given gamma."""
        return backend.float(self.alpha_0 - (1 / (gamma * gamma)))

    @property
    def alpha_0(self) -> Optional[backend.float]:
        """Alias for momentum compaction factor."""
        return self.momentum_compaction_factor

    def get_line(self):
        return self._line_internal


class EnergyUpdateXsuite(BeamPhysicsRelevant):
    # todo
    """Class to update the synchronous energy from the momentum program in BLonD.

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
        from xtrack import ReferenceEnergyIncrease, ZetaShift

        # Load momentum program
        self.momentum = momentum

        # Find initial momentum update
        init_p0c = self.momentum[1] - self.momentum[0]

        # Enter the initial momentum update in the ReferenceEnergyIncrease class in xsuite
        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def track(self, particles: Particles):
        """
        # todo move this to inside Xsuite drift, should not be separate classes?
        Track method for the class to update the synchronous energy.


        Parameters
        ----------
        particles : xtrack.Particles
            Particles class from xtrack.

        Attributes
        ----------
        xsuite_energy_update : xtrack.ReferenceEnergyIncrease class
            Class to update the momentum in xsuite.
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
