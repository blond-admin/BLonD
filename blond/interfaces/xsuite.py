'''
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**
'''

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.constants import c as clight

from xtrack import ReferenceEnergyIncrease, ZetaShift
from blond.trackers.tracker import RingAndRFTracker
from blond.impedances.impedance import TotalInducedVoltage

if TYPE_CHECKING:
    from xtrack import Particles
    from xtrack import Line
    from blond.beam.beam import Beam
    from blond.beam.profile import Profile
    from typing import Any, Union


def blond_to_xsuite_transform(dt: Union[float, np.ndarray], de: Union[float, np.ndarray], beta0: float,
                              energy0: float, omega_rf: float, phi_s: float = 0):
    r'''
    Coordinate transformation from Xsuite to BLonD at a given turn or multiple turns if numpy arrays ar given.

    :param dt: float or NDArray.
        The deviation in time from the reference clock in BLonD.
    :param de: float or NDArray.
        The deviation in energy from the synchronous particle.
    :param beta0: float.
        Synchronous beta.
    :param energy0: float.
        Synchronous energy.
    :param omega_rf: float.
        The rf angular frequency.
    :param phi_s: float.
        Synchronous phase in radians equivalent to Xsuite's phi_rf (below transition energy input should be phi_s-phi_rf).
        The default value is 0.

    :return zeta, ptau: as numpy-arrays (or single variable)
    '''

    ptau = de / (beta0 * energy0)
    zeta = -(dt - phi_s / omega_rf) * beta0 * clight
    return zeta, ptau


def xsuite_to_blond_transform(zeta: Union[float, np.ndarray], ptau: Union[float, np.ndarray],
                              beta0: float, energy0: float, omega_rf: float, phi_s: float = 0):
    r'''
    Coordinate transformation from Xsuite to BLonD.

    :param zeta: float or numpy-array.
        The zeta coordinate as defined in Xsuite.
    :param ptau: float or numpy-array.
        The ptau coordinate as defined in Xsuite.
    :param beta0: float.
        The synchronous beta.
    :param energy0: float.
        The synchronous energy in eV.
    :param omega_rf: float.
        The rf angular frequency.
    :param phi_s: float.
        The synchronous phase in radians equivalent to Xsuite's phi_rf
        (below transition energy input should be phi_s-phi_rf)

    :return dt, dE: as numpy-arrays (or single variable)
    '''

    dE = ptau * beta0 * energy0
    dt = -zeta / (beta0 * clight) + phi_s / omega_rf
    return dt, dE


class BlondElement:

    def __init__(self, trackable: Any, beam: Beam, update_zeta: bool = False) -> None:
        r"""
        The BlondElement class contains a trackable object from the BLonD simulation suite and the Beam object from
        BLonD. The class is intended to be an element to be added to the Line in XTrack instead of the default
        RF cavities in XTrack.

        The behavior of the tracking of the class depends on what trackable object is passed.
        If the RingAndRFTracker class is passed then a coordinate transformation will be performed and the energy
        deviation of the particles will be updated. If the TotalInducedVoltage class is passed then the
        TotalInducedVoltage.induced_voltage_sum() method will be called. Lastly, if any other class is passed then
        the tracking of BlondElement will track this class.

        trackable : BLonD trackable object
            BLonD object to be tracked, e.g. the RingAndRFTracker.
        beam : blond.beam.beam.Beam class
            BLonD Beam class used for tracking.
        """

        self.trackable = trackable
        self.beam = beam
        self.update_zeta = update_zeta

        # Initialize time- and orbit-shift to BLonD coordinates
        self._dt_shift = None
        self.orbit_shift = ZetaShift(dzeta=0)

        # Check what BLonD trackable has been passed and track the object accordingly
        if isinstance(self.trackable, RingAndRFTracker):
            self.track = getattr(self, "rf_track")
        elif isinstance(self.trackable, TotalInducedVoltage):
            self.track = getattr(self, "ind_track")
        else:
            self.track = getattr(self, "obs_track")

    def rf_track(self, particles: Particles) -> None:
        r"""
        Tracking method which is called if the trackable BLonD class is the RingAndRFTracker.

        particles : xtrack.Particles
            Particles class from xtrack
        """

        # Compute the shift to BLonD coordinates
        self._get_time_shift()

        # Convert the Xsuite coordinates to BLonD coordinates
        self.xsuite_part_to_blond_beam(particles)

        # Track with only the energy kick
        self.trackable.track()

        # Convert the BLonD energy coordinate to the equivalent Xsuite coordinate
        self.blond_beam_to_xsuite_part(particles, self.update_zeta)

        # Update the zeta shift due to potential frequency shifts in the RF
        self._orbit_shift(particles)

    def ind_track(self, particles: Particles) -> None:
        r"""
        Tracking method which is called if the trackable BLonD class is the TotalInducedVoltage.

        particles : xtrack.Particles
            Particles class from xtrack
        """
        self.trackable.induced_voltage_sum()

    def obs_track(self, particles: Particles) -> None:
        r"""
        Tracking method which is called if the trackable BLonD class is not the RingAndRFTracker or TotalInducedVoltage.

        particles : xtrack.Particles
            Particles class from xtrack
        """
        self.trackable.track()

    def xsuite_part_to_blond_beam(self, particles: Particles) -> None:
        r"""
        Coordinate transformation from Xsuite to BLonD.
        It uses the initial particle coordinates and beam properties in stored in the Particles class from xtrack.

        particles : xtrack.Particles
            Particles class from xtrack
        """
        # Convert Xsuite momentum to BLonD energy deviation
        self.beam.dE[:] = particles.beta0 * particles.energy0 * particles.ptau

        # Convert Xsuite zeta coordinate to BLonD time deviation
        self.beam.dt[:] = -particles.zeta / particles.beta0 / clight + self._dt_shift

        # Check what particles are still alive
        self.beam.id[:] = np.int_(particles.state > 0)

    def blond_beam_to_xsuite_part(self, particles: Particles, update_zeta: bool = False) -> None:
        r"""
        Coordinate transformation from BLonD to Xsuite.
        It uses the particle coordinates stored in Beam class of BLonD
        It uses the beam properties in stored in the Particles class from xtrack

        particles : xtrack.Particles
            Particles class from xtrack
        """
        # Subtract the given acceleration kick in BLonD, in Xsuite this is dealt with differently
        if isinstance(self.trackable, RingAndRFTracker):
            self.beam.dE = self.beam.dE - self.trackable.acceleration_kick[self.trackable.counter[0] - 1]

        # Convert BLonD energy deviation to Xsuite momentum
        particles.ptau = self.beam.dE / (particles.beta0 * particles.energy0)

        # Convert BLonD time deviation to Xsuite zeta.
        # This step is not needed usually because the BLonD simulation only does the kick, so dt is not changed.
        if update_zeta:
            particles.zeta = - (self.beam.dt - self._dt_shift) * particles.beta0 * clight

        # Check what particles are still alive after the BLonD track
        mask_lost = (self.beam.id <= 0) & particles.state > 0

        # If the particle is lost its state is set to -500 by convention
        particles.state[mask_lost] = -500

    def _get_time_shift(self) -> None:
        r'''
        Computes the time-shift between the Xsuite and BLonD coordinate systems.
        '''
        # Get turn counter from the RingAndRFTracker
        counter = self.trackable.rf_params.counter[0]

        # Compute the time-shift based on the synchronous phase
        self._dt_shift = ((self.trackable.rf_params.phi_s[counter] - self.trackable.rf_params.phi_rf[0, counter])
                          / self.trackable.rf_params.omega_rf[0, counter])

    def _orbit_shift(self, particles) -> None:
        r'''
        Computes the radial steering due to rf periods which are not an integer multiple of the revolution period.
        This is for example needed when tracking with global LLRF feedback loops.

        particles : xtrack.Particles
            Particles class from xtrack
        '''
        # Get turn counter from the RingAndRFTracker
        counter = self.trackable.counter[0]

        # Compute the orbit shift due to the difference in rf frequency
        dzeta = self.trackable.rf_params.ring_circumference
        omega_rf = self.trackable.rf_params.omega_rf[:, counter]
        omega_rf_design = (2 * np.pi * self.trackable.rf_params.harmonic[:, counter]
                           / self.trackable.rf_params.t_rev[counter])
        domega = omega_rf - omega_rf_design

        dzeta *= domega / omega_rf_design

        # Apply the shift
        self.orbit_shift.dzeta = dzeta

        # Track the shift
        self.orbit_shift.track(particles)


class EnergyUpdate:

    def __init__(self, momentum: Union[list, np.ndarray]) -> None:
        r"""
        Class to update the synchronous energy from the momentum program in BLonD.

        :param momentum: list or numpy-array.
            Momentum program from BLonD in units of eV
        """

        # Load momentum program
        self.momentum = momentum

        # Find initial momentum update
        init_p0c = self.momentum[1] - self.momentum[0]

        # Enter the initial momentum update in the ReferenceEnergyIncrease class in xsuite
        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def track(self, particles: Particles) -> None:
        r'''
        Track method for the class to update the synchronous energy.

        particles : xtrack.Particles
            Particles class from xtrack
        '''
        # Check for particles which are still alive
        mask_alive = particles.state > 0

        # Use the still alive particles to find the current turn momentum
        p0c_before = particles.p0c[mask_alive]

        # Find the momentum for the next turn
        p0c_after = self.momentum[particles.at_turn[mask_alive][0]]

        # Update the energy increment
        self.xsuite_energy_update.Delta_p0c = p0c_after - p0c_before[0]

        # Apply the energy increment to the particles
        self.xsuite_energy_update.track(particles)


class EnergyFrequencyUpdate:

    def __init__(self, momentum: np.ndarray, f_rf: np.ndarray, line: Line, cavity_name: str) -> None:
        r"""
        Class to update energy of Particles class turn-by-turn with the ReferenceEnergyIncrease function
        from xtrack. Additionally it updates the frequency of the xtrack cavity in the line.
        Intended to be used without BLonD-Xsuite interface.

        :param momentum: numpy-array
            The momentum program from BLonD in eV
        :param f_rf: numpy-array
            The frequency program from BLonD in Hz.
        particles : xtrack.Line
            Line class from xtrack
        :param cavity_name: string
            Name of cavity to update frequency.
        """

        # Load the parameters
        self.momentum = momentum
        self.f_rf = f_rf
        self.line = line
        self.cavity_name = cavity_name

        # Find initial momentum update
        init_p0c = self.momentum[1] - self.momentum[0]

        # Enter the initial momentum update in the ReferenceEnergyIncrease class in xsuite
        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def track(self, particles: Particles) -> None:
        r'''
        Track-method from for the class. This method updates the synchronous momentum and the rf frequency.

        particles : xtrack.Particles
            Particles class from xtrack
        '''
        # Check for particles which are still alive
        mask_alive = particles.state > 0

        # Use the still alive particles to find the current turn momentum
        p0c_before = particles.p0c[mask_alive]

        # Find the momentum for the next turn
        p0c_after = self.momentum[particles.at_turn[mask_alive][0]]

        # Update the energy increment
        self.xsuite_energy_update.Delta_p0c = p0c_after - p0c_before[0]

        # Apply the energy increment to the particles
        self.xsuite_energy_update.track(particles)

        # Update the rf frequency
        self.line[self.cavity_name].frequency = self.f_rf[particles.at_turn[mask_alive][0]]


class BlondObserver(BlondElement):

    def __init__(self, trackable: Any, beam: Beam, blond_cavity: bool, update_zeta: bool = False,
                 profile: Profile = None) -> None:
        r'''
        Child-class of the BlondElement, except that it updates the coordinates
        in BLonD when an observing element is used such as BunchMonitor.

        :param trackable: BLonD trackable object
            BLonD object to be tracked, e.g. the RingAndRFTracker.
        :param beam: blond.beam.beam.Beam class
            BLonD Beam class used for tracking.
        :param blond_cavity: bool.
            If there is no BlondCavity (bool = False), it updates its own turn-counter.
        :param update_zeta: bool.
            Boolean that decides whether zeta is converter back to dt after tracking object or not.
            Usually not necessary so default is False.
        :param profile: blond.beam.profile.Profile class
            BLonD Profile class used for tracking.
        '''
        # Initialize the parent class
        super().__init__(trackable, beam, update_zeta)

        # Load the parameters
        self.blond_cavity = blond_cavity
        # For bunch monitoring we need the profile separately added
        self.profile = profile

        # Initializing arrays for storing some turn-by-turn properties of xtrack.Particles class
        self.xsuite_ref_energy = np.zeros(self.trackable.rf_params.n_turns + 1)
        self.xsuite_trev = np.zeros(self.trackable.rf_params.n_turns + 1)

    def obs_track(self, particles: Particles) -> None:
        r'''
        observation tracker which performs the coordinate transformations.

        particles : xtrack.Particles
            Particles class from xtrack
        '''
        # Compute the shift to BLonD coordinates
        self._get_time_shift()

        # Convert the Xsuite coordinates to BLonD coordinates
        self.xsuite_part_to_blond_beam(particles)

        # Track profile if given
        if self.profile is not None:
            self.profile.track()

        # Track
        self.trackable.track()

        if not self.blond_cavity:
            # Updating the beam synchronous momentum etc.
            turn = self.trackable.rf_params.counter[0]

            self.beam.beta = self.trackable.rf_params.beta[turn + 1]
            self.beam.gamma = self.trackable.rf_params.gamma[turn + 1]
            self.beam.energy = self.trackable.rf_params.energy[turn + 1]
            self.beam.momentum = self.trackable.rf_params.momentum[turn + 1]

            # Update custom counter
            self.trackable.rf_params.counter[0] += 1

        # Convert the BLonD energy coordinate to the equivalent Xsuite coordinate
        self.blond_beam_to_xsuite_part(particles, self.update_zeta)

        # Track properties of xtrack.Particles
        self.xsuite_ref_energy[self.trackable.rf_params.counter[0] - 1] = particles.energy0[0]
        self.xsuite_trev[self.trackable.rf_params.counter[0] - 1] = particles.t_sim  # Does not update per turn!
