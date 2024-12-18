# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Various beam phase loops with optional synchronisation/frequency/radial loops
for the CERN machines**

:Authors: **Helga Timko**, **Alexandre Lasheen**
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional
    from ..input_parameters.rf_parameters import RFStation
    from ..input_parameters.ring import Ring
    from ..beam.profile import Profile
    from ..utils.types import DeviceType
    from .rf_noise import LHCNoiseFB


class BeamFeedback(ABC):
    """One-turn phase loop adjusting RF frequency and phase.

    One-turn beam phase loop for different machines with different hardware.
    Use 'dt' for a phase loop that is active only in certain turns.
    The phase loop acts directly on the RF frequency of all harmonics and
    affects the RF phase as well.

    Parameters
    ----------
    ring
        A Ring type class
    rf_station
        An RFStation type class
    profile
        An Profile type class
    gain_phase_loop
        Phase loop gain. Implementation depends on machine.
    machine
        Machine name
    phase_noise
        Optional import of RF phase_noise object
        # TODO deprecated ?
    delay
        # todo
    alpha
        Band-pass filter window coefficient for beam phase calculation.
    time_offset
        Determines from which RF-buckets the band-pass filter starts to act

    Attributes
    ----------
    ring
        Import Ring
    rf_station
        Import RFStation
    profile
        Import Profile
    alpha
        Band-pass filter window coefficient for beam phase calculation.
    gain_phase_loop
        Phase loop gain. Implementation depends on machine.
    drho
        Relative radial displacement [1], for radial loop.
    domega_rf
        Phase loop frequency correction of the main RF system.
    phi_beam
        Beam phase measured at the main RF frequency.
    dphi
        Phase difference between beam and RF.
    reference
        Reference signal for secondary loop to test step response.
    rf_noise
        Optional import of RF phase_noise object

    """

    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 machine: str, phase_noise: None = None,  # todo class doesnt exist anymore??
                 alpha=0., time_offset=None,
                 turn_delay: int = 1) -> None:

        self.machine = str(machine)
        # Import Ring
        self.ring = ring

        # Import RFStation
        self.rf_station = rf_station

        # Import Profile
        self.profile: Profile = profile

        # Band-pass filter window coefficient for beam phase calculation.
        self.alpha = alpha

        # determines from which RF-buckets the band-pass filter starts to act
        self.time_offset = time_offset

        # Delay between measurement and application of correction
        self.turn_delay = turn_delay

        # Relative radial displacement [1], for radial loop.
        self.drho = 0.

        # Phase loop frequency correction of the main RF system.
        self.domega_rf = 0.

        # Beam phase measured at the main RF frequency.
        self.phi_beam = 0.

        # Phase difference between beam and RF.
        self.dphi = 0.

        # Reference signal for secondary loop to test step response.
        self.reference = 0.

        # Optional import of RF phase_noise object
        self.rf_noise = phase_noise
        if (self.rf_noise is not None
                and (len(self.rf_noise.dphi) != ring.n_turns + 1)):
            # PhaseNoiseError
            raise RuntimeError(
                'Phase noise has to have a length of n_turns + 1')

    @property
    def RFnoise(self):
        warnings.warn("Use rf_noise instead", DeprecationWarning)

        return self.rf_noise

    @RFnoise.setter
    def RFnoise(self, val):
        self.rf_noise = val

    @abstractmethod
    def _track(self):
        """Calculate correction on RF"""
        pass

    def track(self):
        """Calculate PL correction on main RF frequency

        Calculate PL correction on main RF frequency depending on machine and
        propagate it to other RF systems.
        The update of the RF phase and frequency for the next turn,
        for all systems is done in the tracker.
        """

        # Calculate PL correction on RF frequency
        self._track()

        # Update the RF frequency of all systems for the next turn
        counter = self.rf_station.counter[0] + self.turn_delay
        self.rf_station.omega_rf[:, counter] += self.domega_rf * \
                                                self.rf_station.harmonic[:, counter] / \
                                                self.rf_station.harmonic[0, counter]

    def beam_phase(self) -> None:
        """
        *Beam phase measured at the main RF frequency and phase. The beam is
        convolved with the window function of the band-pass filter of the
        machine. The coefficients of sine and cosine components determine the
        beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
        phase is already w.r.t. the instantaneous RF phase.*
        """

        # Main RF frequency at the present turn
        omega_rf = self.rf_station.omega_rf[0, self.rf_station.counter[0]]
        phi_rf = self.rf_station.phi_rf[0, self.rf_station.counter[0]]

        if self.time_offset is None:
            coeff = bm.beam_phase(self.profile.bin_centers,
                                  self.profile.n_macroparticles,
                                  self.alpha, omega_rf, phi_rf,
                                  self.profile.bin_size)
        else:
            indexes = self.profile.bin_centers >= self.time_offset
            coeff = bm.beam_phase(self.profile.bin_centers[indexes],
                                  self.profile.n_macroparticles[indexes],
                                  self.alpha, omega_rf, phi_rf,
                                  self.profile.bin_size)

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(coeff) + np.pi

    def beam_phase_sharpWindow(self):
        warnings.warn("Use 'beam_phase_sharp_window' instead!", DeprecationWarning)
        self.beam_phase_sharp_window()

    def beam_phase_sharp_window(self):
        """
        *Beam phase measured at the main RF frequency and phase. The beam is
        averaged over a window. The coefficients of sine and cosine components
        determine the beam phase, projected to the range -Pi/2 to 3/2 Pi.
        Note that this beam phase is already w.r.t. the instantaneous RF phase.*
        """

        # Main RF frequency at the present turn
        turn = self.rf_station.counter[0]
        omega_rf = self.rf_station.omega_rf[0, turn]
        phi_rf = self.rf_station.phi_rf[0, turn]

        if self.alpha != 0.0:
            indexes = bm.logical_and((self.time_offset - np.pi / omega_rf)
                                     <= self.profile.bin_centers,
                                     self.profile.bin_centers
                                     <= (-1 / self.alpha + self.time_offset -
                                         2 * np.pi / omega_rf))
        else:
            indexes = bm.ones(self.profile.n_slices, dtype=bool)

        # Convolve with window function
        scoeff = bm.trapz(bm.sin(omega_rf * self.profile.bin_centers[indexes]
                                 + phi_rf)
                          * self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)
        ccoeff = bm.trapz(bm.cos(omega_rf * self.profile.bin_centers[indexes]
                                 + phi_rf) *
                          self.profile.n_macroparticles[indexes],
                          dx=self.profile.bin_size)

        # Project beam phase to (pi/2,3pi/2) range
        self.phi_beam = np.arctan(scoeff / ccoeff) + np.pi

    def phase_difference(self):
        """
        *Phase difference between beam and RF phase of the main RF system.
        Optional: add RF phase noise through dphi directly.*
        """

        # Correct for design stable phase
        counter = self.rf_station.counter[0]
        self.dphi = self.phi_beam - self.rf_station.phi_s[counter]

        # Possibility to add RF phase noise through the PL
        # TODO REWRITE
        if self.rf_noise is not None:
            if self.noiseFB is not None:
                self.dphi += self.noiseFB.x * self.rf_noise.dphi[counter]
            else:
                if self.machine == 'PSB':
                    self.dphi = self.dphi
                else:
                    self.dphi += self.rf_noise.dphi[counter]

    def to_gpu(self, recursive=True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return

        # No arrays need to be transferred

        # to make sure it will not be called again
        self._device: DeviceType = 'GPU'

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':
            return

        # No arrays need to be transferred

        # to make sure it will not be called again
        self._device: DeviceType = 'CPU'


class LHCBeamFeedback(BeamFeedback):
    """Calculation of the LHC RF frequency correction

    Calculation of the LHC RF frequency correction from the phase difference
    between beam and RF (actual synchronous phase). The transfer function is

    .. math::
        \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

    where the phase noise for the controlled blow-up can be optionally
    activated.
    Using 'gain_synchro_loop', a synchro loop can be activated in addition to remove
    long-term frequency drifts:

    .. math::
        \\Delta \\omega_{rf}^{SL} = - g_{SL} (y + a \\Delta\\varphi_{rf}) ,

    where we use the recursion

    .. math::
        y_{n+1} = (1 - \\tau) y_n + (1 - a) \\tau \\Delta\\varphi_{rf} ,

    with a and \tau being defined through the synchrotron frequency f_s and
    the synchrotron tune Q_s as

    .. math::
        a (f_s) \\equiv 5.25 - \\frac{f_s}{\\pi 40~\\text{Hz}} ,

    .. math::
        \\tau(f_s) \\equiv 2 \\pi Q_s \\sqrt{ \\frac{a}{1 + \\frac{g_{PL}}{g_{SL}} \\sqrt{\\frac{1 + 1/a}{1 + a}} }}

    Parameters
    ----------
    ring
        A Ring type class.
    rf_station
        An RFStation type class.
    profile
        A Profile type class.
    gain_phase_loop
        Phase loop gain.
    gain_synchro_loop
        Synchronisation loop gain.
    lhc_noise_feedback
        Feedback on phase noise amplitude for LHC blow-up.
    kwargs
        Additional optional BeamFeedback arguments.

    Attributes
    ----------
    lhc_noise_feedback
        Amplitude-scaling feedback object for LHC noise.
    gain_synchro_loop
        Synchronisation loop gain.
    lhc_y
        LHC Synchronisation loop recursion variable.
    lhc_a
        LHC Synchronisation loop coefficient [1].
    lhc_t
        LHC Synchronisation loop time constant [turns].

    """
    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 gain_phase_loop: float, gain_synchro_loop=0.0,
                 lhc_noise_feedback: Optional[LHCNoiseFB] = None,
                 **kwargs):

        super().__init__(ring=ring, rf_station=rf_station, profile=profile, machine="LHC", **kwargs)

        # Optional import of amplitude-scaling feedback object LHCNoiseFB
        self.lhc_noise_feedback = lhc_noise_feedback

        # Beam-phase loop gain
        self.gain_phase_loop = gain_phase_loop

        # Synchronisation loop gain.
        self.gain_synchro_loop = gain_synchro_loop

        # LHC Synchronisation loop recursion variable
        self.lhc_y = 0

        if self.gain_synchro_loop != 0:

            # LHC Synchronisation loop coefficient [1]
            self.lhc_a = 5.25 - self.rf_station.omega_s0 / (np.pi * 40.)
            # LHC Synchronisation loop time constant [turns]
            self.lhc_t = (2 * np.pi * self.rf_station.Q_s * np.sqrt(self.lhc_a)) / \
                         np.sqrt(1 + self.gain_phase_loop / self.gain_synchro_loop *
                                 np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a)))

        else:

            self.lhc_a = np.zeros(self.rf_station.n_turns + 1)
            self.lhc_t = np.zeros(self.rf_station.n_turns + 1)

    @property
    def noiseFB(self):
        warnings.warn("Use 'lhc_noise_feedback' instead !", DeprecationWarning)
        return self.lhc_noise_feedback

    @noiseFB.setter
    def noiseFB(self, val):
        self.lhc_noise_feedback = val

    def _track(self):

        counter = self.rf_station.counter[0]
        dphi_rf = self.rf_station.dphi_rf[0]

        self.beam_phase()
        self.phase_difference()

        # Frequency correction from phase loop and synchro loop
        self.domega_rf = - self.gain_phase_loop * self.dphi \
                         - self.gain_synchro_loop * (self.lhc_y + self.lhc_a[counter]
                                                     * (dphi_rf + self.reference))

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter]) * self.lhc_y + \
                     (1 - self.lhc_a[counter]) * self.lhc_t[counter] * \
                     (dphi_rf + self.reference)


class LHCBeamFeedbackFrequency(BeamFeedback):
    """Calculation of the LHC RF frequency correction

    Calculation of the LHC RF frequency correction from the phase difference
    between beam and RF (actual synchronous phase). The transfer function is

    .. math::
        \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

    where the phase noise for the controlled blow-up can be optionally
    activated.
    Using 'gain_frequency_loop', a frequency loop can be activated in addition to remove
    long-term frequency drifts:

    .. math::
        \\Delta \\omega_{rf}^{FL} = - g_{FL} (\\omega_{rf} - h \\omega_{0})

    Parameters
    ----------
    ring
        A Ring type class.
    rf_station
        An RFStation type class.
    profile
        A Profile type class.
    gain_phase_loop
        Phase loop gain.
    gain_frequency_loop
        Frequency loop gain.
    lhc_noise_feedback
        Feedback on phase noise amplitude for LHC blow-up.
    kwargs
        Additional optional BeamFeedback arguments.

    Attributes
    ----------
    lhc_noise_feedback
        Amplitude-scaling feedback object for LHC noise.
    gain_frequency_loop
        Frequency loop gain.

        """
    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 gain_phase_loop: float, gain_frequency_loop=0.0, lhc_noise_feedback: Optional[LHCNoiseFB] = None,
                 **kwargs):
        super().__init__(ring=ring, rf_station=rf_station, profile=profile, machine="LHC_F", **kwargs)

        # Beam-phase loop gain
        self.gain_phase_loop = gain_phase_loop

        # Optional import of amplitude-scaling feedback object LHCNoiseFB
        self.lhc_noise_feedback = lhc_noise_feedback

        # Frequency loop gain.
        self.gain_frequency_loop = gain_frequency_loop

    def _track(self):

        counter = self.rf_station.counter[0]

        self.beam_phase()
        self.phase_difference()

        # Frequency correction from phase loop and frequency loop
        self.domega_rf = - self.gain_phase_loop * self.dphi \
                         - self.gain_frequency_loop * (self.rf_station.omega_rf[0, counter] -
                                                       self.rf_station.omega_rf_d[0, counter] +
                                                       self.reference)


class SPSBeamFeedbackRadial(BeamFeedback):
    """Calculation of the SPS RF frequency correction

    Calculation of the SPS RF frequency correction from the phase difference
    between beam and RF (actual synchronous phase). The transfer function is

    .. math::
        \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

    where the phase noise for the controlled blow-up can be optionally
    activated.
    Using 'gain_radial_loop', a radial loop can be activated in addition to remove
    long-term frequency drifts

    Parameters
    ----------
    ring
        A Ring type class.
    rf_station
        An RFStation type class.
    profile
        A Profile type class.
    gain_phase_loop
        Phase loop gain.
    gain_radial_loop
        Radial loop gain.
    sample_dE
        Number of particles to sample from dE for orbit calculation
    kwargs
        Additional optional BeamFeedback arguments.

    Attributes
    ----------
    gain_radial_loop
        Phase loop gain
    sample_dE
        Number of particles to sample from dE for orbit calculation


    """
    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 gain_phase_loop: float, gain_radial_loop=0.0, sample_dE=1,
                 **kwargs):
        super().__init__(ring=ring, rf_station=rf_station, profile=profile, machine="SPS_RL", **kwargs)

        self.gain_phase_loop = gain_phase_loop

        # Phase loop gain.
        self.gain_radial_loop = gain_radial_loop

        # Number of particles to sample from dE for orbit calculation
        self.sample_dE = sample_dE

    def radial_difference(self):
        """
        *Radial difference between beam and design orbit.*
        """

        counter = self.rf_station.counter[0]

        # Correct for design orbit
        #        self.average_dE = np.mean(self.profile.beam.dE[(self.profile.beam.dt >
        #            self.profile.bin_centers[0])*(self.profile.beam.dt <
        #                                         self.profile.bin_centers[-1])])
        self.average_dE = bm.mean(self.profile.beam.dE[::self.sample_dE])

        self.drho = self.ring.alpha_0[0, counter] * \
                    self.ring.ring_radius * self.average_dE / \
                    (self.ring.beta[0, counter] ** 2.
                     * self.ring.energy[0, counter])

    def radial_steering_from_freq(self):
        """
        *Frequency and phase change for the current turn due to the radial steering program.*
        """

        counter = self.rf_station.counter[0]

        self.radial_steering_domega_rf = - self.rf_station.omega_rf_d[0, counter] * \
                                         self.rf_station.eta_0[counter] / self.ring.alpha_0[0, counter] * \
                                         self.reference / self.ring.ring_radius

        self.rf_station.omega_rf[:, counter] += self.radial_steering_domega_rf * \
                                                self.rf_station.harmonic[:, counter] / \
                                                self.rf_station.harmonic[0, counter]

        # Update the RF phase of all systems for the next turn
        # Accumulated phase offset due to PL in each RF system
        # FIXME dphi_rf_steering never declared, this will crash
        self.rf_station.dphi_rf_steering += 2. * np.pi * self.rf_station.harmonic[:, counter] * \
                                            (self.rf_station.omega_rf[:, counter] -
                                             self.rf_station.omega_rf_d[:, counter]) / \
                                            self.rf_station.omega_rf_d[:, counter]

        # Total phase offset
        self.rf_station.phi_rf[:, counter] += self.rf_station.dphi_rf_steering

    def _track(self):
        counter = self.rf_station.counter[0]

        if self.reference != 0:
            self.radial_steering_from_freq()

        self.beam_phase()
        self.phase_difference()
        self.radial_difference()

        # Frequency correction from phase loop and radial loop
        self.domega_dphi = - self.gain_phase_loop * self.dphi
        self.domega_dR = - bm.sign(self.rf_station.eta_0[counter]) * self.gain_radial_loop * \
                         (self.reference - self.drho) / self.ring.ring_radius

        self.domega_rf = self.domega_dphi + self.domega_dR


class SPSBeamFeedbackFrequency(BeamFeedback):
    """Calculation of the SPS RF frequency correction

    Calculation of the SPS RF frequency correction from the phase
    difference between beam and RF (actual synchronous phase). Same as
    LHC_F, except the calculation of the beam phase.

    Parameters
    ----------
    ring
        A Ring type class.
    rf_station
        An RFStation type class.
    profile
        A Profile type class.
    gain_phase_loop
        Phase loop gain.
    gain_frequency_loop
        Frequency loop gain.
    kwargs
        Additional optional BeamFeedback arguments.

    Attributes
    ----------
    gain_frequency_loop
        Frequency loop gain.

    """
    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 gain_phase_loop: float, gain_frequency_loop=0., **kwargs):
        super().__init__(ring=ring, rf_station=rf_station, profile=profile, machine="SPS_F", **kwargs)

        # Beam-phase loop gain.
        self.gain_phase_loop = gain_phase_loop

        # Frequency loop gain.
        self.gain_frequency_loop = gain_frequency_loop

    def _track(self):
        """
        Calculation of the SPS RF frequency correction from the phase
        difference between beam and RF (actual synchronous phase). Same as
        LHC_F, except the calculation of the beam phase.
        """

        counter = self.rf_station.counter[0]

        self.beam_phase_sharp_window()
        self.phase_difference()

        # Frequency correction from phase loop and frequency loop
        self.domega_dphi = - self.gain_phase_loop * self.dphi
        self.domega_df = - self.gain_frequency_loop * (self.rf_station.omega_rf[0, counter] -
                                                       self.rf_station.omega_rf_d[0, counter])

        self.domega_rf = self.domega_dphi + self.domega_df


class PSBBeamFeedback(BeamFeedback):
    """Phase and radial loops for PSB.


    Phase and radial loops for PSB. See documentation on-line for details.

    Parameters
    ----------
    ring
        A Ring type class.
    rf_station
        An RFStation type class.
    profile
        A Profile type class.
    gain_phase_loop
        Phase loop gain.
    gain_radial_loop
        Radial loop gain.
    dt
        Phase Loop sampling period [s]
    coefficients
        Array of transfer function coefficients.
    kwargs
        Additional optional BeamFeedback arguments.

    Attributes
    ----------
    gain
        Phase loop gain.
    gain_radial_loop
        Radial loop gain, proportional [1] and integral [1/s].
    dt
        Phase Loop sampling period [s]
    PL_counter
        Counter of turns passed since last time the PL was active
    on_time
        Phase and radial loop active on certain turns
    coefficients
        Array of transfer function coefficients
    dphi_sum
        Memory of previous phase correction, for phase loop
    dphi_av
        Memory of previous phase correction, for phase loop
    dphi_av_prev
        Memory of previous phase correction, for phase loop
    dR_over_R_prev
        Memory of previous relative radial correction, for rad loop.
    domega_PL
        Phase loop phase correction [1/s]
    domega_RL
        Radial loop radial correction [1/s]
    dR_over_R
        # TODO
    """
    @handle_legacy_kwargs
    def __init__(self, ring: Ring, rf_station: RFStation, profile: Profile,
                 gain_phase_loop: float, gain_radial_loop=(0., 0.), dt=10.e-6,
                 coefficients=(0.999019, -0.999019, 0., 1., -0.998038, 0.),
                 delay: int = 0, **kwargs):
        super().__init__(ring=ring, rf_station=rf_station, profile=profile, machine="PSB", **kwargs)

        self.gain = gain_phase_loop * np.ones(self.ring.n_turns + 1)

        self.delay = delay

        # Radial loop gain, proportional [1] and integral [1/s].
        self.gain_radial_loop = list(gain_radial_loop)

        # Optional: PL & RL acting only in certain time intervals/turns.
        self.gain_radial_loop[0] = self.gain_radial_loop[0] * np.ones(self.ring.n_turns + 1)
        self.gain_radial_loop[1] = self.gain_radial_loop[1] * np.ones(self.ring.n_turns + 1)

        # Phase Loop sampling period [s]
        self.dt = dt

        # Counter of turns passed since last time the PL was active
        self.PL_counter = 0
        self.on_time = np.array([])

        self.precalculate_time(self.ring)

        # Array of transfer function coefficients.
        self.coefficients = list(coefficients)

        # Memory of previous phase correction, for phase loop.
        self.dphi_sum = 0.
        self.dphi_av = 0.
        self.dphi_av_prev = 0.

        # Memory of previous relative radial correction, for rad loop.
        self.dR_over_R_prev = 0.

        # Phase loop phase correction [1/s]
        self.domega_PL = 0.

        # Radial loop radial correction [1/s]
        self.domega_RL = 0.

        self.dR_over_R = 0

    @handle_legacy_kwargs
    def precalculate_time(self, ring: Ring):
        """
        *For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.*
        """

        if self.dt > 0:
            n = self.delay + 1
            while n < ring.t_rev.size:
                summa = 0
                while summa < self.dt:
                    try:
                        summa += ring.t_rev[n]
                        n += 1
                    except Exception as exc:
                        warnings.warn(str(exc))
                        self.on_time = np.append(self.on_time, 0)
                        return
                self.on_time = np.append(self.on_time, n - 1)
        else:
            self.on_time = np.arange(ring.t_rev.size)

    def _track(self):
        """
        Phase and radial loops for PSB. See documentation on-line for details.
        """

        # Average phase error while frequency is updated
        counter = self.rf_station.counter[0]
        self.beam_phase()
        self.phase_difference()

        self.dphi_sum += self.dphi

        # Phase and radial loop active on certain turns
        if counter == self.on_time[self.PL_counter] and counter >= self.delay:
            # Phase loop
            self.dphi_av = self.dphi_sum / (self.on_time[self.PL_counter]
                                            - self.on_time[self.PL_counter - 1])

            if self.rf_noise is not None:
                self.dphi_av += self.rf_noise.dphi[counter]

            self.domega_PL = 0.99803799 * self.domega_PL \
                             + self.gain[counter] * (0.99901903 * self.dphi_av -
                                                     0.99901003 * self.dphi_av_prev)

            self.dphi_av_prev = self.dphi_av
            self.dphi_sum = 0.

            # Radial loop
            self.dR_over_R = (self.rf_station.omega_rf[0, counter] -
                              self.rf_station.omega_rf_d[0, counter]) / (
                                     self.rf_station.omega_rf_d[0, counter] *
                                     (1. / (self.ring.alpha_0[0, counter] *
                                            self.rf_station.gamma[counter] ** 2) - 1.))

            self.domega_RL = (self.domega_RL + self.gain_radial_loop[0][counter]
                              * (self.dR_over_R - self.dR_over_R_prev)
                              + self.gain_radial_loop[1][counter] * self.dR_over_R)

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        # Apply frequency correction
        self.domega_rf = - self.domega_PL - self.domega_RL
