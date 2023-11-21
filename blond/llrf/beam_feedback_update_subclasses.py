from __future__ import division


import numpy as np

from ..utils import bmath as bm
from blond.llrf.beam_feedback_abc import BeamFeedback

### LHC SUBCLASS
class BeamFeedback_LHC(BeamFeedback):
    """
    Beam Feedback subclass for the LHC. Implements 3 loops with their respective gains. Each loop can be
    deactivated by setting the respective gain to 0.

    1.  Phase Loop using PL_gain, using the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

        where the phase noise for the controlled blow-up can be optionally
        activated.
        
    2.   Synchro loop using SL_gain: a synchro loop to remove
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

        
    3.  Frequency loop using 'freq_gain': a frequency loop to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{FL} = - g_{FL} (\\omega_{rf} - h \\omega_{0})
    """
    def __init__(self, Ring, RFStation, Profile,
                 PhaseNoise=None, time_offset=None, window_coefficient=0, delay = 0,
                 PL_gain = 0, SL_gain = 0, freq_gain = 0, frequency_reference = 0):
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param delay [int]: Turn from which the loops start to act

        :param PL_gain [float]: Phase loop gain, in rad/s
        :param SL_gain [float]: Synchro Loop gain, in rad/s
        :param freq_gain [float]: Frequency Loop gain, unitless
        :param frequency_reference [float]: Reference Frequency for the frequency loop, in Hz
        """

        super().__init__(Ring, RFStation, Profile,
                 PhaseNoise, time_offset, window_coefficient, delay)

        self.PL_gain = PL_gain
        self.SL_gain = SL_gain
        self.frequency_reference = frequency_reference
        self.freq_gain = freq_gain

        self.lhc_y = 0

        if self.SL_gain != 0:
            #: | *LHC Synchronisation loop coefficient [1]*
            self.lhc_a = 5.25 - self.rf_station.omega_s0 / (np.pi * 40.)
            #: | *LHC Synchronisation loop time constant [turns]*
            self.lhc_t = (2 * np.pi * self.rf_station.Q_s * np.sqrt(self.lhc_a)) / \
                         np.sqrt(1 + self.PL_gain / self.SL_gain *
                                 np.sqrt((1 + 1 / self.lhc_a) / (1 + self.lhc_a)))

        else:
            self.lhc_a = np.zeros(self.rf_station.n_turns + 1)
            self.lhc_t = np.zeros(self.rf_station.n_turns + 1)

    def adjust_rf_frequency(self):
        """
        Calculates the frequency adjustment due to LHC Phase and Synchro or Frequency loops

        :return: Angular frequency correction due to loops domega_rf
        """
        counter = self.rf_station.counter[0]
        dphi_rf = self.rf_station.dphi_rf[0]

        dphi = self.phase_difference()

        # Frequency correction from phase loop and synchro loop
        self.domega_dphi = - self.PL_gain * dphi
        self.domega_dS = - self.SL_gain * (self.lhc_y + self.lhc_a[counter] \
                                         * (dphi_rf + self.frequency_reference))
        self.domega_df = - self.freq_gain * (self.rf_station.omega_rf[0, counter] -
                            self.rf_station.omega_rf_d[0, counter] +
                            self.frequency_reference)

        # Update recursion variable
        self.lhc_y = (1 - self.lhc_t[counter]) * self.lhc_y + \
                     (1 - self.lhc_a[counter]) * self.lhc_t[counter] * \
                     (dphi_rf + self.frequency_reference)

        return self.domega_dphi + self.domega_dS + self.domega_df


### SPS SUBCLASS
class BeamFeedback_SPS(BeamFeedback):
    """
    Beam Feedback subclass for the SPS. Implements 3 loops with their respective gains. Each loop can be
    deactivated by setting the respective gain to 0.

    1.  Phase Loop using PL_gain, using the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} (\\Delta\\varphi_{PL} + \\phi_{N})

        where the phase noise for the controlled blow-up can be optionally
        activated.

    2.   Radial loop using RL_gain: a radial loop to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{RL} = - sign(\\eta_{0}) g_{RL} \\frac{R_{ref} - \\Delta R}{\\rho}  ,


    3.  Frequency loop using 'freq_gain': a frequency loop to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{FL} = - g_{FL} (\\omega_{rf} - h \\omega_{0})
    """
    def __init__(self, Ring, RFStation, Profile,
                 PhaseNoise=None, time_offset=None, window_coefficient=0, delay = 0,
                 PL_gain = 0, RL_gain = 0, freq_gain = 0, radial_reference = 0.):
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param delay [int]: Turn from which the loops start to act

        :param PL_gain [float]: Phase loop gain, in rad/s
        :param RL_gain [float]: Radial Loop gain, in 1/s
        :param freq_gain [float]: Frequency Loop gain, unitless
        :param radial_reference [float]: Reference radial offset for the radial loop.
        """

        super().__init__(Ring, RFStation, Profile,
                 PhaseNoise, time_offset, window_coefficient, delay, radial_reference)

        self.PL_gain = PL_gain
        self.RL_gain = RL_gain
        self.radial_reference = radial_reference
        self.freq_gain = freq_gain

    def adjust_rf_frequency(self):
        """
        Calculates the frequency adjustment due to SPS Phase and Radial or Frequency loops

        :return: Angular frequency correction due to loops domega_rf
        """
        counter = self.rf_station.counter[0]

        if self.radial_reference != 0:
            self.radial_steering_from_freq()

        dphi = self.phase_difference()
        drho = self.radial_difference()

        # Frequency correction from phase loop and radial loop and frequency steering
        self.domega_dphi = - self.proportional_control(self.PL_gain, dphi)

        self.domega_dR = - bm.sign(self.rf_station.eta_0[counter]) * self.RL_gain * \
            (self.radial_reference - drho) / self.ring.ring_radius

        self.domega_df = - self.freq_gain * (self.rf_station.omega_rf[0, counter] -
                                         self.rf_station.omega_rf_d[0, counter])

        return self.domega_dphi + self.domega_dR + self.domega_df

class BeamFeedback_SPS_PL_PID(BeamFeedback):
    """
    Example SPS Subclass implementing a PID phase loop, with a proportional, integral and differential term with
    their respective gains

    """
    def __init__(self, Ring, RFStation, Profile,
                 PhaseNoise=None, time_offset=None, window_coefficient=0, delay = 0,
                 P_gain = 1, I_gain = 1, D_gain = 1):

        super().__init__(Ring, RFStation, Profile,
                 PhaseNoise, time_offset, window_coefficient,
                         delay)
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param delay [int]: Turn from which the loops start to act

        :param P_gain [float]: Proportional gain, in rad/s
        :param I_gain [float]: Integral gain, in rad/s
        :param D_gain [float]: Differential gain, in rad/s
        """
        self.P_gain = P_gain
        self.I_gain = I_gain
        self.D_gain = D_gain

        self.prev_out = 0
        self.prev_dphi = 0
        self.prevprev_dphi = 0

    def adjust_rf_frequency(self):
        """
        Implements the frequency adjustment due to a PID-controled phase loop. Can be used for any machine.
        """
        counter = self.rf_station.counter[0]


        dphi = self.phase_difference()

        # Frequency correction from phase loop
        domega_dphi = - self.PID_control(self.P_gain, self.I_gain, self.D_gain, self.prev_out,
                    dphi, self.prev_dphi, self.prevprev_dphi)

        self.prevprev_dphi = self.prev_dphi
        self.prev_dphi = dphi
        self.prev_out = domega_dphi

        return domega_dphi


### PS SUBCLASS
class BeamFeedback_PS(BeamFeedback):
    """
    Beam Feedback subclass for the SPS. Implements 2 loops with their respective gains. Each loop can be
    deactivated by setting the respective gain to 0.

    1.  Phase Loop using PL_gain, using the phase difference
        between beam and RF (actual synchronous phase). The transfer function is

        .. math::
            \\Delta \\omega_{rf}^{PL} = - g_{PL} \\Delta \\varphi_{out}

        with the transfered phase being calculated as

        .. math::
            \\Delta \\varphi_{out} = g_{diff} (\\Delta\\varphi_{PL} + \\phi_{N} - \\Delta \\varphi_{prev}) + g_{int} \\ Delta \\varphi_{out,prev}


        where the phase noise for the controlled blow-up can be optionally
        activated.

    2.   Radial loop using RL_gain: a radial loop to remove
        long-term frequency drifts:

        .. math::
            \\Delta \\omega_{rf}^{RL} =  g_{RL} \\Delta \\rho_{out} ,

        with

        .. math::

            \\Delta \\rho_{out} = (1-g_{internal}) \\Delta \\rho + g_{internal} \\Delta \\rho_{prev}

    """
    def __init__(self, Ring, RFStation, Profile,
                 PhaseNoise=None, time_offset=None, window_coefficient=0, delay = 0, radial_reference = 0,
                 PL_gain = 0, RL_gain = 0,
                 gd_pl = 5.704, gi_pl = 1-8.66e-5, g_rl = 1-1.853e-1):
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param delay [int]: Turn from which the loops start to act
        :param radial_reference [float]: Reference radial offset for the radial loop.

        :param PL_gain [float]: Phase loop gain, in rad/s
        :param RL_gain [float]: Radial Loop gain, in 1/m s
        :param gd_pl [float]: Hardware determined differential gain of the phase loop
        :param gi_pl [float]: Hardware determined integral gain of the phase loop
        :param g_rl [float]: Hardware determined radial loop gain parameter
        """

        super().__init__(Ring, RFStation, Profile,
                 PhaseNoise, time_offset, window_coefficient, delay, radial_reference)

        self.PL_gain = PL_gain
        self.RL_gain = RL_gain
        self.gi_pl = gi_pl
        self.gd_pl = gd_pl
        self.g_rl = g_rl

        self.prev_in_phase = 0
        self.prev_out_phase = 0
        self.prev_out_radial = 0

    def adjust_rf_frequency(self):
        """
        Calculates the frequency adjustment due to PS Phase and Radial Loops

        :return: Angular frequency correction due to loops domega_rf
        """

        dphi = self.phase_difference()
        drho = self.radial_difference()
        drho = (self.radial_reference - drho)

        # Frequency correction from phase loop and radial loop
        dphi_out = self.gd_pl * (dphi - self.prev_in_phase)  + self.gi_pl * self.prev_out_phase
        self.domega_dphi = - self.PL_gain * dphi_out
        self.prev_in_phase = dphi
        self.prev_out_phase = dphi_out

        drho_out = (1 - self.g_rl) * drho + self.g_rl * self.prev_out_radial
        self.domega_dR =  self.RL_gain * drho_out
        self.prev_out_radial = drho_out

        return self.domega_dphi + self.domega_dR


### PSB SUBCLASSES
class BeamFeedback_PSB(BeamFeedback):
    def __init__(self, Ring, RFStation, Profile,
                 PhaseNoise=None, time_offset=None, window_coefficient=0, delay = 0,
                 PL_gain = 1. / 25.e-6, RL_gain = [1.e7, 1.e11], period = 10.e-6,
                 transfer_coeff = [0.99803799, 0.99901903, 0.99901003]):
        """

        :param Ring: Ring Object
        :param RFStation: RF Station Object
        :param Profile: Profile Object
        :param PhaseNoise: array with length n_turns + 1
                Phase Noise injected into the machine. Will add the given phase at specified turn to the rf phase.
        :param sample_dE [int]: Determines which particles to sample for mean energy calculation.
                Every <sample_dE>. particle is sampled
        :param time_offset [float]: The beam phase is calculated in the dt interval [<time_offset>,<profile.cut_right>],
                i.e. from the time_offset value to the rightmost bucket
        :param window_coefficient [float]: Band-pass filter window coefficient for beam phase calculation
        :param delay [int]: Turn from which the loops start to act

        :param PL_gain [float]: Phase loop gain, in rad/s
        :param RL_gains [2-tuple of floats]: Radial Loop gains in 1/s. The first entry is the proportional, the second the integral gain
        :param period [float]: Peroid between turns? TODO
        :param transfer_coeff [3-tuple of floats]:  Gains set by the LLRF Phase Loop system. The list includes 3 gains:
                [<Gain of previous output>, <gain of current (average) phase difference>, <gain of previous (average) phase difference>]
        """
        super().__init__(Ring, RFStation, Profile,
                 PhaseNoise=PhaseNoise, time_offset=time_offset, window_coefficient=window_coefficient,
                         delay=delay, radial_reference = 0)

        # Phase Loop Gain
        self.PL_gain = PL_gain

        #: | *Radial loop gain, proportional [1] and integral [1/s].*

        self.RL_gain = RL_gain

        #: | *Optional: PL & RL acting only in certain time intervals/turns.*
        self.dt = period

        # Counter of turns passed since last time the PL was active
        self.PL_counter = 0
        self.on_time = np.array([])


        self.precalculate_time(Ring)

        #: | *Array of transfer function coefficients.*
        self.transfer_coeff = transfer_coeff

        #: | *Memory of previous phase correction, for phase loop.*
        self.dphi_sum = 0.
        self.dphi_avg = 0.
        self.dphi_avg_prev = 0.

        #: | *Memory of previous relative radial correction, for rad loop.*
        self.dR_over_R_prev = 0.

        #: | *Phase loop frequency correction [1/s]*
        self.domega_PL = 0.

        #: | *Radial loop frequency correction [1/s]*
        self.domega_RL = 0.

        self.dR_over_R = 0

    def adjust_rf_frequency(self):
        '''
        Phase and radial loops for PSB. See documentation on-line for details.
        '''

        # Average phase error while frequency is updated
        counter = self.rf_station.counter[0]

        dphi = self.phase_difference()

        self.dphi_sum += dphi

        # Phase and radial loop active on certain turns
        if counter == self.on_time[self.PL_counter]:
            # Phase loop
            self.dphi_avg = self.dphi_sum / (self.on_time[self.PL_counter]
                                             - self.on_time[self.PL_counter - 1])

            if self.RFnoise is not None:
                self.dphi_avg += self.RFnoise.dphi[counter]
            # Phase Loop
            self.domega_PL = self.transfer_coeff[0] * self.domega_PL \
                             + self.PL_gain * (self.transfer_coeff[1] * self.dphi_avg -
                                               self.transfer_coeff[2] * self.dphi_avg_prev)

            self.dphi_avg_prev = self.dphi_avg
            self.dphi_sum = 0.

            # Radial loop
            self.dR_over_R = (self.rf_station.omega_rf[0, counter] -
                              self.rf_station.omega_rf_d[0, counter]) / (
                                     self.rf_station.omega_rf_d[0, counter] *
                                     (1. / (self.ring.alpha_0[0, counter] *
                                            self.rf_station.gamma[counter] ** 2) - 1.))

            self.domega_RL = self.domega_RL + self.RL_gain[0] * (self.dR_over_R
                                                                        - self.dR_over_R_prev) \
                             + self.RL_gain[1] * self.dR_over_R

            self.dR_over_R_prev = self.dR_over_R

            # Counter to pick the next time step when the PL & RL will be active
            self.PL_counter += 1

        return - self.domega_PL - self.domega_RL

    def phase_difference(self, sharpWindow = False):
        '''
        TODO Docs
        '''
        if sharpWindow:
            self.beam_phase_sharpWindow()
        else:
            self.beam_phase()
        # Correct for design stable phase
        counter = self.rf_station.counter[0]
        self.dphi = self.phi_beam - self.rf_station.phi_s[counter]
        return self.dphi


    def precalculate_time(self, Ring):
        '''
        *For machines like the PSB, where the PL acts only in certain time
        intervals, pre-calculate on which turns to act.*
        '''

        if self.dt > 0:
            n = self.delay + 1
            while n < Ring.t_rev.size:
                summa = 0
                while summa < self.dt:
                    try:
                        summa += Ring.t_rev[n]
                        n += 1
                    except Exception:
                        self.on_time = np.append(self.on_time, 0)
                        return
                self.on_time = np.append(self.on_time, n - 1)
        else:
            self.on_time = np.arange(Ring.t_rev.size)