# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Various cavity loops for the CERN machines**

:Authors: **Helga Timko**
'''

from __future__ import division
import ctypes
import logging
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.constants import e

from ..llrf.signal_processing import comb_filter, cartesian_to_polar, \
    polar_to_cartesian, modulator, moving_average, rf_beam_current
from ..llrf.impulse_response import SPS4Section200MHzTWC, SPS5Section200MHzTWC
# from ..setup_cpp import libblond
from .. import libblond

from ..beam.profile import Profile, CutOptions


class CavityFeedbackCommissioning(object):

    def __init__(self, debug=False, open_loop=False, open_FB=False,
                 open_drive=False):

        self.debug = bool(debug)
        # Multiply with zeros if open == True
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_FB = int(np.invert(bool(open_FB)))
        self.open_drive = int(np.invert(bool(open_drive)))


class SPSCavityFeedback(object):
    """Class determining the turn-by-turn total RF voltage and phase correction
    originating from the individual cavity feedbacks. Assumes two 4-section and
    two 5-section travelling wave cavities and a voltage partition proportional
    to the number of sections.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        A Profile type class
    G_llrf : float or list
        LLRF Gain [1]; if passed as a float, both 4- and 5-section cavities
        have the same G_llrf; if passed as a list, the first and second
        elements correspond to the G_llrf of the 4- and 5-section cavity
        feedback; default is 10
    G_tx : float or list
        Transmitter gain [1] of the cavity feedback; convention same as G_llrf;
        default is 0.5
    open_loop : int(bool)
        Open (0) or closed (1) feedback loop; default is 1

    Attributes
    ----------
    OTFB_4 : class
        An SPSOneTurnFeedback type class
    OTFB_5 : class
        An SPSOneTurnFeedback type class
    V_sum : complex array
        Vector sum of RF voltage from all the cavities
    V_corr : float array
        RF voltage correction array to be applied in the tracker
    phi_corr : float array
        RF phase correction array to be applied in the tracker
    logger : logger
        Logger of the present class

    """

    def __init__(self, RFStation, Beam, Profile, G_llrf=10, G_tx=0.5,
                 a_comb=15/16, turns=1000,
                 Commissioning=CavityFeedbackCommissioning()):

        # Options for commissioning the feedback
        self.Commissioning = Commissioning

        self.rf = RFStation

        # Parse input for G_llrf
        if type(G_llrf) is list:
            G_llrf_4 = G_llrf[0]
            G_llrf_5 = G_llrf[1]
        else:
            G_llrf_4 = G_llrf
            G_llrf_5 = G_llrf

        if type(G_tx) is list:
            G_tx_4 = G_tx[0]
            G_tx_5 = G_tx[1]
        else:
            G_tx_4 = G_tx
            G_tx_5 = G_tx

        # Voltage partition proportional to the number of sections
        self.OTFB_4 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                         n_cavities=2, V_part=4/9,
                                         G_llrf=float(G_llrf_4),
                                         G_tx=float(G_tx_4),
                                         a_comb=float(a_comb),
                                         Commissioning=self.Commissioning)
        self.OTFB_5 = SPSOneTurnFeedback(RFStation, Beam, Profile, 5,
                                         n_cavities=2, V_part=5/9,
                                         G_llrf=float(G_llrf_5),
                                         G_tx=float(G_tx_5),
                                         a_comb=float(a_comb),
                                         Commissioning=self.Commissioning)

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")

        # Initialise OTFB without beam
        self.turns = int(turns)
        if turns < 1:
            raise RuntimeError("ERROR in SPSCavityFeedback: 'turns' has to" +
                               " be a positive integer!")
        self.track_init(debug=Commissioning.debug)

    def track(self):

        self.OTFB_4.track()
        self.OTFB_5.track()

        self.V_sum = self.OTFB_4.V_fine_tot + self.OTFB_5.V_fine_tot

        self.V_corr, alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr = 0.5*np.pi - alpha_sum \
            - self.rf.phi_rf[0, self.rf.counter[0]]

    def track_init(self, debug=False):
        r''' Tracking of the SPSCavityFeedback without beam.
        '''

#        cmap = plt.get_cmap('jet')
#        colors = cmap(np.linspace(0,1, self.turns))
#        plt.figure('voltage')
#        plt.clf()
#        plt.grid()

        for i in range(self.turns):
            #            print('OTFB pre-tracking iteration ', i)
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_4.track_no_beam()
#            plt.plot(self.OTFB_4.profile.bin_centers*1e6,
#                     np.abs(self.OTFB_4.V_fine_tot),
#                     color=colors[i])
#             plt.plot(self.OTFB_4.rf_centers*1e6,
#                      np.abs(self.OTFB_4.V_coarse_tot), color=colors[i],
#                      linestyle='', marker='.')
            self.OTFB_5.track_no_beam()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            self.OTFB_4.profile.bin_centers, self.OTFB_4.rf_centers,
            self.OTFB_4.V_coarse_ind_gen + self.OTFB_5.V_coarse_ind_gen)

        self.V_corr, alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr = 0.5*np.pi - alpha_sum \
            - self.rf.phi_rf[0, self.rf.counter[0]]


class SPSOneTurnFeedback(object):

    r'''Voltage feedback around a travelling wave cavity with given amount of
    sections. The quantities of the LLRF system cover one turn with a coarse
    resolution.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        Beam profile object
    n_sections : int
        Number of sections in the cavities
    n_cavities : int
        Number of cavities of the same type
    V_part : float
        Voltage partition for the given n_cavities; in range (0,1)
    G_tx : float
        Transmitter gain [A/V]; default is :math:`(50 \Omega)^{-1}`
    open_loop : int(bool)
        Open (0) or closed (1) feedback loop; default is 1

    Attributes
    ----------
    TWC : class
        A TravellingWaveCavity type class
    counter : int
        Counter of the current time step
    omega_c : float
        Carrier revolution frequency [1/s] at the current time step
    omega_r : const float
        Resonant revolution frequency [1/s] of the travelling wave cavities
    n_coarse : int
        Number of bins for the coarse grid (equals harmonic number)
    V_gen : complex array
        Generator voltage [V] of the present turn in (I,Q) coordinates
    V_gen_prev : complex array
        Generator voltage [V] of the previous turn in (I,Q) coordinates
    V_fine_ind_beam : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the fine grid
        used for tracking the beam
    V_coarse_ind_beam : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the coarse grid used
        tracking the LLRF
    V_coarse_ind_gen : complex array
        Generator-induced voltage [V] in (I,Q) coordinates on the coarse grid
        used tracking the LLRF
    V_coarse_tot : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates which is used
        for tracking the LLRF
    V_fine_tot : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates which is used
        for tracking the beam
    a_comb : float
        Recursion constant of the comb filter; :math:`a_{\mathsf{comb}}=15/16`
    n_mov_av : const int
        Number of points for moving average modelling cavity response;
        :math:`n_{\mathsf{mov.av.}} = \frac{f_r}{f_{\mathsf{bw,cav}}}`, where
        :math:`f_r` is the cavity resonant frequency of TWC_4 and TWC_5
    logger : logger
        Logger of the present class

    '''

    def __init__(self, RFStation, Beam, Profile_, n_sections, n_cavities=2,
                 V_part=4/9, G_llrf=10, G_tx=0.5, a_comb=15/16,
                 Commissioning=CavityFeedbackCommissioning()):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Commissioning options
        self.open_loop = Commissioning.open_loop
        self.logger.debug("Opening overall OTFB loop")
        self.open_FB = Commissioning.open_FB
        self.logger.debug("Opening feedback of drive correction")
        self.open_drive = Commissioning.open_drive
        self.logger.debug("Opening drive to generator")

        # Read input
        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile_
        self.n_cavities = int(n_cavities)
        if self.n_cavities < 1:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_cavities has invalid value!")
        self.V_part = float(V_part)
        if self.V_part*(1 - self.V_part) < 0:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: V_part" +
                               " should be in range (0,1)!")

        # Gain settings
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx)

        # 200 MHz travelling wave cavity (TWC) model
        if n_sections in [4, 5]:
            self.TWC = eval("SPS" + str(n_sections) + "Section200MHzTWC()")
        else:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_sections has invalid value!")
        self.logger.debug("SPS OTFB cavities: %d, sections: %d, voltage" +
                          " partition %.2f, gain: %.2e", self.n_cavities,
                          n_sections, self.V_part, self.G_tx)

        # TWC resonant frequency
        self.omega_r = self.TWC.omega_r

        # Initialise bunch-by-bunch voltage array with LENGTH OF PROFILE
        self.V_fine_tot = np.zeros(self.profile.n_slices, dtype=complex)

        # Length of arrays in LLRF
        self.n_coarse = int(self.rf.harmonic[0, 0])

        # Array to hold the bucket-by-bucket voltage with LENGTH OF LLRF
        self.V_coarse_tot = np.zeros(self.n_coarse, dtype=complex)

        # Centers of the RF-buckets
        self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.rf.t_rf[0, 0]

        # TODO: Bin size can change! Update affected variables!!
        self.logger.debug("Length of arrays in generator path %d",
                          self.n_coarse)

        # Initialise comb filter
        self.dV_gen_prev = np.zeros(self.n_coarse, dtype=complex)
        self.a_comb = float(a_comb)

        # Initialise cavity filter (moving average)
        self.n_mov_av = int(self.TWC.tau/self.rf.t_rf[0, 0])
        self.logger.debug("Moving average over %d points", self.n_mov_av)
        # TODO: update condition for new n_mov_av
        if self.n_mov_av < 2:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: profile has to" +
                               " have at least 12.5 ns resolution!")
        self.dV_mov_av_prev = np.zeros(self.n_coarse, dtype=complex)

        # Initialise generator-induced voltage
        self.I_gen_prev = np.zeros(self.n_mov_av, dtype=complex)

        # Pre-compute factor for semi-analytic method
        self.pre_compute_semi_analytic_factor(self.rf_centers)

        self.logger.info("Class initialized")

    def track(self):
        """Turn-by-turn tracking method."""

        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0, self.counter]
        # Present delay time
        self.n_delay = int((self.rf.t_rev[self.counter] - self.TWC.tau)
                           / self.rf.t_rf[0, self.counter])

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_c, self.rf_centers)
        self.TWC.impulse_response_beam(self.omega_c, self.profile.bin_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.generator_induced_voltage()

        # Beam-induced voltage from beam profile
        self.beam_induced_voltage(lpf=False)

        # Sum and generator- and beam-induced voltages for coarse grid
        self.V_coarse_tot = self.V_coarse_ind_gen + self.V_coarse_ind_beam
        # Obtain generator-induced voltage on the fine grid by interpolation
        self.V_fine_tot = self.V_fine_ind_beam \
            + np.interp(self.profile.bin_centers,
                        self.rf_centers, self.V_coarse_ind_gen)

    def track_no_beam(self):
        """Initial tracking method, before injecting beam."""

        # Present time step
        self.counter = int(0)
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0, self.counter]
        # Present delay time
        self.n_delay = int((self.rf.t_rev[self.counter] - self.TWC.tau)
                           / self.rf.t_rf[0, self.counter])

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_c, self.rf_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.generator_induced_voltage()
        self.logger.debug("Total voltage to generator %.3e V",
                          np.mean(np.absolute(self.V_gen)))
        self.logger.debug("Total current from generator %.3e A",
                          np.mean(np.absolute(self.I_gen))
                          / self.profile.bin_size)

        # Without beam, total voltage equals generator-induced voltage
        self.V_coarse_tot = self.V_coarse_ind_gen

        self.logger.debug(
            "Average generator voltage, last half of array %.3e V",
            np.mean(np.absolute(self.V_coarse_ind_gen[int(0.5*self.n_coarse):])))

    def llrf_model(self):
        """Models the LLRF part of the OTFB.

        Attributes
        ----------
        V_set : complex array
            Voltage set point [V] in (I,Q); :math:`V_{\mathsf{set}}`, amplitude
            proportional to voltage partition
        dV_gen : complex array
            Generator voltage [V] in (I,Q);
            :math:`dV_{\mathsf{gen}} = V_{\mathsf{set}} - V_{\mathsf{tot}}`
        """

        # Voltage set point of current turn (I,Q); depends on voltage partition
        # Sinusoidal voltage completely in Q

        self.V_set = polar_to_cartesian(
            self.V_part*self.rf.voltage[0, self.counter],
            0.5*np.pi - self.rf.phi_rf[0, self.counter])

        # Convert to array
        self.V_set *= np.ones(self.n_coarse)

        # Difference of set point and actual voltage
        self.dV_gen = self.V_set - self.open_loop*self.V_coarse_tot

        # Closed-loop gain
        self.dV_gen *= self.G_llrf
        self.logger.debug("Set voltage %.6f MV",
                          1e-6*np.mean(np.absolute(self.V_set)))
        self.logger.debug("Antenna voltage %.6f MV",
                          1e-6*np.mean(np.absolute(self.V_coarse_tot)))
        self.logger.debug("Voltage error %.6f MV",
                          1e-6*np.mean(np.absolute(self.dV_gen)))

        # One-turn delay comb filter; memorise the value of the previous turn
        self.dV_gen = comb_filter(self.dV_gen_prev, self.dV_gen, self.a_comb)
        self.dV_gen_prev = np.copy(self.dV_gen)

        # Modulate from omega_rf to omega_r
        self.dV_gen = modulator(self.dV_gen, self.omega_c, self.omega_r,
                                self.rf.t_rf[0, self.counter])

        # Shift signals with the delay time
        dV_gen_in = np.copy(self.dV_gen)
        self.dV_gen = np.concatenate((self.dV_mov_av_prev[-self.n_delay:],
                                      self.dV_gen[:self.n_coarse-self.n_delay]))

        # Cavity filter: CIRCULAR moving average over filling time
        # Memorize last points of previous turn for beginning of next turn
        self.dV_gen = moving_average(
            self.dV_gen, self.n_mov_av,
            x_prev=self.dV_mov_av_prev[-self.n_delay-self.n_mov_av+1:
                                       -self.n_delay])

        self.dV_mov_av_prev = np.copy(dV_gen_in)

    def generator_induced_voltage(self):
        r"""Calculates the generator-induced voltage. The transmitter model is
        a simple linear gain [C/V] converting voltage to charge.

        .. math:: I = G_{\mathsf{tx}}\,\frac{V}{R_{\mathsf{gen}}} \, ,

        where :math:`R_{\mathsf{gen}}` is the generator resistance,
        :py:attr:`llrf.impulse_response.TravellingWaveCavity.R_gen`

        Attributes
        ----------
        I_gen : complex array
            RF component of the generator charge [C] at the present time step
        V_coarse_ind_gen : complex array
            Induced voltage [V] from generator-cavity interaction

        """

        # Add correction to the drive already existing
        self.V_gen = self.open_FB * modulator(self.dV_gen, self.omega_r,
                                              self.omega_c,
                                              self.rf.t_rf[0, self.counter]) \
            + self.open_drive*self.V_set

        # Generator charge from voltage, transmitter model
        self.I_gen = self.G_tx*self.V_gen\
            / self.TWC.R_gen*self.rf.t_rf[0, self.counter]

        # Circular convolution: attach last points of previous turn
        self.I_gen = np.concatenate((self.I_gen_prev, self.I_gen))

        # Generator-induced voltage
        self.induced_voltage('gen')
        # Update memory of previous turn
        self.I_gen_prev = self.I_gen[-self.n_mov_av:]

    def induced_voltage(self, name):
        r"""Generation of beam- or generator-induced voltage from the beam or
        generator current, at a given carrier frequency and turn. The induced
        voltage :math:`V(t)` is calculated from the impulse response matrix
        :math:`h(t)` as follows:

        .. math::
            \left( \begin{matrix} V_I(t) \\
            V_Q(t) \end{matrix} \right)
            = \left( \begin{matrix} h_s(t) & - h_c(t) \\
            h_c(t) & h_s(t) \end{matrix} \right)
            * \left( \begin{matrix} I_I(t) \\
            I_Q(t) \end{matrix} \right) \, ,

        where :math:`*` denotes convolution,
        :math:`h(t)*x(t) = \int d\tau h(\tau)x(t-\tau)`. If the carrier
        frequency is close to the cavity resonant frequency, :math:`h_c = 0`.

        :see also: :py:class:`llrf.impulse_response.TravellingWaveCavity`

        The impulse response is made to be the same length as the beam profile.

        """

        self.logger.debug("Matrix convolution for V_ind")

        if name == "beam":
            # Compute the beam-induced voltage on the fine grid by convolution
            self.__setattr__("V_fine_ind_"+name,
                             self.matr_conv(self.__getattribute__("I_"+name),
                                            self.TWC.__getattribute__("h_"+name)))

            self.V_fine_ind_beam = -self.n_cavities \
                * self.V_fine_ind_beam[:self.profile.n_slices]

            # Beam-induced voltage on the coarse grid from semi-analytic method
            self.V_coarse_ind_beam = -self.n_cavities \
                * self.beam_induced_voltage_semi_analytic()

        elif name == "gen":
            self.__setattr__("V_coarse_ind_"+name,
                             self.matr_conv(self.__getattribute__("I_"+name),
                                            self.TWC.__getattribute__("h_"+name)))

            # Circular convolution
            self.V_coarse_ind_gen = +self.n_cavities \
                * self.V_coarse_ind_gen[self.n_mov_av:
                                        self.n_coarse+self.n_mov_av]

    def beam_induced_voltage(self, lpf=False):
        """Calculates the beam-induced voltage

        Parameters
        ----------
        lpf : bool
            Apply low-pass filter for beam current calculation;
            default is False

        Attributes
        ----------
        I_beam : complex array
            RF component of the beam charge [C] at the present time step
        V_coarse_ind_beam : complex array
            Induced voltage [V] from beam-cavity interaction on the coarse grid
        V_fine_ind_beam : complex array
            Induced voltage [V] from beam-cavity interaction on the fine grid
        """

        # Beam current from profile
        self.I_beam = rf_beam_current(self.profile, self.omega_c,
                                      self.rf.t_rev[self.counter], lpf=lpf)

        # Beam-induced voltage
        self.induced_voltage('beam')

    def pre_compute_semi_analytic_factor(self, time):
        r""" Pre-computes factor for semi-analytic method, which is used to
        compute the beam-induced voltage on the coarse grid.

        Parameters
        ----------
        time : float array [s]
            Time array at which to compute the beam-induced voltage

        Attributes
        ----------
        profile_coarse : class
            Beam profile with 20 bins per RF-bucket
        semi_analytic_factor : complex array [:math:`\Omega\,s`]
            Factor that is used to compute the beam-induced voltage
        """

        self.logger.info("Pre-computing semi-analytic factor")

        n_slices_per_bucket = 20

        n_buckets = int(np.round(
            (self.profile.cut_right - self.profile.cut_left)
            / self.rf.t_rf[0, 0]))

        self.profile_coarse = Profile(self.beam, CutOptions=CutOptions(
            cut_left=self.profile.cut_left,
            cut_right=self.profile.cut_right,
            n_slices=n_buckets*n_slices_per_bucket))

        # pre-factor [Ohm s]

        pre_factor = 2*self.TWC.R_beam / self.TWC.tau**2 / self.omega_r**3

        # Matrix of time differences [1]
        dt1 = np.zeros(shape=(len(time), self.profile_coarse.n_slices))

        for i in range(len(time)):
            dt1[i] = (time[i] - self.profile_coarse.bin_centers) * self.omega_r

#        dt2 = dt1 - self.TWC.tau * self.omega_r

#        phase1 = np.exp(-1j * dt1)
        phase = np.exp(-1j * self.TWC.tau * self.TWC.omega_r)

#        diff1 = 2j - dt1 + self.TWC.tau * self.omega_r

#        diff2 = (2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)

        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r
               + (2j - dt1 + self.TWC.tau*self.omega_r) * np.exp(-1j * dt1))\
            * np.sign(dt1) \
            - ((2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)
               + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
            * np.sign(dt1 - self.TWC.tau * self.omega_r) \
            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)

#        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r + diff2) * np.sign(dt1) \
#            - (diff2 + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
#                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
#            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)

#        tmp = (diff1.conjugate() + diff2) * np.sign(dt1) \
#            - (diff2 + diff1.conjugate() * phase) \
#                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
#            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)

        tmp *= pre_factor

        self.semi_analytic_factor = np.diff(tmp)

    def beam_induced_voltage_semi_analytic(self):
        r"""Computes the beam-induced voltage in (I,Q) at the present carrier
        frequency :math:`\omega_c` using the semi-analytic method. It requires
        that pre_compute_semi_analytic_factor() was called previously.

        Returns
        -------
        complex array [V]
            Beam-induced voltage in (I,Q) at :math:`\omega_c`
        """

        # Update the coarse profile
        self.profile_coarse.track()

        # Slope of line segments [A/s]
        kappa = self.beam.ratio*self.beam.Particle.charge*e \
            * np.diff(self.profile_coarse.n_macroparticles) \
            / self.profile_coarse.bin_size**2

        return np.exp(1j*self.rf_centers*self.omega_c)\
            * np.sum(self.semi_analytic_factor * kappa, axis=1)

    def matr_conv(self, I, h):
        """Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        return scipy.signal.fftconvolve(I, h, mode='full')

    def call_conv(self, signal, kernel):
        """Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1)
        libblond.convolution(signal.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(signal)),
                             kernel.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(len(kernel)),
                             result.ctypes.data_as(ctypes.c_void_p))

        return result
