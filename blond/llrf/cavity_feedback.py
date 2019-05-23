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
import logging
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy
from scipy.constants import e

from ..llrf.signal_processing import comb_filter, cartesian_to_polar, \
    polar_to_cartesian, modulator, moving_average, rf_beam_current
from ..utils import bmath as bm
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
            #FeedbackError
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
        
        if debug:
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0,1, self.turns))
            plt.figure('voltage')
            plt.clf()
            plt.grid()

        for i in range(self.turns):
            #            print('OTFB pre-tracking iteration ', i)
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_4.track_no_beam()
            if debug:
                plt.plot(self.OTFB_4.profile.bin_centers*1e6,
                         np.abs(self.OTFB_4.V_fine_tot), color=colors[i])
                plt.plot(self.OTFB_4.rf_centers*1e6,
                         np.abs(self.OTFB_4.V_coarse_tot), color=colors[i],
                         linestyle='', marker='.')
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
            #FeedbackError
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_cavities has invalid value!")
        self.V_part = float(V_part)
        if self.V_part*(1 - self.V_part) < 0:
            #FeedbackError
            raise RuntimeError("ERROR in SPSOneTurnFeedback: V_part" +
                               " should be in range (0,1)!")

        # Gain settings
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx)

        # 200 MHz travelling wave cavity (TWC) model
        if n_sections in [4, 5]:
            self.TWC = eval("SPS" + str(n_sections) + "Section200MHzTWC()")
        else:
            #FeedbackError
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
            #FeedbackError
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
        bm.convolve(signal, kernel, result)

        #  libblond.convolution(signal.ctypes.data_as(ctypes.c_void_p),
        #                       ctypes.c_int(len(signal)),
        #                       kernel.ctypes.data_as(ctypes.c_void_p),
        #                       ctypes.c_int(len(kernel)),
        #                       result.ctypes.data_as(ctypes.c_void_p))

        return result



class LHCRFFeedback(object):
    r'''RF Feedback settings for LHC ACS cavity loop.

    Parameters
    ----------
    alpha : float
        One-turn feedback memory parameter; default is 15/16
    d_phi_ad : float
        Phase misalignment of digital FB w.r.t. analog FB [deg]
    G_a : float
        Analog FB gain [1]
    G_d : float
        Digital FB gain, w.r.t. analog gain [1]
    G_o : float
        One-turn feedback gain
    tau_a : float
        Analog FB delay time [s]
    tau_d : float
        Digital FB delay time [s]
    tau_o : float
        AC-coupling delay time of one-turn feedback [s]
    open_drive : bool
        Open (True) or closed (False) cavity loop at drive; default is False
    open_loop : bool
        Open (True) or closed (False) cavity loop at RFFB; default is False
    open_otfb : bool
        Open (true) or closed (False) one-turn feedback; default is False
    open_rffb : bool
        Open (True) or closed (False) RFFB; default is False

    Attributes
    ----------
    d_phi_ad : float
        Phase misalignment of digital FB w.r.t. analog FB [rad]
    open_drive : int(bool)
        Open (0) or closed (1) cavity loop at drive; default is 1
    open_loop : int(bool)
        Open (0) or closed (1) cavity loop at RFFB; default is 1
    open_rffb : int(bool)
        Open (0) or closed (1) RFFB; default is 1
    '''

    def __init__(self, alpha=15/16, d_phi_ad=0, G_a=0.00001, G_d=10, G_o=10,
                 tau_a=170e-6, tau_d=400e-6, tau_o=110e-6, open_drive=False,
                 open_loop=False, open_otfb=False, open_rffb=False,
                 excitation=False, excitation_otfb=False, seed1=1234,
                 seed2=7564):

        # Import variables
        self.alpha = alpha
        self.d_phi_ad = d_phi_ad*np.pi/180
        self.G_a = G_a
        self.G_d = G_d
        self.G_o = G_o
        self.tau_a = tau_a
        self.tau_d = tau_d
        self.tau_o = tau_o
        self.excitation = excitation
        self.excitation_otfb = excitation_otfb
        self.seed1 = seed1
        self.seed2 = seed2

        # Multiply with zeros if open == True
        self.open_drive = int(np.invert(bool(open_drive)))
        self.open_drive_inv = int(bool(open_drive))
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_otfb = int(np.invert(bool(open_otfb)))
        self.open_rffb = int(np.invert(bool(open_rffb)))


    def generate_white_noise(self, n_points):

        rnd.seed(self.seed1)
        r1 = rnd.random_sample(n_points)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(n_points)

        return np.exp(2*np.pi*1j*r1) * np.sqrt(-2*np.log(r2))



class LHCCavityLoop(object):
    r'''Cavity loop to regulate the RF voltage in the LHC ACS cavities.
    The loop contains a generator, a switch-and-protect device, an RF FB and a
    OTFB. The arrays of the LLRF system cover one turn with exactly one tenth
    of the harmonic (i.e.\ the typical sampling time is about 25 ns).

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Profile : class
        Beam profile object
    RFFB : class
        LHCRFFeedback type class containing RF FB gains and delays
    f_c : float
        Central cavity frequency [Hz]
    G_gen : float
        Overall driver chain gain [1]
    I_gen_offset : float
        Generator current offset [A]
    n_cav : int
        Number of cavities per beam (default is 8)
    n_pretrack : int
        Number of turns to pre-track without beam (default is 1)
    Q_L : float
        Cavity loaded quality factor (default is 20000)
    R_over_Q : float
        Cavity R/Q [Ohm] (default is 45 Ohms)
    tau_loop : float
        Total loop delay [s]
    tau_otfb : float
        Total loop delay as seen by OTFB [s]
    Ts : float
        Sampling time of the LLRF loops [s] (default is 25 ns)

    Attributes
    ----------
    n_coarse : int
        Number of bins for the coarse grid (equals harmonic number)
    t_centers : float array
        Time shift w.r.t. clock, corresponding to voltage arrays
    omega_c : float
        Central cavity revolution frequency [1/s]
    V_coarse_tot : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates which is used
        for tracking the LLRF
    logger : logger
        Logger of the present class
    '''

    def __init__(self, RFStation, Profile, f_c=400.789e6, G_gen=1,
                 I_gen_offset=0, n_cav=8, n_pretrack=200, Q_L=20000,
                 R_over_Q=45, tau_loop=650e-9, tau_otfb=1472e-9,
                 RFFB=LHCRFFeedback()):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("LHCCavityLoop class initialized")

        # Import classes and parameters
        self.rf = RFStation
        self.profile = Profile
        self.RFFB = RFFB
        self.I_gen_offset = I_gen_offset
        self.G_gen = G_gen
        self.n_cav = n_cav
        self.n_pretrack = n_pretrack
        self.omega_c = 2*np.pi*f_c
        # TODO: implement optimum loaded Q
        self.Q_L = Q_L
        self.R_over_Q = R_over_Q
        self.tau_loop = tau_loop
        self.tau_otfb = tau_otfb
        #self.T_s = T_s
        self.logger.debug("Cavity loaded Q is %.0f", self.Q_L)

        # Import RF FB properties
        self.open_drive = self.RFFB.open_drive
        self.open_drive_inv = self.RFFB.open_drive_inv
        self.open_loop = self.RFFB.open_loop
        self.open_otfb = self.RFFB.open_otfb
        self.open_rffb = self.RFFB.open_rffb
        self.alpha = self.RFFB.alpha
        self.d_phi_ad = self.RFFB.d_phi_ad
        self.G_a = self.RFFB.G_a
        self.G_d = self.RFFB.G_d
        self.G_o = self.RFFB.G_o
        self.tau_a = self.RFFB.tau_a
        self.tau_d = self.RFFB.tau_d
        self.tau_o = self.RFFB.tau_o
        self.excitation = self.RFFB.excitation
        self.excitation_otfb = self.RFFB.excitation_otfb

        # Length of arrays in LLRF  #TODO: could change over time
        self.n_coarse = int(self.rf.harmonic[0, 0]/10)
        self.logger.debug("Length of arrays in generator path %d",
                          self.n_coarse)

        # Initialise antenna voltage to set point value
        self.update_variables()
        self.logger.debug("Relative detuning is %.4e", self.detuning)

        self.V_ANT = np.zeros(2*self.n_coarse, dtype=complex)
        self.V_EXC = np.zeros(2*self.n_coarse, dtype=complex)
        self.V_FB_IN = np.zeros(2*self.n_coarse, dtype=complex)
        self.V_OTFB = np.zeros(2*self.n_coarse, dtype=complex)
        self.V_OTFB_INT = np.zeros(2*self.n_coarse, dtype=complex)
        self.I_GEN = np.zeros(2*self.n_coarse, dtype=complex)
        self.I_BEAM = np.zeros(2*self.n_coarse, dtype=complex)
        self.I_TEST = np.zeros(2 * self.n_coarse, dtype=complex)

        # Scalar variables
        self.V_a_in_prev = 0
        self.V_a_out_prev = 0
        self.V_d_out_prev = 0
        self.V_fb_in_prev = 0
        self.V_otfb_prev = 0

        # Pre-track without beam
        self.logger.debug("Track without beam for %d turns", self.n_pretrack)
        if self.excitation:
            self.logger.debug("Injecting noise in voltage set point")
            self.track_no_beam_excitation(self.n_pretrack)
        elif self.excitation_otfb:
            self.logger.debug("Injecting noise at OTFB output")
            self.track_no_beam_excitation_otfb(self.n_pretrack)
        else:
            self.track_no_beam(self.n_pretrack)


    def cavity_response(self):
        r'''ACS cavity reponse model'''

        self.V_ANT[self.ind] = self.I_GEN[self.ind-1]*self.R_over_Q* \
            self.samples + self.V_ANT[self.ind-1]*(1 - 0.5*self.samples/ \
            self.Q_L + 1j*self.detuning*self.samples) - \
            self.I_BEAM[self.ind-1]*0.5*self.R_over_Q*self.samples


    def generator_current(self):
        r'''Generator response

        Attributes
        I_TEST : complex array
            Test point for open loop measurements (when injecting a generator
            offset)
        '''

        # From V_swap_out in closed loop, constant in open loop
        # TODO: missing terms for changing voltage and beam current
        self.I_TEST[self.ind] = self.G_gen*self.V_swap_out
        self.I_GEN[self.ind] = self.open_drive*self.I_TEST[self.ind] + \
            self.open_drive_inv*self.I_gen_offset


    def generator_power(self):
        r'''Calculation of generator power from generator current'''

        return 0.5*self.R_over_Q*self.Q_L*np.absolute(self.I_GEN)**2


    def one_turn_feedback(self):

        # AC coupling at input
        self.V_OTFB_INT[self.ind] = self.V_OTFB_INT[self.ind - 1] * (
            1 - self.T_s / self.tau_o) + \
            self.V_FB_IN[self.ind-self.n_coarse+self.n_otfb] - self.V_FB_IN[self.ind-self.n_coarse+self.n_otfb-1]
        #self.V_FB_IN[self.ind - self.n_coarse + self.n_delay] - self.V_FB_IN[
        #    self.ind - self.n_coarse + self.n_delay - 1]
        # OTFB response
        self.V_OTFB[self.ind] = self.alpha*self.V_OTFB[
        self.ind-self.n_coarse] \
           + self.G_o*(1 - self.alpha)*self.V_OTFB_INT[self.ind] #-self.n_coarse+self.n_delay]
        # AC coupling at output
        self.V_otfb = self.V_otfb_prev*(1 - self.T_s/self.tau_o) + \
            self.V_OTFB[self.ind] - self.V_OTFB[self.ind-1]
        # Update memory
        self.V_otfb_prev = self.V_otfb



    def rf_beam_current(self):
        r'''RF beam current calculation from beam profile'''

        # Beam current at rf frequency from profile
        self.I_BEAM_FINE = rf_beam_current(self.profile, self.omega,
            self.rf.t_rev[self.counter], lpf=False)/self.T_s  #self.rf.t_rev[self.counter] #self.profile.bin_size
        self.I_BEAM_FINE *= np.exp(-1j*0.5*np.pi) # 90 deg phase shift w.r.t. V_set in real

        # Find which index in fine grid matches index in coarse grid
        ind_fine = np.floor(self.profile.bin_centers/self.T_s
                           - 0.5*self.profile.bin_size)
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

        # Pick total current within one coarse grid
        self.I_BEAM[self.n_coarse] = np.sum(self.I_BEAM_FINE[np.arange(indices[0])])
        for i in range(1,len(indices)):
            self.I_BEAM[self.n_coarse+i] = np.sum(self.I_BEAM_FINE[np.arange(indices[i-1],indices[i])])


    def rf_feedback(self):
        r'''Analog and digital RF feedback response'''

        # Calculate voltage difference to act on
        self.V_fb_in = (self.V_SET[self.ind] -
                        self.open_loop*self.V_ANT[self.ind-self.n_delay])
        self.V_FB_IN[self.ind] = self.V_fb_in

        # On the analog branch, OTFB can contribute
        self.one_turn_feedback()
        #self.V_a_in = int(np.invert(bool(self.excitation_otfb)))*self.V_fb_in \
        self.V_a_in = self.V_fb_in + self.open_otfb*self.V_otfb \
            + int(bool(self.excitation_otfb))*self.V_EXC[self.ind]

        # Output of analog feedback (separate branch)
        self.V_a_out = self.V_a_out_prev*(1 - self.T_s/self.tau_a) + \
            self.G_a*(self.V_a_in - self.V_a_in_prev)

        # Output of digital feedback (separate branch)
        self.V_d_out = self.V_d_out_prev*(1 - self.T_s/self.tau_d) + \
            self.T_s/self.tau_d*self.G_a*self.G_d*np.exp(1j*self.d_phi_ad)*\
            self.V_fb_in_prev

        # Total output: sum of analog and digital feedback
        self.V_fb_out = self.open_rffb*(self.V_a_out + self.V_d_out)

        # Update memory
        self.V_a_in_prev = self.V_a_in
        self.V_a_out_prev = self.V_a_out
        self.V_d_out_prev = self.V_d_out
        self.V_fb_in_prev = self.V_fb_in


    def set_point(self):
        r'''Voltage set point'''

        V_set = polar_to_cartesian(self.rf.voltage[0, self.counter]/self.n_cav,
            self.rf.phi_rf[0, self.counter])

        return self.open_drive*V_set*np.ones(self.n_coarse)


    def swap(self):
        r'''Model of the Switch and Protect module: clamping of the output
        power above a given input power.'''

        #TODO: to be implemented
        self.V_swap_out = self.V_fb_out


    def track(self):
        r'''Tracking with beam'''

        self.update_variables()
        self.update_arrays()
        self.update_set_point()
        self.rf_beam_current()
        self.track_one_turn()


    def track_simple(self, I_rf_pk):
        r'''Simplified model with proportional gain and step beam current of
        1000 samples lengthBM7_ACS_with_beam.py

        Parameters
        ----------
        I_rf_peak : float
            Peak RF current
        '''

        self.update_variables()
        self.update_arrays()
        self.update_set_point()
        self.I_BEAM[self.n_coarse:self.n_coarse+1000] = 1j*I_rf_pk

        for i in range(self.n_coarse):
            self.ind = i + self.n_coarse
            self.cavity_response()
            self.V_fb_out = self.G_a*(self.V_SET[self.ind] - self.V_ANT[self.ind-self.n_delay])
            self.I_GEN[self.ind] = self.V_fb_out + self.V_SET[self.ind]/(self.R_over_Q)*(0.5/self.Q_L -1j*self.detuning) + 0.5*1j*I_rf_pk


    def track_one_turn(self):
        r'''Single-turn tracking, index by index.'''

        for i in range(self.n_coarse):
            self.ind = i + self.n_coarse
            self.cavity_response()
            self.rf_feedback()
            self.swap()
            self.generator_current()


    def track_no_beam_excitation(self, n_turns):
        r'''Pre-tracking for n_turns turns, without beam. With excitation; set
        point from white noise. V_EXC_IN and V_EXC_OUT can be used to measure
        the transfer function of the system at set point.

        Attributes
        ----------
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse*n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse*n_turns
            elements
        '''

        self.V_EXC_IN = 1000*self.RFFB.generate_white_noise(self.n_coarse*n_turns)
        self.V_EXC_OUT = np.zeros(self.n_coarse*n_turns, dtype=complex)
        self.V_SET = np.concatenate((np.zeros(self.n_coarse, dtype=complex),
                                     self.V_EXC_IN[0:self.n_coarse]))
        self.track_one_turn()
        self.V_EXC_OUT[0:self.n_coarse] = self.V_ANT[self.n_coarse:2*self.n_coarse]
        for n in range(1, n_turns):
            self.update_arrays()
            self.update_set_point_excitation(self.V_EXC_IN, n)
            self.track_one_turn()
            self.V_EXC_OUT[n*self.n_coarse:(n+1)*self.n_coarse] = \
                self.V_ANT[self.n_coarse:2*self.n_coarse]


    def track_no_beam_excitation_otfb(self, n_turns):
        r'''Pre-tracking for n_turns turns, without beam. With excitation; set
        point from white noise. V_EXC_IN and V_EXC_OUT can be used to measure
        the transfer function of the system at otfb.

        Attributes
        ----------
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse*n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse*n_turns
            elements
        '''

        self.V_EXC_IN = 10000*self.RFFB.generate_white_noise(self.n_coarse*n_turns)
        self.V_EXC_OUT = np.zeros(self.n_coarse*n_turns, dtype=complex)
        #self.V_SET = np.concatenate((np.zeros(self.n_coarse, dtype=complex),
        #                             self.set_point()))
        self.V_SET = np.zeros(2*self.n_coarse, dtype=complex)
        self.V_EXC = np.concatenate((np.zeros(self.n_coarse, dtype=complex),
                                     self.V_EXC_IN[0:self.n_coarse]))

        self.track_one_turn()
        self.V_EXC_OUT[0:self.n_coarse] = self.V_FB_IN[self.n_coarse:2*self.n_coarse]
        for n in range(1, n_turns):
            self.update_arrays()
            self.V_EXC = np.concatenate(
                (np.zeros(self.n_coarse, dtype=complex),
                 self.V_EXC_IN[n*self.n_coarse:(n+1)*self.n_coarse]))

            #self.update_set_point()
            self.track_one_turn()
            self.V_EXC_OUT[n*self.n_coarse:(n+1)*self.n_coarse] = \
                self.V_FB_IN[self.n_coarse:2*self.n_coarse]


    def track_no_beam(self, n_turns):
        r'''Pre-tracking for n_turns turns, without beam. No excitation; set
        point from design RF voltage.'''

        # Initialise set point voltage
        self.V_SET = np.concatenate((np.zeros(self.n_coarse, dtype=complex),
                                     self.set_point()))
        self.track_one_turn()
        for n in range(1, n_turns):
            self.update_arrays()
            self.update_set_point()
            self.track_one_turn()


    def update_arrays(self):
        r'''Moves the array indices by one turn (n_coarse points) from the
        present turn to prepare the next turn. All arrays except for V_SET.'''

        # TODO: update n_coarse and array sizes
        self.V_ANT = np.concatenate((self.V_ANT[self.n_coarse:],
                                    np.zeros(self.n_coarse, dtype=complex)))
        self.V_FB_IN = np.concatenate((self.V_FB_IN[self.n_coarse:],
                                    np.zeros(self.n_coarse, dtype=complex)))
        self.V_OTFB = np.concatenate((self.V_OTFB[self.n_coarse:],
                                    np.zeros(self.n_coarse, dtype=complex)))
        self.V_OTFB_INT = np.concatenate((self.V_OTFB_INT[self.n_coarse:],
                                    np.zeros(self.n_coarse, dtype=complex)))
        self.I_BEAM = np.concatenate((self.I_BEAM[self.n_coarse:],
                                     np.zeros(self.n_coarse, dtype=complex)))
        self.I_GEN = np.concatenate((self.I_GEN[self.n_coarse:],
                                    np.zeros(self.n_coarse, dtype=complex)))
        self.I_TEST = np.concatenate((self.I_TEST[self.n_coarse:],
                                     np.zeros(self.n_coarse, dtype=complex)))


    def update_set_point(self):
        r'''Updates the set point for the next turn based on the design RF
        voltage.'''

        self.V_SET = np.concatenate((self.V_SET[self.n_coarse:],
                                    self.set_point()))


    def update_set_point_excitation(self, excitation, turn):
        r'''Updates the set point for the next turn based on the excitation to
        be injected.'''

        self.V_SET = np.concatenate((self.V_SET[self.n_coarse:],
            excitation[turn*self.n_coarse:(turn+1)*self.n_coarse]))


    def update_variables(self):
        r'''Update counter and frequency-dependent variables in a given turn'''

        # Present time step
        self.counter = self.rf.counter[0]
        # Present sampling time
        self.T_s = self.rf.t_rev[self.counter]/self.n_coarse
        # Delay time
        self.n_delay = int(self.tau_loop/self.T_s)
        self.n_otfb = int(self.tau_otfb/self.T_s)
        # Present rf frequency
        self.omega = self.rf.omega_rf[0, self.counter]
        # Present detuning
        self.d_omega = self.omega_c - self.omega
        # Dimensionless quantities
        self.samples = self.omega*self.T_s
        self.detuning = self.d_omega/self.omega


    @staticmethod
    def half_detuning(imag_peak_beam_current, R_over_Q, rf_frequency, voltage):
        '''Optimum detuning for half-detuning scheme

        Parameters
        ----------
        peak_beam_current : float
            Peak RF beam current
        R_over_Q : float
            Cavity R/Q
        rf_frequency : float
            RF frequency
        voltage : float
            RF voltage amplitude in the cavity

        Returns
        -------
        float
            Optimum detuning (revolution) frequency in the half-detuning scheme
        '''

        return -0.25*R_over_Q*imag_peak_beam_current/voltage*rf_frequency


    @staticmethod
    def half_detuning_power(peak_beam_current, voltage):
        '''RF power consumption half-detuning scheme with optimum detuning

        Parameters
        ----------
        peak_beam_current : float
            Peak RF beam current
        voltage : float
            Cavity voltage

        Returns
        -------
        float
            Optimum detuning (revolution) frequency in the half-detuning scheme
        '''

        return 0.125*peak_beam_current*voltage


    @staticmethod
    def optimum_Q_L(detuning, rf_frequency):
        '''Optimum loaded Q when no real part of RF beam current is present

        Parameters
        ----------
        detuning : float
            Detuning frequency
        rf_frequency : float
            RF frequency

        Returns
        -------
        float
            Optimum loaded Q
        '''

        return np.fabs(0.5*rf_frequency/detuning)


    @staticmethod
    def optimum_Q_L_beam(R_over_Q, real_peak_beam_current, voltage):
        '''Optimum loaded Q when a real part of RF beam current is present

        Parameters
        ----------
        peak_beam_current : float
            Peak RF beam current
        R_over_Q : float
            Cavity R/Q
        voltage : float
            Cavity voltage

        Returns
        -------
        float
            Optimum loaded Q
        '''

        return voltage/(R_over_Q*real_peak_beam_current)

