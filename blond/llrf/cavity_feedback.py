# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
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
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.constants import e
import sys

from ..llrf.signal_processing import comb_filter, cartesian_to_polar, \
    polar_to_cartesian, modulator, moving_average, rf_beam_current
from ..llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC, SPS5Section200MHzTWC
from ..llrf.signal_processing import feedforward_filter_TWC3, \
    feedforward_filter_TWC4, feedforward_filter_TWC5
from ..utils import bmath as bm
from ..beam.profile import Profile, CutOptions


class CavityFeedbackCommissioning(object):

    def __init__(self, debug=False, open_loop=False, open_FB=False,
                 open_drive=False, open_FF=False):
        """Class containing commissioning settings for the cavity feedback

        Parameters
        ----------
        debug : bool
            Debugging output active (True/False); default is False
        open_loop : int(bool)
            Open (True) or closed (False) cavity loop; default is False
        open_FB : int(bool)
            Open (True) or closed (False) feedback; default is False
        open_drive : int(bool)
            Open (True) or closed (False) drive; default is False
        open_FF : int(bool)
            Open (True) or closed (False) feed-forward; default is False
        """

        self.debug = bool(debug)
        # Multiply with zeros if open == True
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_FB = int(np.invert(bool(open_FB)))
        self.open_drive = int(np.invert(bool(open_drive)))
        self.open_FF = int(np.invert(bool(open_FF)))


class SPSCavityFeedback(object):
    """Class determining the turn-by-turn total RF voltage and phase correction
    originating from the individual cavity feedbacks. Assumes two 4-section and
    two 5-section travelling wave cavities in the pre-LS2 scenario and four
    3-section and two 4-section cavities in the post-LS2 scenario. The voltage
    partitioning is proportional to the number of sections.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        A Profile type class
    G_llrf : float or list
        LLRF Gain [1]; if passed as a float, both 3- and 4-section (4- and
        5-section) cavities have the same G_llrf in the post- (pre-)LS2
        scenario. If passed as a list, the first and second elements correspond
        to the G_llrf of the 3- and 4-section (4- and 5-section) cavity
        feedback in the post- (pre-)LS2 scenario; default is 10
    G_tx : float or list
        Transmitter gain [1] of the cavity feedback; convention same as G_llrf;
        default is 0.5
    a_comb : float
        Comb filter ratio [1]; default is 15/16
    turns :  int
        Number of turns to pre-track without beam
    post_LS2 : bool
        Activates pre-LS2 scenario (False) or post-LS2 scenario (True); default
        is True
    V_part : float
        Voltage partitioning of the shorter cavities; has to be in the range
        (0,1). Default is None and will result in 6/10 for the 3-section
        cavities in the post-LS2 scenario and 4/9 for the 4-section cavities in
        the pre-LS2 scenario

    Attributes
    ----------
    OTFB_1 : class
        An SPSOneTurnFeedback type class; 3/4-section cavity for post/pre-LS2
    OTFB_2 : class
        An SPSOneTurnFeedback type class; 4/5-section cavity for post/pre-LS2
    V_sum : complex array
        Vector sum of RF voltage from all the cavities
    V_corr : float array
        RF voltage correction array to be applied in the tracker
    phi_corr : float array
        RF phase correction array to be applied in the tracker
    logger : logger
        Logger of the present class

    """

    def __init__(self, RFStation, Beam, Profile, G_ff=1, G_llrf=10, G_tx=0.5,
                 a_comb=15/16, turns=1000, post_LS2=True, V_part=None,
                 Commissioning=CavityFeedbackCommissioning()):

        # Options for commissioning the feedback
        self.Commissioning = Commissioning

        self.rf = RFStation

        # Parse input for gains
        if type(G_llrf) is list:
            G_ff_1 = G_ff[0]
            G_ff_2 = G_ff[1]
        else:
            G_ff_1 = G_ff
            G_ff_2 = G_ff

        if type(G_llrf) is list:
            G_llrf_1 = G_llrf[0]
            G_llrf_2 = G_llrf[1]
        else:
            G_llrf_1 = G_llrf
            G_llrf_2 = G_llrf

        if type(G_tx) is list:
            G_tx_1 = G_tx[0]
            G_tx_2 = G_tx[1]
        else:
            G_tx_1 = G_tx
            G_tx_2 = G_tx

        # Voltage partitioning has to be a fraction
        if V_part and V_part*(1 - V_part) < 0:
            raise RuntimeError("SPS cavity feedback: voltage partitioning has to be in the range (0,1)!")

        # Voltage partition proportional to the number of sections
        if post_LS2:
            if not V_part:
                V_part = 6/10
            self.OTFB_1 = SPSOneTurnFeedback(RFStation, Beam, Profile, 3,
                                             n_cavities=4, V_part=V_part,
                                             G_ff=float(G_ff_1),
                                             G_llrf=float(G_llrf_1),
                                             G_tx=float(G_tx_1),
                                             a_comb=float(a_comb),
                                             Commissioning=self.Commissioning)
            self.OTFB_2 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                             n_cavities=2, V_part=1-V_part,
                                             G_ff=float(G_ff_2),
                                             G_llrf=float(G_llrf_2),
                                             G_tx=float(G_tx_2),
                                             a_comb=float(a_comb),
                                             Commissioning=self.Commissioning)
        else:
            if not V_part:
                V_part = 4/9
            self.OTFB_1 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                             n_cavities=2, V_part=V_part,
                                             G_ff=float(G_ff_1),
                                             G_llrf=float(G_llrf_1),
                                             G_tx=float(G_tx_1),
                                             a_comb=float(a_comb),
                                             Commissioning=self.Commissioning)
            self.OTFB_2 = SPSOneTurnFeedback(RFStation, Beam, Profile, 5,
                                             n_cavities=2, V_part=1-V_part,
                                             G_ff=float(G_ff_2),
                                             G_llrf=float(G_llrf_2),
                                             G_tx=float(G_tx_2),
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

        self.OTFB_1.track()
        self.OTFB_2.track()

        self.V_sum = self.OTFB_1.V_fine_tot + self.OTFB_2.V_fine_tot

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
            plt.figure('Pre-tracking without beam')
            ax = plt.axes([0.18, 0.1, 0.8, 0.8])
            ax.grid()
            ax.set_ylabel('Voltage [V]')

        for i in range(self.turns):
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_1.track_no_beam()
            if debug:
                ax.plot(self.OTFB_1.profile.bin_centers*1e6,
                         np.abs(self.OTFB_1.V_fine_tot), color=colors[i])
                ax.plot(self.OTFB_1.rf_centers*1e6,
                         np.abs(self.OTFB_1.V_coarse_tot), color=colors[i],
                         linestyle='', marker='.')
            self.OTFB_2.track_no_beam()
        if debug:
            plt.show()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            self.OTFB_1.profile.bin_centers, self.OTFB_1.rf_centers,
            self.OTFB_1.V_coarse_ind_gen + self.OTFB_2.V_coarse_ind_gen)

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
    G_ff : float
        FF gain [1]; default is 1
    G_llrf : float
        LLRF gain [1]; default is 10
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

    def __init__(self, RFStation, Beam, Profile, n_sections, n_cavities=2,
                 V_part=4/9, G_ff=1, G_llrf=10, G_tx=0.5, a_comb=15/16,
                 Commissioning=CavityFeedbackCommissioning()):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Commissioning options
        self.open_loop = Commissioning.open_loop
        if self.open_loop == 0:
            self.logger.debug("Opening overall OTFB loop")
        elif self.open_loop == 1:
            self.logger.debug("Closing overall OTFB loop")
        self.open_FB = Commissioning.open_FB
        if self.open_FB == 0:
            self.logger.debug("Opening feedback of drive correction")
        elif self.open_FB == 1:
            self.logger.debug("Closing feedback of drive correction")
        self.open_drive = Commissioning.open_drive
        if self.open_drive == 0:
            self.logger.debug("Opening drive to generator")
        elif self.open_drive == 1:
            self.logger.debug("Closing drive to generator")
        self.open_FF = Commissioning.open_FF
        if self.open_FF == 0:
            self.logger.debug("Opening feed-forward on beam current")
        elif self.open_FF == 1:
            self.logger.debug("Closing feed-forward on beam current")

        # Read input
        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile
        self.n_cavities = int(n_cavities)
        if self.n_cavities < 1:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_cavities has invalid value!")
        self.V_part = float(V_part)
        if self.V_part*(1 - self.V_part) < 0:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: V_part" +
                               " should be in range (0,1)!")

        # Gain settings
        self.G_ff = float(G_ff)
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx)

        # 200 MHz travelling wave cavity (TWC) model
        if n_sections in [3, 4, 5]:
            self.TWC = eval("SPS" + str(n_sections) + "Section200MHzTWC()")
            if self.open_FF == 1:
                # Feed-forward filter
                self.coeff_FF = getattr(sys.modules[__name__],
                    "feedforward_filter_TWC" + str(n_sections))
                self.n_FF = len(self.coeff_FF)
                self.n_FF_delay = int(0.5*(self.n_FF - 1) +
                                      0.5*self.TWC.tau/self.rf.t_rf[0, 0]/5)
                self.logger.debug("Feed-forward delay in samples %d",
                                  self.n_FF_delay)
                # Multiply gain by normalisation factors from filter and
                # beam-to generator current
                self.G_ff *= self.TWC.R_beam/(self.TWC.R_gen *
                                              np.sum(self.coeff_FF))
        else:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_sections has invalid value!")
        self.logger.debug("SPS OTFB cavities: %d, sections: %d, voltage" +
                          " partition %.2f, gain: %.2e", self.n_cavities,
                          n_sections, self.V_part, self.G_tx)

        # TWC resonant frequency
        self.omega_r = self.TWC.omega_r
        # Length of arrays in LLRF
        self.n_coarse = int(self.rf.t_rev[0]/self.rf.t_rf[0, 0])
        # Initialise turn-by-turn variables
        self.update_variables()

        # Initialise bunch-by-bunch voltage array with LENGTH OF PROFILE
        self.V_fine_tot = np.zeros(self.profile.n_slices, dtype=complex)
        # Array to hold the bucket-by-bucket voltage with LENGTH OF LLRF
        self.V_coarse_tot = np.zeros(self.n_coarse, dtype=complex)
        self.logger.debug("Length of arrays on coarse grid %d", self.n_coarse)

        # Initialise comb filter
        self.dV_comb_out_prev = np.zeros(self.n_coarse, dtype=complex)
        self.a_comb = float(a_comb)

        # Initialise cavity filter (moving average)
        self.n_mov_av = int(self.TWC.tau/self.rf.t_rf[0, 0])
        self.logger.debug("Moving average over %d points", self.n_mov_av)
        # Initialise moving average
        if self.n_mov_av < 2:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: profile has to" +
                               " have at least 12.5 ns resolution!")
        self.dV_ma_in_prev = np.zeros(self.n_coarse, dtype=complex)
        # Initialise generator-induced voltage
        self.I_gen_prev = np.zeros(self.n_mov_av, dtype=complex)
        self.logger.info("Class initialized")

        # Initialise feed-forward; sampled every 5 buckets
        if self.open_FF == 1:
            self.logger.debug("Feed-forward active")
            self.n_coarse_FF = int(self.n_coarse/5)
            self.I_beam_coarse_prev = np.zeros(self.n_coarse_FF, dtype=complex)
            self.I_ff_corr = np.zeros(self.n_coarse_FF, dtype=complex)
            self.V_ff_corr = np.zeros(self.n_coarse_FF, dtype=complex)

    def beam_induced_voltage(self, lpf=False):
        """Calculates the beam-induced voltage

        Parameters
        ----------
        lpf : bool
            Apply low-pass filter for beam current calculation;
            default is False

        Attributes
        ----------
        I_beam_coarse : complex array
            RF component of the beam charge [C] at the present time step,
            calculated in coarse grid
        I_beam_fine : complex array
            RF component of the beam charge [C] at the present time step,
            calculated in fine grid
        V_coarse_ind_beam : complex array
            Induced voltage [V] from beam-cavity interaction on the
            coarse grid
        V_fine_ind_beam : complex array
            Induced voltage [V] from beam-cavity interaction on the fine
            grid
        """

        # Beam current from profile
        self.I_beam_fine, self.I_beam_coarse = \
            rf_beam_current(self.profile,
                            self.omega_c, self.rf.t_rev[self.counter],
                            lpf=lpf,
                            downsample={'Ts': self.T_s,
                                        'points': self.n_coarse})

        # Beam-induced voltage
        self.induced_voltage('beam')
        self.induced_voltage('beam_coarse')

        if self.open_FF == 1:
            # Calculate correction based on previous turn on coarse grid
            for ind in range(self.n_coarse_FF):
                self.I_ff_corr[ind] = self.coeff_FF[0]* \
                                      self.I_beam_coarse_prev[ind]
                for k in range(self.n_FF):
                    self.I_ff_corr[ind] += self.coeff_FF[k] * \
                                           self.I_beam_coarse_prev[ind-k]
            self.V_ff_corr = self.G_ff* \
                self.matr_conv(self.I_ff_corr, self.TWC.h_gen[::5])

            # Compensate for FIR filter delay
            self.dV_ff = np.concatenate((self.V_ff_corr[self.n_FF_delay:],
                np.zeros(self.n_FF_delay, dtype=np.complex)))

            # Interpolate to finer grids
            self.V_ff_corr_coarse = np.interp(self.rf_centers,
                self.rf_centers[::5], self.dV_ff)
            self.V_ff_corr_fine = np.interp(self.profile.bin_centers,
                self.rf_centers[::5], self.dV_ff)

            # Add to beam-induced voltage (opposite sign)
            self.V_coarse_ind_beam += self.n_cavities*self.V_ff_corr_coarse
            self.V_fine_ind_beam += self.n_cavities*self.V_ff_corr_fine

            # Update vector from previous turn
            self.I_beam_coarse_prev = np.copy(self.I_beam_coarse[::5])

    def call_conv(self, signal, kernel):
        """Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1)
        bm.convolve(signal, kernel, result)

        return result

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
        self.V_gen = self.open_FB*modulator(self.dV_gen, self.omega_r,
            self.omega_c, self.rf.t_rf[0, self.counter]) \
            + self.open_drive*self.V_set

        # Generator charge from voltage, transmitter model
        self.I_gen = self.G_tx*self.V_gen/self.TWC.R_gen*self.T_s

        # Circular convolution: attach last points of previous turn
        self.I_gen = np.concatenate((self.I_gen_prev, self.I_gen))

        # Generator-induced voltage
        self.induced_voltage('gen')
        # Update memory of previous turn
        self.I_gen_prev = self.I_gen[-self.n_mov_av:]

    def induced_voltage(self, name):
        r"""Generation of beam- or generator-induced voltage from the
        beam or
        generator current, at a given carrier frequency and turn. The
        induced
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
        frequency is close to the cavity resonant frequency, :math:`h_c
        = 0`.

        :see also: :py:class:`llrf.impulse_response.TravellingWaveCavity`

        The impulse response is made to be the same length as the beam
        profile.

        """

        self.logger.debug("Matrix convolution for V_ind")

        if name == "beam":
            # Compute the beam-induced voltage on the fine grid
            self.__setattr__("V_fine_ind_"+name,
                self.matr_conv(self.__getattribute__("I_"+name+"_fine"),
                               self.TWC.__getattribute__("h_"+name)))
            self.V_fine_ind_beam *= -self.n_cavities

        if name == "beam_coarse" and hasattr(self.TWC, "h_beam_coarse"):
            # Compute the beam-induced voltage on the coarse grid
            self.__setattr__("V_coarse_ind_beam",
                self.matr_conv(self.__getattribute__("I_"+name),
                               self.TWC.__getattribute__("h_"+name)))
            self.V_coarse_ind_beam *= -self.n_cavities

        if name == "gen":
            # Compute the generator-induced voltage on the coarse grid
            self.__setattr__("V_coarse_ind_" + name,
                self.matr_conv(self.__getattribute__("I_"+name),
                               self.TWC.__getattribute__("h_"+name)))
            # Circular convolution
            self.V_coarse_ind_gen = +self.n_cavities \
                *self.V_coarse_ind_gen[self.n_mov_av:self.n_coarse+self.n_mov_av]

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
        self.dV_comb_out = comb_filter(self.dV_comb_out_prev, self.dV_gen,
                                       self.a_comb)

        # Shift signals with the delay time (to make exactly one turn)
        self.dV_gen = np.concatenate((self.dV_comb_out_prev[-self.n_delay:],
            self.dV_comb_out[:self.n_coarse-self.n_delay]))

        # For comb filter, update memory of previous turn
        self.dV_comb_out_prev = np.copy(self.dV_comb_out)

        # Modulate from omega_rf to omega_r
        self.dV_gen = modulator(self.dV_gen, self.omega_c, self.omega_r,
                                self.rf.t_rf[0, self.counter])


        # Cavity filter: CIRCULAR moving average over filling time
        # Memorize last points of previous turn for beginning of next turn
        self.dV_ma_in = np.copy(self.dV_gen)
        self.dV_gen = moving_average(self.dV_gen, self.n_mov_av,
            x_prev=self.dV_ma_in_prev[-self.n_mov_av+1:])
        self.dV_ma_in_prev = np.copy(self.dV_ma_in)

    def matr_conv(self, I, h):
        """Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        return scipy.signal.fftconvolve(I, h, mode='full')[:I.shape[0]]

    def track(self):
        """Turn-by-turn tracking method."""

        # Update turn-by-turn variables
        self.update_variables()

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_c, self.rf_centers)
        self.TWC.impulse_response_beam(self.omega_c, self.profile.bin_centers,
                                       self.rf_centers)

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
            + np.interp(self.profile.bin_centers, self.rf_centers,
                        self.V_coarse_ind_gen)

    def track_no_beam(self):
        """Initial tracking method, before injecting beam."""

        # Update turn-by-turn variables
        self.update_variables()

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

    def update_variables(self):
        '''Update counter and frequency-dependent variables in a given turn'''

        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0,self.counter]
        # Present sampling time
        self.T_s = self.rf.t_rf[0,self.counter]
        # Present coarse grid
        self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.T_s
        # Check number of samples required per turn
        n_coarse = int(self.rf.t_rev[self.counter]/self.T_s)
        if self.n_coarse != n_coarse:
            raise RuntimeError("Error in SPSOneTurnFeedback: changing number" +
                " of coarse samples. This option isnot yet implemented!")
        # Present delay time
        self.n_delay = int((self.rf.t_rev[self.counter] - self.TWC.tau)
                           / self.rf.t_rf[0, self.counter])


#    def pre_compute_semi_analytic_factor(self, time):
#        r""" Pre-computes factor for semi-analytic method, which is used to
#        compute the beam-induced voltage on the coarse grid.
#
#        Parameters
#        ----------
#        time : float array [s]
#            Time array at which to compute the beam-induced voltage
#
#        Attributes
#        ----------
#        profile_coarse : class
#            Beam profile with 20 bins per RF-bucket
#        semi_analytic_factor : complex array [:math:`\Omega\,s`]
#            Factor that is used to compute the beam-induced voltage
#        """
#
#        self.logger.info("Pre-computing semi-analytic factor")
#
#        n_slices_per_bucket = 20
#
#        n_buckets = int(np.round(
#            (self.profile.cut_right - self.profile.cut_left)
#            / self.rf.t_rf[0, 0]))
#
#        self.profile_coarse = Profile(self.beam, CutOptions=CutOptions(
#            cut_left=self.profile.cut_left,
#            cut_right=self.profile.cut_right,
#            n_slices=n_buckets*n_slices_per_bucket))
#
#        # pre-factor [Ohm s]
#
#        pre_factor = 2*self.TWC.R_beam / self.TWC.tau**2 / self.omega_r**3
#
#        # Matrix of time differences [1]
#        dt1 = np.zeros(shape=(len(time), self.profile_coarse.n_slices))
#
#        for i in range(len(time)):
#            dt1[i] = (time[i] - self.profile_coarse.bin_centers) * self.omega_r
#
##        dt2 = dt1 - self.TWC.tau * self.omega_r
#
##        phase1 = np.exp(-1j * dt1)
#        phase = np.exp(-1j * self.TWC.tau * self.TWC.omega_r)
#
##        diff1 = 2j - dt1 + self.TWC.tau * self.omega_r
#
##        diff2 = (2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)
#
#        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r
#               + (2j - dt1 + self.TWC.tau*self.omega_r) * np.exp(-1j * dt1))\
#            * np.sign(dt1) \
#            - ((2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)
#               + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
#            * np.sign(dt1 - self.TWC.tau * self.omega_r) \
#            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
##        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r + diff2) * np.sign(dt1) \
##            - (diff2 + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
##                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
##            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
##        tmp = (diff1.conjugate() + diff2) * np.sign(dt1) \
##            - (diff2 + diff1.conjugate() * phase) \
##                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
##            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
#        tmp *= pre_factor
#
#        self.semi_analytic_factor = np.diff(tmp)
#
#    def beam_induced_voltage_semi_analytic(self):
#        r"""Computes the beam-induced voltage in (I,Q) at the present carrier
#        frequency :math:`\omega_c` using the semi-analytic method. It requires
#        that pre_compute_semi_analytic_factor() was called previously.
#
#        Returns
#        -------
#        complex array [V]
#            Beam-induced voltage in (I,Q) at :math:`\omega_c`
#        """
#
#        # Update the coarse profile
#        self.profile_coarse.track()
#
#        # Slope of line segments [A/s]
#        kappa = self.beam.ratio*self.beam.Particle.charge*e \
#            * np.diff(self.profile_coarse.n_macroparticles) \
#            / self.profile_coarse.bin_size**2
#
#        return np.exp(1j*self.rf_centers*self.omega_c)\
#            * np.sum(self.semi_analytic_factor * kappa, axis=1)
#
#
