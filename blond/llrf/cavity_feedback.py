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
:Authors: **Birk Emil Karlsen-Baeck**, **Helga Timko**
'''


import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sys


from blond.llrf.signal_processing import comb_filter, cartesian_to_polar,\
    polar_to_cartesian, modulator, moving_average,\
    rf_beam_current, moving_average_improved
from blond.llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC, SPS5Section200MHzTWC
from blond.llrf.signal_processing import feedforward_filter_TWC3, \
    feedforward_filter_TWC4, feedforward_filter_TWC5
from blond.utils import bmath as bm


def get_power_gen_I2(I_gen_per_cav, Z_0):
    ''' RF generator power from generator current (physical, in [A]), for any f_r (and thus any tau) '''
    return 0.5 * Z_0 * np.abs(I_gen_per_cav)**2


class CavityFeedbackCommissioning(object):

    def __init__(self, debug=False, open_loop=False, open_FB=False,
                 open_drive=False, open_FF=False, V_SET=None,
                 cpp_conv=False, pwr_clamp=False, rot_IQ=1):
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
        V_SET : complex array
            Array set point voltage; default is False
        cpp_conv : bool
            Enable (True) or disable (False) convolutions using a C++ implementation; default is False
        pwr_clamp : bool
            Enable (True) or disable (False) power clamping; default is False
        rot_IQ : complex
            Option to rotate the set point and beam induced voltages in the complex plane.
        """

        self.debug = bool(debug)
        # Multiply with zeros if open == True
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_FB = int(np.invert(bool(open_FB)))
        self.open_drive = int(np.invert(bool(open_drive)))
        self.open_FF = int(np.invert(bool(open_FF)))
        self.V_SET = V_SET
        self.cpp_conv = cpp_conv
        self.pwr_clamp = pwr_clamp
        self.rot_IQ = rot_IQ


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
    G_ff : float or list
        FF gain [1]; if passed as a float, both 3- and 4-section (4- and
        5-section) cavities have the same G_ff in the post- (pre-)LS2
        scenario. If passed as a list, the first and second elements correspond
        to the G_ff of the 3- and 4-section (4- and 5-section) cavity
        feedback in the post- (pre-)LS2 scenario; default is 10
    G_llrf : float or list
        LLRF Gain [1]; convention same as G_ff; default is 10
    G_tx : float or list
        Transmitter gain [1] of the cavity feedback; convention same as G_ff;
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
    df : float or list
        Frequency difference between measured frequency and desired frequency;
        same convetion as G_ff; default is 0

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
                 a_comb=None, turns=1000, post_LS2=True, V_part=None, df=0,
                 Commissioning=CavityFeedbackCommissioning()):


        # Options for commissioning the feedback
        self.Commissioning = Commissioning
        self.rot_IQ = Commissioning.rot_IQ

        self.rf = RFStation

        # Parse input for gains
        if type(G_ff) is list:
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

        if type(df) is list:
            df_1 = df[0]
            df_2 = df[1]
        else:
            df_1 = df
            df_2 = df

        # Voltage partitioning has to be a fraction
        if V_part and V_part*(1 - V_part) < 0:
            raise RuntimeError("SPS cavity feedback: voltage partitioning has to be in the range (0,1)!")

        # Voltage partition proportional to the number of sections
        if post_LS2:
            if not a_comb:
                a_comb = 63/64

            if V_part is None:
                V_part = 6/10
            self.OTFB_1 = SPSOneTurnFeedback(RFStation, Beam, Profile, 3,
                                             n_cavities=4, V_part=V_part,
                                             G_ff=float(G_ff_1),
                                             G_llrf=float(G_llrf_1),
                                             G_tx=float(G_tx_1),
                                             a_comb=float(a_comb),
                                             df=float(df_1),
                                             Commissioning=self.Commissioning)
            self.OTFB_2 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                             n_cavities=2, V_part=1-V_part,
                                             G_ff=float(G_ff_2),
                                             G_llrf=float(G_llrf_2),
                                             G_tx=float(G_tx_2),
                                             a_comb=float(a_comb),
                                             df=float(df_2),
                                             Commissioning=self.Commissioning)
        else:
            if not a_comb:
                a_comb = 15/16

            if V_part is None:
                V_part = 4/9
            self.OTFB_1 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                             n_cavities=2, V_part=V_part,
                                             G_ff=float(G_ff_1),
                                             G_llrf=float(G_llrf_1),
                                             G_tx=float(G_tx_1),
                                             a_comb=float(a_comb),
                                             df=float(df_1),
                                             Commissioning=self.Commissioning)
            self.OTFB_2 = SPSOneTurnFeedback(RFStation, Beam, Profile, 5,
                                             n_cavities=2, V_part=1-V_part,
                                             G_ff=float(G_ff_2),
                                             G_llrf=float(G_llrf_2),
                                             G_tx=float(G_tx_2),
                                             a_comb=float(a_comb),
                                             df=float(df_2),
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

        self.V_sum = self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_slices:] \
                     + self.OTFB_2.V_ANT_FINE[-self.OTFB_2.profile.n_slices:]

        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr = (self.alpha_sum - np.angle(self.OTFB_1.V_SET[-self.OTFB_1.n_coarse]))

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
                         np.abs(self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_slices:]), color=colors[i])
                ax.plot(self.OTFB_1.rf_centers*1e6,
                         np.abs(self.OTFB_1.V_ANT[-self.OTFB_1.n_coarse:]), color=colors[i],
                         linestyle='', marker='.')
            self.OTFB_2.track_no_beam()
        if debug:
            plt.show()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            self.OTFB_1.profile.bin_centers, self.OTFB_1.rf_centers,
            self.OTFB_1.V_IND_COARSE_GEN[-self.OTFB_1.n_coarse:] + self.OTFB_2.V_IND_COARSE_GEN[-self.OTFB_2.n_coarse:])

        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr = (self.alpha_sum - np.angle(self.OTFB_1.V_SET[-self.OTFB_1.n_coarse]))



class SPSOneTurnFeedback(object):
    r"""Voltage feedback around a travelling wave cavity with a given amount of
    sections. The quantities of the LLRF system cover two turns with a coarse
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
        Number of cavities of the same type; default is 4
    V_part : float
        Voltage partition for the given n_cavities; in range (0,1); default is 4/9
    G_ff : float
        FF gain [1]; default is 1
    G_llrf : float
        LLRF gain [1]; default is 10
    G_tx : float
        Transmitter gain [A/V]; default is :math:`(50 \Omega)^{-1}`
    a_comb : float
        Coefficient for Comb-filter; default is 63/64
    df : float
        Frequency difference between measured frequency and desired frequency; default is 0
    Commissioning : class
        A CavityFeedbackCommissioning object

    Attributes
    ----------
    TWC : class
        A TravellingWaveCavity type class
    counter : int
        Counter of the current time step
    omega_c : float
        Carrier angular frequency [rad/s] at the current time step
    omega_r : const float
        Resonant angular frequency [rad/s] of the travelling wave cavities
    n_coarse : int
        Number of bins for the coarse gird (equal to the harmonic number)
    V_IND_COARSE_GEN : complex array
        Generator voltage [V] of the present turn in (I,Q) coordinates
    V_IND_FINE_BEAM : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the fine grid
        used for tracking the beam
    V_IND_COARSE_BEAM : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the coarse grid
        used internally for LLRF tracking
    V_ANT_FINE : complex array
        Antenna voltage [V] at present and last turn in (I,Q) coordinates
        which is used for tracking the beam
    V_ANT : complex array
        Antenna voltage [V] at present and last turn in (I,Q) coordinates
        which is used internally for LLRF tracking
    logger : logger
        Logger of the present class

    Note: All currents are in units of charge because the sampling time drops out during the convolution calculation
    """

    def __init__(self, RFStation, Beam, Profile, n_sections, n_cavities=4,
                 V_part=4/9, G_ff=1, G_llrf=10, G_tx=0.5, a_comb=63/64, df=0,
                 Commissioning=CavityFeedbackCommissioning()):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Commissioning options
        self.open_loop = Commissioning.open_loop
        if self.open_loop == 0:                                 # Open Loop
            self.logger.debug("Opening overall OTFB loop")
        elif self.open_loop == 1:
            self.logger.debug("Closing overall OTFB loop")
        self.open_FB = Commissioning.open_FB
        if self.open_FB == 0:                                   # Open Feedback
            self.logger.debug("Opening feedback of drive correction")
        elif self.open_FB == 1:
            self.logger.debug("Closing feedback of drive correction")
        self.open_drive = Commissioning.open_drive
        if self.open_drive == 0:                                # Open Drive
            self.logger.debug("Opening drive to generator")
        elif self.open_drive == 1:
            self.logger.debug("Closing drive to generator")
        self.open_FF = Commissioning.open_FF
        if self.open_FF == 0:                                   # Open Feedforward
            self.logger.debug("Opening feed-forward on beam current")
        elif self.open_FF == 1:
            self.logger.debug("Closing feed-forward on beam current")
        self.V_SET = Commissioning.V_SET
        if self.V_SET is None:                                  # Vset as array or not
            self.set_point_modulation = False
        else:
            self.set_point_modulation = True

        self.cpp_conv = Commissioning.cpp_conv
        self.rot_IQ = Commissioning.rot_IQ

        # Read input
        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile
        self.n_cavities = int(n_cavities)
        self.n_sections = int(n_sections)
        if self.n_cavities < 1:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_cavities has invalid value!")
        self.V_part = float(V_part)
        if self.V_part * (1 - self.V_part) < 0:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: V_part" +
                               " should be in range (0,1)!")

        # Gain settings
        self.G_ff = float(G_ff)
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx) / (self.n_cavities)

        # 200 Hz travelling wave cavity (TWC) model
        if n_sections in [3, 4, 5]:
            self.TWC = eval("SPS" + str(n_sections) + "Section200MHzTWC(" + str(df) + ")")
            if self.open_FF == 1:
                # Feed-forward fitler
                self.coeff_FF = getattr(sys.modules[__name__],
                                "feedforward_filter_TWC" + str(n_sections))
                self.n_FF = len(self.coeff_FF)          # Number of coefficients for FF
                self.n_FF_delay = int(0.5 * (self.n_FF - 1) +
                                      0.5 * self.TWC.tau/self.rf.t_rf[0,0]/5)
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

        if self.cpp_conv:
            self.conv = getattr(self, 'call_conv')
        else:
            self.conv = getattr(self, 'matr_conv')

        # TWC resonant frequency
        self.omega_r = self.TWC.omega_r
        # Length of arrays in LLRF
        self.n_coarse = int(round(self.rf.t_rev[0]/self.rf.t_rf[0, 0]))
        # Initialize turn-by-turn variables
        self.dphi_mod = 0
        self.update_variables()

        # Check array length for set point modulation
        if self.set_point_modulation:
            if self.V_SET.shape[0] != 2 * self.n_coarse:
                raise RuntimeError("V_SET length should be %d" %(2*self.n_coarse))
            self.set_point = getattr(self, "set_point_mod")
        else:
            self.set_point = getattr(self, "set_point_std")
            self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize bunch-by-bunch voltage array with lenght of profile
        self.V_ANT_FINE = np.zeros(2 * self.profile.n_slices, dtype=complex)
        # Array to hold the bucket-by-bucket voltage with length LLRF
        self.V_ANT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.DV_GEN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.logger.debug("Length of arrays on coarse grid 2x %d", self.n_coarse)

        # LLRF MODEL ARRAYS
        # Initialize comb filter
        self.DV_COMB_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.a_comb = float(a_comb)

        # Initialize the delayed signal
        self.DV_DELAYED = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize modulated signal (to fr)
        self.DV_MOD_FR = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize moving average
        self.n_mov_av = int(self.TWC.tau/self.rf.t_rf[0, 0])
        self.DV_MOV_AVG = np.zeros(2 * self.n_coarse, dtype=complex)
        self.logger.debug("Moving average over %d points", self.n_mov_av)
        if self.n_mov_av < 2:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: profile has to" +
                               " have at least 12.5 ns resolution!")

        # GENERATOR MODEL ARRAYS
        # Initialize modulated signal (to frf)
        self.DV_MOD_FRF = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize generator current
        self.I_GEN = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize induced voltage on coarse grid
        self.V_IND_COARSE_GEN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.CONV_RES = np.zeros(2 * self.n_coarse, dtype=complex)
        self.CONV_PREV = np.zeros(self.n_coarse, dtype=complex)

        # BEAM MODEL ARRAYS
        # Initialize beam current coarse and fine
        self.I_FINE_BEAM = np.zeros(2 * self.profile.n_slices, dtype=complex)
        self.I_COARSE_BEAM = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize induced beam voltage coarse and fine
        self.V_IND_FINE_BEAM = np.zeros(2 * self.profile.n_slices, dtype=complex)
        self.V_IND_COARSE_BEAM = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialise feed-forward; sampled every fifth bucket
        if self.open_FF == 1:
            self.logger.debug('Feed-forward active')
            self.n_coarse_FF = int(self.n_coarse/5)
            self.I_BEAM_COARSE_FF = np.zeros(2 * self.n_coarse_FF, dtype=complex)
            self.I_BEAM_COARSE_FF_MOD = np.zeros(2 * self.n_coarse_FF, dtype=complex)
            self.I_FF_CORR_MOD = np.zeros(2 * self.n_coarse_FF, dtype=complex)
            self.I_FF_CORR = np.zeros(2 * self.n_coarse_FF, dtype=complex)
            self.V_FF_CORR = np.zeros(2 * self.n_coarse_FF, dtype=complex)
            self.DV_FF = np.zeros(2 * self.n_coarse_FF, dtype=complex)

        self.logger.info("Class initialized")


    def track(self):

        # Update turn-by-turn variables
        self.update_variables()

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_c, self.rf_centers)
        self.TWC.impulse_response_beam(self.omega_c, self.profile.bin_centers,
                                       self.rf_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.gen_model()

        # Beam-induced voltage from beam profile
        self.beam_model(lpf=False)

        # Sum generator- and beam-induced voltages for coarse grid
        self.V_ANT_START = np.copy(self.V_ANT)
        self.V_ANT[:self.n_coarse] = self.V_ANT[-self.n_coarse:]
        self.V_ANT[-self.n_coarse:] = self.V_IND_COARSE_GEN[-self.n_coarse:] \
                                      + self.V_IND_COARSE_BEAM[-self.n_coarse:]

        # Obtain generator-induced voltage on the fine grid by interpolation
        self.V_ANT_FINE_START = np.copy(self.V_ANT_FINE)
        self.V_ANT_FINE[:self.profile.n_slices] = self.V_ANT_FINE[-self.profile.n_slices:]
        self.V_ANT_FINE[-self.profile.n_slices:] = self.V_IND_FINE_BEAM[-self.profile.n_slices:] \
                                                   + np.interp(self.profile.bin_centers, self.rf_centers,
                                                               self.V_IND_COARSE_GEN[-self.n_coarse:])

        # Feed-forward corrections
        if self.open_FF == 1:
            self.V_ANT[-self.n_coarse:] = self.V_FF_CORR_COARSE + self.V_ANT[-self.n_coarse:]
            self.V_ANT_FINE[-self.profile.n_slices:] = self.V_FF_CORR_FINE + self.V_ANT_FINE[-self.profile.n_slices:]

    def track_no_beam(self):

        # Update variables
        self.update_variables()

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_c, self.rf_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Apply generator model
        self.gen_model()

        self.logger.debug("Total voltage to generator %.3e V",
                          np.mean(np.absolute(self.DV_MOD_FRF)))
        self.logger.debug("Total current from generator %.3e A",
                          np.mean(np.absolute(self.I_GEN))
                          / self.profile.bin_size)

        # Without beam, the total voltage is equal to the induced generator voltage
        self.V_ANT_START = np.copy(self.V_ANT)
        self.V_ANT[:self.n_coarse] = self.V_ANT[-self.n_coarse:]
        self.V_ANT[-self.n_coarse:] = self.V_IND_COARSE_GEN[-self.n_coarse:]

        self.logger.debug(
            "Average generator voltage, last half of array %.3e V",
            np.mean(np.absolute(self.V_IND_COARSE_GEN[int(0.5 * self.n_coarse):])))

    def llrf_model(self):

        self.set_point()
        self.error_and_gain()
        self.comb()
        self.one_turn_delay()
        self.mod_to_fr()
        self.mov_avg()


    def gen_model(self):

        self.mod_to_frf()
        self.sum_and_gain()
        self.gen_response()

    def beam_model(self, lpf=False):

        # Beam current from profile
        self.I_COARSE_BEAM[:self.n_coarse] = self.I_COARSE_BEAM[-self.n_coarse:]
        self.I_FINE_BEAM[:self.profile.n_slices] = self.I_FINE_BEAM[-self.profile.n_slices:]
        self.I_FINE_BEAM[-self.profile.n_slices:], self.I_COARSE_BEAM[-self.n_coarse:] = \
                rf_beam_current(self.profile, self.omega_c, self.rf.t_rev[self.counter],
                                lpf=lpf, downsample={'Ts': self.T_s, 'points': self.n_coarse},
                                external_reference=True)

        self.I_FINE_BEAM[-self.profile.n_slices:] = -self.rot_IQ * self.I_FINE_BEAM[-self.profile.n_slices:] / \
                                                    self.profile.bin_size
        self.I_COARSE_BEAM[-self.n_coarse:] = -self.rot_IQ * self.I_COARSE_BEAM[-self.n_coarse:] / self.T_s

        # Beam-induced voltage
        self.beam_response(coarse=False)
        self.beam_response(coarse=True)

        # Feed-forward
        if self.open_FF == 1:
            # Calculate correction based on previous turn on coarse grid
            # TODO: do a test where central frequency is at the RF frequency

            # Resample RF beam current to FF sampling frequency
            self.I_BEAM_COARSE_FF[:self.n_coarse_FF] = self.I_BEAM_COARSE_FF[-self.n_coarse_FF:]
            I_COARSE_BEAM_RESHAPED = np.copy(self.I_COARSE_BEAM[-self.n_coarse:])
            I_COARSE_BEAM_RESHAPED = I_COARSE_BEAM_RESHAPED.reshape((self.n_coarse//self.n_coarse_FF, self.n_coarse_FF))
            self.I_BEAM_COARSE_FF[-self.n_coarse_FF:] = np.sum(I_COARSE_BEAM_RESHAPED, axis=0) / 5

            # Do a down-modulation to the resonant frequency of the TWC
            self.I_BEAM_COARSE_FF_MOD[:self.n_coarse_FF] = self.I_BEAM_COARSE_FF_MOD[-self.n_coarse_FF:]
            self.I_BEAM_COARSE_FF_MOD[-self.n_coarse_FF:] = modulator(self.I_BEAM_COARSE_FF[-self.n_coarse_FF:],
                                                                  omega_i=self.omega_c, omega_f=self.omega_r,
                                                                  T_sampling= 5 * self.T_s,
                                                                  phi_0=(self.dphi_mod + self.rf.dphi_rf[0]))

            self.I_FF_CORR[:self.n_coarse_FF] = self.I_FF_CORR[-self.n_coarse_FF:]
            for ind in range(self.n_coarse_FF, 2 * self.n_coarse_FF):
                for k in range(self.n_FF):
                    self.I_FF_CORR[ind] += self.coeff_FF[k] \
                                      * self.I_BEAM_COARSE_FF_MOD[ind-k]

            # Do a down-modulation to the resonant frequency of the TWC
            self.I_FF_CORR_MOD[:self.n_coarse_FF] = self.I_FF_CORR_MOD[-self.n_coarse_FF:]
            self.I_FF_CORR_MOD[-self.n_coarse_FF:] = modulator(self.I_FF_CORR[-self.n_coarse_FF:],
                                                           omega_i=self.omega_r, omega_f=self.omega_c,
                                                           T_sampling=5 * self.T_s,
                                                           phi_0=-(self.dphi_mod + self.rf.dphi_rf[0]))

            # Find voltage from convolution with generator response
            self.V_FF_CORR[:self.n_coarse_FF] = self.V_FF_CORR[-self.n_coarse_FF:]
            self.V_FF_CORR[-self.n_coarse_FF:] = self.G_ff \
                            * self.matr_conv(self.I_FF_CORR_MOD, self.TWC.h_gen[::5])[-self.n_coarse_FF:] * 5 * self.T_s

            # Compensate for FIR filter delay
            self.DV_FF[:self.n_coarse_FF] = self.DV_FF[-self.n_coarse_FF:]
            self.DV_FF[-self.n_coarse_FF:] = self.V_FF_CORR[self.n_coarse_FF - self.n_FF_delay: - self.n_FF_delay]

            # Interpolate to finer grids
            self.V_FF_CORR_COARSE = np.interp(self.rf_centers, self.rf_centers[::5], self.DV_FF[-self.n_coarse_FF:])
            self.V_FF_CORR_FINE = np.interp(self.profile.bin_centers, self.rf_centers[::5], self.DV_FF[-self.n_coarse_FF:])


    # INDIVIDUAL COMPONENTS ---------------------------------------------------

    # LLRF MODEL
    def set_point_std(self):

        self.logger.debug("Entering %s function" %sys._getframe(0).f_code.co_name)
        # Read RF voltage from rf object
        self.V_set = polar_to_cartesian(
            self.V_part * self.rf.voltage[0, self.counter],
            0.5 * np.pi - self.rf.phi_rf[0, self.counter] + np.angle(self.rot_IQ))

        # Convert to array
        self.V_SET[:self.n_coarse] = self.V_SET[-self.n_coarse:]
        self.V_SET[-self.n_coarse:] = self.V_set * np.ones(self.n_coarse) # * self.rot_IQ


    def set_point_mod(self):

        self.logger.debug("Entering %s function" %sys._getframe(0).f_code.co_name)
        pass


    def error_and_gain(self):

        self.DV_GEN[:self.n_coarse] = self.DV_GEN[-self.n_coarse:]
        self.DV_GEN[-self.n_coarse:] = self.G_llrf * (self.V_SET[-self.n_coarse:] -
                                                      self.open_loop * self.V_ANT[-self.n_coarse:])
        self.logger.debug("In %s, average set point voltage %.6f MV",
                          sys._getframe(0).f_code.co_name,
                          1e-6 * np.mean(np.absolute(self.V_SET)))
        self.logger.debug("In %s, average antenna voltage %.6f MV",
                          sys._getframe(0).f_code.co_name,
                          1e-6 * np.mean(np.absolute(self.V_ANT)))
        self.logger.debug("In %s, average voltage error %.6f MV",
                          sys._getframe(0).f_code.co_name,
                          1e-6 * np.mean(np.absolute(self.DV_GEN)))


    def comb(self):

        # Shuffle present data to previous data
        self.DV_COMB_OUT[:self.n_coarse] = self.DV_COMB_OUT[-self.n_coarse:]
        # Update present data
        self.DV_COMB_OUT[-self.n_coarse:] = comb_filter(self.DV_COMB_OUT[:self.n_coarse],
                                                        self.DV_GEN[-self.n_coarse:],
                                                        self.a_comb)


    def one_turn_delay(self):

        self.DV_DELAYED[:self.n_coarse] = self.DV_DELAYED[-self.n_coarse:]
        self.DV_DELAYED[-self.n_coarse:] = self.DV_COMB_OUT[self.n_coarse-self.n_delay:-self.n_delay]


    def mod_to_fr(self):
        self.DV_MOD_FR[:self.n_coarse] = self.DV_MOD_FR[-self.n_coarse:]
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        self.DV_MOD_FR[-self.n_coarse:] = modulator(self.DV_DELAYED[-self.n_coarse:],
                                                    self.omega_c, self.omega_r,
                                                    self.rf.t_rf[0, self.counter],
                                                    phi_0= (self.dphi_mod + self.rf.dphi_rf[0]))


    def mov_avg(self):
        self.DV_MOV_AVG[:self.n_coarse] = self.DV_MOV_AVG[-self.n_coarse:]
        self.DV_MOV_AVG[-self.n_coarse:] = moving_average(self.DV_MOD_FR[-self.n_mov_av - self.n_coarse + 1:], self.n_mov_av)


    # GENERATOR MODEL
    def mod_to_frf(self):

        self.DV_MOD_FRF[:self.n_coarse] = self.DV_MOD_FRF[-self.n_coarse:]
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        self.DV_MOD_FRF[-self.n_coarse:] = self.open_FB * modulator(self.DV_MOV_AVG[-self.n_coarse:],
                                                                    self.omega_r, self.omega_c,
                                                                    self.rf.t_rf[0, self.counter],
                                                                    phi_0=-(self.dphi_mod + self.rf.dphi_rf[0]))


    def sum_and_gain(self):

        self.I_GEN[:self.n_coarse] = self.I_GEN[-self.n_coarse:]
        self.I_GEN[-self.n_coarse:] = self.DV_MOD_FRF[-self.n_coarse:] + self.open_drive * self.V_SET[-self.n_coarse:]
        self.I_GEN[-self.n_coarse:] *= self.G_tx / self.TWC.R_gen


    def gen_response(self):

        self.V_IND_COARSE_GEN[:self.n_coarse] = self.V_IND_COARSE_GEN[-self.n_coarse:]
        self.V_IND_COARSE_GEN[-self.n_coarse:] = self.n_cavities * self.matr_conv(self.I_GEN,
                                                 self.TWC.h_gen)[-self.n_coarse:] * self.T_s


    # BEAM MODEL
    def beam_response(self, coarse=False):
        self.logger.debug('Matrix convolution for V_ind')

        if coarse:
            self.V_IND_COARSE_BEAM[:self.n_coarse] = self.V_IND_COARSE_BEAM[-self.n_coarse:]
            self.V_IND_COARSE_BEAM[-self.n_coarse:] = self.n_cavities * self.matr_conv(self.I_COARSE_BEAM,
                                                        self.TWC.h_beam_coarse)[-self.n_coarse:] * self.T_s
        else:
            self.V_IND_FINE_BEAM[:self.profile.n_slices] = self.V_IND_FINE_BEAM[-self.profile.n_slices:]
            # Only convolve the slices for the current turn because the fine grid points can be less
            # than one turn in length
            self.V_IND_FINE_BEAM[-self.profile.n_slices:] = self.n_cavities \
                                                            * self.matr_conv(self.I_FINE_BEAM[-self.profile.n_slices:],
                                                            self.TWC.h_beam)[-self.profile.n_slices:] * self.profile.bin_size


    def matr_conv(self, I, h):
        """Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        return scipy.signal.fftconvolve(I, h, mode='full')[:I.shape[0]]


    def call_conv(self, signal, kernel):
        """Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1)
        bm.convolve(signal, kernel, result=result, mode='full')

        return result


    def update_variables(self):

        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0, self.counter]
        # Present sampling time
        self.T_s = self.rf.t_rf[0, self.counter]
        # Phase offset at the end of a 1-turn modulated signal (for demodulated, multiply by -1 as c and r reversed)
        self.phi_mod_0 = (self.omega_c - self.omega_r) * self.T_s * (self.n_coarse) % (2 * np.pi)
        self.dphi_mod += self.phi_mod_0
        # Present coarse grid
        self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.T_s
        # Check number of samples required per turn
        n_coarse = int(round(self.rf.t_rev[self.counter]/self.T_s))
        if self.n_coarse != n_coarse:
            raise RuntimeError("Error in SPSOneTurnFeedback: changing number" +
                " of coarse samples. This option isnot yet implemented!")
        # Present delay time
        self.n_mov_av = int(self.TWC.tau / self.rf.t_rf[0, self.counter])
        self.n_delay = self.n_coarse - self.n_mov_av

    # Power related functions
    def calc_power(self):
        self.II_COARSE_GEN = np.copy(self.I_GEN)
        self.P_GEN = get_power_gen_I2(self.II_COARSE_GEN, 50)

    def wo_clamping(self):
        pass

    def w_clamping(self):
        pass



