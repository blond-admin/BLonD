'''
Author: Helga Timko, Birk Emil Karlsen-Bæck

Rewriting the cavity feedback objects for the SPS machine for easier debugging.
'''

# Imports ---------------------------------------------------------------------
import logging
import numpy as np
import sys
import scipy.signal
import matplotlib.pyplot as plt

from blond.llrf.signal_processing import comb_filter, cartesian_to_polar,\
    polar_to_cartesian, modulator, moving_average, moving_average_improved,\
    rf_beam_current
from blond.llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC, SPS5Section200MHzTWC
from blond.llrf.signal_processing import feedforward_filter_TWC3, \
    feedforward_filter_TWC4, feedforward_filter_TWC5
from blond.utils import bmath as bm


# Classes ---------------------------------------------------------------------
class CavityFeedbackCommissioning_new(object):

    def __init__(self, debug=False, open_loop=False, open_FB=False,
                 open_drive=False, open_FF=False, V_SET=None,
                 cpp_conv = False):
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
        """

        self.debug = bool(debug)
        # Multiply with zeros if open == True
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_FB = int(np.invert(bool(open_FB)))
        self.open_drive = int(np.invert(bool(open_drive)))
        self.open_FF = int(np.invert(bool(open_FF)))
        self.V_SET = V_SET
        self.cpp_conv = cpp_conv


class SPSOneTurnFeedback_new(object):

    def __init__(self, RFStation, Beam, Profile, n_sections, n_cavities=2,
                 V_part=4/9, G_ff=1, G_llrf=10, G_tx=0.5, a_comb=63/64,
                 Commissioning=CavityFeedbackCommissioning_new()):

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

        # Read input
        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile
        self.n_cavities = int(n_cavities)
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
        self.G_tx = float(G_tx)

        # 200 Hz travelling wave cavity (TWC) model
        if n_sections in [3, 4, 5]:
            self.TWC = eval("SPS" + str(n_sections) + "Section200MHzTWC()")
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
        self.n_coarse = int(self.rf.t_rev[0]/self.rf.t_rf[0, 0])
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

        # TODO: Initialize bunch-by-bunch voltage array with lenght of profile

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

        self.logger.info("Class initialized")


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


    def set_point_std(self):

        self.logger.debug("Entering %s function" %sys._getframe(0).f_code.co_name)
        # Read RF voltage from rf object
        self.V_set = polar_to_cartesian(
            self.V_part * self.rf.voltage[0, self.counter],
            0.5 * np.pi - self.rf.phi_rf[0, self.counter])

        # Convert to array
        self.V_SET[:self.n_coarse] = self.V_SET[-self.n_coarse]
        self.V_SET[-self.n_coarse:] = self.V_set * np.ones(self.n_coarse)


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
                                                    phi_0= self.dphi_mod + self.rf.dphi_rf[0])

    def mov_avg(self):
        #self.n_mov_av = 50
        self.DV_MOV_AVG[:self.n_coarse] = self.DV_MOV_AVG[-self.n_coarse:]
        self.DV_MOV_AVG[-self.n_coarse:] = moving_average(self.DV_MOD_FR[-self.n_mov_av - self.n_coarse + 1:], self.n_mov_av)#,
                                                #x_prev=self.DV_MOD_FR[self.n_coarse-self.n_mov_av + 1:self.n_coarse])


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
        self.I_GEN[-self.n_coarse:] *= self.G_tx * self.T_s / self.TWC.R_gen


    def gen_response(self):

        self.V_IND_COARSE_GEN[:self.n_coarse] = self.V_IND_COARSE_GEN[-self.n_coarse:]
        self.V_IND_COARSE_GEN[-self.n_coarse:] = self.n_cavities * self.matr_conv(self.I_GEN,        # TODO: originally self.n_mov_av + self.n_coarse + 1
                                                                    self.TWC.h_gen)[-self.n_coarse:]
        # TODO: This
        #self.V_IND_COARSE_GEN[-self.n_coarse:] = self.n_cavities * self.conv(self.I_GEN[-self.n_coarse:],
        #                                                           self.TWC.h_gen)[-self.n_coarse:]

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
        self.phi_mod_0 = (self.omega_c - self.omega_r) * self.T_s * (self.n_coarse) % (2 * np.pi) # TODO: self.n_coarse - 1
        self.dphi_mod += self.phi_mod_0
        # Present coarse grid
        self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.T_s
        # Check number of samples required per turn
        n_coarse = int(self.rf.t_rev[self.counter]/self.T_s)
        if self.n_coarse != n_coarse:
            raise RuntimeError("Error in SPSOneTurnFeedback: changing number" +
                " of coarse samples. This option isnot yet implemented!")
        # Present delay time
        self.n_mov_av = int(self.TWC.tau / self.rf.t_rf[0, self.counter])
        self.n_delay = self.n_coarse - self.n_mov_av







