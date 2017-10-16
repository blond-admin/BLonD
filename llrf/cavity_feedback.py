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
from llrf.signal_processing import comb_filter, cartesian_to_polar, \
    polar_to_cartesian, modulator, moving_average, rf_beam_current
from llrf.impulse_response import SPS4Section200MHzTWC, SPS5Section200MHzTWC
from setup_cpp import libblond


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
    G_tx_4 : float
        Transmitter gain [1] of the 4-section cavity feedback; default is 10
    G_tx_5 : float
        Transmitter gain [1] of the 5-section cavity feedback; default is 10
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

        # Voltage partition proportional to the number of sections
        self.OTFB_4 = SPSOneTurnFeedback(RFStation, Beam, Profile, 4,
                                         n_cavities=2, V_part=4/9, G_llrf=float(G_llrf), G_tx=float(G_tx),
                                         a_comb=float(a_comb),
                                         Commissioning=self.Commissioning)
        self.OTFB_5 = SPSOneTurnFeedback(RFStation, Beam, Profile, 5,
                                         n_cavities=2, V_part=5/9, G_llrf=float(G_llrf), G_tx=float(G_tx),
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
        self.V_sum = self.OTFB_4.V_tot + self.OTFB_5.V_tot

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr, self.phi_corr = cartesian_to_polar(self.V_sum)
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr -= self.rf.phi_rf[0, self.rf.counter[0]]

    def track_init(self, debug=False):

        if debug == True:
            import matplotlib.pyplot as plt
            f, ax1 = plt.subplots()
            ax2 = plt.twinx(ax1)
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Total voltage, real part [V]")
            ax2.set_ylabel("Total voltage, imaginary part (dotted) [V]")
            ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.figure(2)
            ax = plt.axes()
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Total voltage, amplitude [V]")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.figure(3)
            ax3 = plt.axes()

            from matplotlib import gridspec
            # Colors
            jet = plt.get_cmap('jet')
            colors = jet(np.linspace(0, 1, self.turns))

            # Plot 1: cavity voltage
            fig1 = plt.figure(4, figsize=(8, 10))
            gs1 = gridspec.GridSpec(2, 1)
            ax1_1 = plt.subplot(gs1[0])
            ax1_2 = plt.subplot(gs1[1], sharex=ax1_1)
            plt.setp(ax1_1.get_xticklabels(), visible=False)
            # remove last tick label for the second subplot
            yticks = ax1_1.yaxis.get_major_ticks()
            yticks[0].set_visible(False)
            plt.subplots_adjust(hspace=.0)
            ax1_1.set_ylabel(r"$Re(V_{\mathsf{cav}})$ [MV]")
            ax1_2.set_xlabel(r"Time [$\mu$s]")
            ax1_2.set_ylabel(r"$Im(V_{\mathsf{cav}})$ [MV]")
            ax1_1.set_ylim((-1, 5))
            ax1_2.set_ylim((0, 7))

        for i in range(self.turns):
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_4.track_no_beam()
            self.OTFB_5.track_no_beam()
            self.V_sum = self.OTFB_4.V_tot + self.OTFB_5.V_tot

            if debug == True:
                ax1.plot(self.OTFB_4.profile.bin_centers, self.V_sum.real)
                ax2.plot(self.OTFB_4.profile.bin_centers, self.V_sum.imag, ':')
#                ax.plot(self.OTFB_4.profile.bin_centers,
#                        np.absolute(self.V_sum))
                ax.plot(self.OTFB_4.profile.bin_centers,
                        np.absolute(self.OTFB_4.V_tot))
                ax3.plot(np.absolute(self.OTFB_4.I_gen))
#                 time = np.linspace(0., self.OTFB_4.n_llrf*self.OTFB_4.profile.bin_size, self.OTFB_4.n_llrf, endpoint=True)
#                 ax1.plot(time, self.V_sum.real)
#                 ax2.plot(time, self.V_sum.imag, ':')
#                 ax.plot(time, np.absolute(self.V_sum))
#                ax.plot(np.absolute(self.OTFB_4.I_gen))
#                ax.plot(self.OTFB_4.profile.bin_centers,
#                        np.absolute(self.OTFB_4.V_gen))
                ax1_1.plot(1e6*self.OTFB_4.profile.bin_centers,
                           1e-6*(self.OTFB_4.V_tot.real +
                                 self.OTFB_5.V_tot.real),
                           color=colors[i])
                ax1_1.fill_between(1e6*self.OTFB_4.profile.bin_centers, 0,
                                   1e-6*(self.OTFB_4.V_tot.real +
                                         self.OTFB_5.V_tot.real),
                                   alpha=0.2, color=colors[i])
                ax1_2.plot(1e6*self.OTFB_4.profile.bin_centers,
                           1e-6*(self.OTFB_4.V_tot.imag +
                                 self.OTFB_5.V_tot.imag),
                           color=colors[i])
                ax1_2.fill_between(1e6*self.OTFB_4.profile.bin_centers, 0,
                                   1e-6*(self.OTFB_4.V_tot.imag +
                                         self.OTFB_5.V_tot.imag),
                                   alpha=0.2, color=colors[i])
                fig1.savefig("fig/V_ant_" + "%d" % (i+1) + ".png")

        # if debug == True:
            # fig.savefig("OTFB.png")
            # plt.show()


class SPSOneTurnFeedback(object):
    r'''Voltage feedback around a travelling wave cavity with given amount of
    sections.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        A Profile type class
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
    V_gen : complex array
        Generator voltage [V] of the present turn in (I,Q) coordinates
    V_gen_prev : complex array
        Generator voltage [V] of the previous turn in (I,Q) coordinates
    V_ind_beam : complex array
        Beam-induced voltage [V] in (I,Q) coordinates
    V_ind_gen : complex array
        Generator-induced voltage [V] in (I,Q) coordinates
    V_tot : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates;
        :math:`V_{\mathsf{tot}}`
    a_comb : float
        Recursion constant of the comb filter; :math:`a_{\mathsf{comb}}=15/16`
    bw_cav : const float
        Cavity bandwidth; :math:`f_{\mathsf{bw,cav}} = 40 MHz`
    n_mov_av : const int
        Number of points for moving average modelling cavity response;
        :math:`n_{\mathsf{mov.av.}} = \frac{f_r}{f_{\mathsf{bw,cav}}}`, where
        :math:`f_r` is the cavity resonant frequency of TWC_4 and TWC_5
    logger : logger
        Logger of the present class

    '''

    def __init__(self, RFStation, Beam, Profile, n_sections, n_cavities=2,
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
        self.V_tot = np.zeros(self.profile.n_slices, dtype=complex)

        # Length of arrays in LLRF and V_ind_gen
        self.n_llrf = int(self.rf.t_rev[0]/self.profile.bin_size)
        # TODO: Bin size can change! Update affected variables!!
        self.logger.debug("Length of arrays in generator path %d", self.n_llrf)
        self.n_diff = self.n_llrf - self.profile.n_slices

        # Initialise comb filter
        self.V_gen_prev = np.zeros(self.n_llrf, dtype=complex)
        self.a_comb = float(a_comb)

        # Initialise cavity filter (moving average)
        self.n_mov_av = int(self.TWC.tau/self.profile.bin_size)
        self.logger.debug("Moving average over %d points", self.n_mov_av)
        if self.n_mov_av < 2:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: profile has to" +
                               " have at least 12.5 ns resolution!")
        self.V_mov_av_prev = np.zeros(self.n_llrf, dtype=complex)
        self.V_mov_av_next = np.zeros(self.n_llrf, dtype=complex)

        # Initialise generator-induced voltage
        self.I_gen_prev = np.zeros(self.n_mov_av, dtype=complex)

        self.logger.info("Class initialized")

    def track(self):
        """Turn-by-turn tracking method."""

        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0, self.counter]
        # Present delay time
        self.n_delay = int((self.rf.t_rev[0] - self.TWC.tau)
                           / self.profile.bin_size)

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response(self.omega_c, self.profile.bin_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.generator_induced_voltage()

        # Beam-induced voltage from beam profile
        self.beam_induced_voltage()

        # Sum and convert to voltage amplitude and phase
        self.V_tot = self.V_ind_beam + self.V_ind_gen[:self.profile.n_slices]

    def track_no_beam(self):
        """Initial tracking method, before injecting beam."""

        # Present time step
        self.counter = int(0)
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0, 0]
        # Present delay time
        self.n_delay = int((self.rf.t_rev[0] - self.TWC.tau)
                           / self.profile.bin_size)

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response(self.omega_c, self.profile.bin_centers)

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.logger.debug("Total voltage to generator %.3e V",
                          np.mean(np.absolute(self.V_gen)))
        self.generator_induced_voltage()
        self.logger.debug("Total current from generator %.3e A",
                          np.mean(np.absolute(self.I_gen))/self.profile.bin_size)

        # Sum and convert to voltage amplitude and phase
        self.V_tot = self.V_ind_gen[:self.profile.n_slices]
        self.logger.debug("Average total voltage, last half of array %.3e V",
                          np.mean(np.absolute(self.V_tot[int(0.5*self.profile.n_slices):])))

    def llrf_model(self):
        """Models the LLRF part of the OTFB.

        Attributes
        ----------
        V_set : complex array
            Voltage set point [V] in (I,Q); :math:`V_{\mathsf{set}}`, amplitude
            proportional to voltage partition
        V_gen : complex array
            Generator voltage [V] in (I,Q); 
            :math:`V_{\mathsf{gen}} = V_{\mathsf{set}} - V_{\mathsf{tot}}`

        """

        # Voltage set point of current turn (I,Q); depends on voltage partition
        # Sinusoidal voltage completely in Q
        self.V_set = polar_to_cartesian(self.V_part *
                                        self.rf.voltage[0,
                                                        self.counter], self.rf.phi_rf[0, self.counter]
                                        + 0.5*np.pi)
        # Convert to array
#        self.V_set *= np.concatenate((np.ones(1000), np.zeros(self.n_llrf - 1000)))
        self.V_set *= np.ones(self.n_llrf)

        # Difference of set point and actual voltage
        self.V_gen = self.V_set - self.open_loop*np.concatenate((self.V_tot,
                                                                 np.zeros(self.n_diff, dtype=complex)))

        # Closed-loop gain
        self.V_gen *= self.G_llrf
        self.logger.debug("Set voltage %.6f MV",
                          1e-6*np.mean(np.absolute(self.V_set)))
        self.logger.debug("Antenna voltage %.6f MV",
                          1e-6*np.mean(np.absolute(self.V_tot)))
        self.logger.debug("Voltage error %.6f MV",
                          1e-6*np.mean(np.absolute(self.V_gen)))

        # One-turn delay comb filter; memorise the value of the previous turn
        self.V_gen = comb_filter(self.V_gen_prev, self.V_gen, self.a_comb)
        self.V_gen_prev = np.copy(self.V_gen)

        # Modulate from omega_rf to omega_r
        self.V_gen = modulator(self.V_gen, self.omega_c, self.omega_r,
                               self.profile.bin_size)

        # Shift signals with the delay time
        V_gen_in = np.copy(self.V_gen)
        self.V_gen = np.concatenate((self.V_mov_av_prev[-self.n_delay:],
                                     self.V_gen[:self.n_llrf-self.n_delay]))

        # Cavity filter: CIRCULAR moving average over filling time
        # Memorize last points of previous turn for beginning of next turn
        self.V_gen = moving_average(
            self.V_gen, self.n_mov_av, x_prev=self.V_mov_av_prev[-self.n_delay-self.n_mov_av:-self.n_delay])
        self.V_mov_av_prev = np.copy(V_gen_in)

    def generator_induced_voltage(self):
        """Calculates the generator-induced voltage. The transmitter model is
        a simple linear gain [C/V] converting voltage to charge.

        ..math:: I = G_{\mathsfs{tx}} \frac{V}{R_{\mathsf{gen}}},

        where :math:`R_{\mathsf{gen}}` is the generator resistance,
        :py:attr:`llrf.impulse_response.TravellingWaveCavity.R_gen`

        Attributes
        ----------
        I_gen : complex array
            RF component of the generator charge [C] at the present time step
        V_ind_gen : complex array
            Induced voltage [V] from generator-cavity interaction

        """

        # Add correction to the drive already existing
        self.V_gen = self.open_FB*modulator(self.V_gen, self.omega_r,
                                            self.omega_c, self.profile.bin_size) + self.open_drive*self.V_set

        # Generator charge from voltage, transmitter model
        self.I_gen = self.G_tx*self.V_gen/self.TWC.R_gen*self.profile.bin_size

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

        :seealso: :py:class:`llrf.impulse_response.TravellingWaveCavity`

        The impulse response is made to be the same length as the beam profile.

        """

        self.__setattr__("V_ind_"+name,
                         self.matr_conv(self.__getattribute__("I_"+name),
                                        self.TWC.__getattribute__("h_"+name)))
        self.logger.debug("Matrix convolution for V_ind")

        if name == "beam":
            self.V_ind_beam = -self.n_cavities \
                * self.V_ind_beam[:self.profile.n_slices]  # WHAT IS THE CORRECT SIGN???

        elif name == "gen":
            # Circular convolution
            self.V_ind_gen = +self.n_cavities \
                * self.V_ind_gen[self.n_mov_av:self.n_llrf+self.n_mov_av]

    def beam_induced_voltage(self, lpf=True):
        """Calculates the beam-induced voltage 

        Parameters
        ----------
        lpf : bool
            Apply low-pass filter for beam current calculation; default is True 

        Attributes
        ----------
        I_beam : complex array
            RF component of the beam charge [C] at the present time step
        V_ind_beam : complex array
            Induced voltage [V] from beam-cavity interaction

        """

        # Beam current from profile
        self.I_beam = rf_beam_current(self.profile, self.omega_c,
                                      self.rf.t_rev[self.counter], lpf=lpf)
        # Beam-induced voltage
        self.induced_voltage('beam')

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
