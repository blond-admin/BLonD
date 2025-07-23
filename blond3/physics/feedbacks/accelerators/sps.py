from __future__ import annotations

import logging
import sys
from typing import Optional, Optional as LateInit, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray as NumpyArray
from scipy.signal import fftconvolve

from .sps_helpers import get_power_gen_i, moving_average, comb_filter, modulator
from ..cavity_feedback import BirksCavityFeedback
from ..helpers import cartesian_to_polar
from ....physics.cavities import MultiHarmonicCavity
from ....physics.profiles import StaticProfile


class SPSCavityLoopCommissioning:
    r"""Class containing commissioning settings for the cavity feedback

    Parameters
    ----------
    debug : bool
        Debugging output active (True/False); default is False
    open_loop : int(bool)
        Open (True) or closed (False) cavity loop; default is False
    open_fb : int(bool)
        Open (True) or closed (False) feedback; default is False
    open_drive : int(bool)
        Open (True) or closed (False) drive; default is False
    open_ff : int(bool)
        Open (True) or closed (False) feed-forward; default is True.
    v_set : complex array
        Array set point voltage; default is False
    cpp_conv : bool
        Enable (True) or disable (False) convolutions using a C++ implementation; default is False
    pwr_clamp : bool
        Enable (True) or disable (False) power clamping; default is False
    rot_iq : complex
        Option to rotate the set point and beam induced voltages in the complex plane.
    excitation : bool
        Excite the model with white noise to perform BBNA measurements
    """

    def __init__(
        self,
        debug: bool = False,
        open_loop: bool = False,
        open_fb: bool = False,
        open_drive: bool = False,
        open_ff: bool = True,
        v_set: Optional[NumpyArray] = None,
        cpp_conv: bool = False,
        pwr_clamp: bool = False,
        rot_iq: complex = 1,
        excitation: bool = False,
    ):
        self.debug = bool(debug)
        self.open_loop = 0 if open_loop else 1
        self.open_fb = 0 if open_fb else 1
        self.open_drive = 0 if open_drive else 1
        self.open_ff = 0 if open_ff else 1
        self.V_SET = v_set
        self.cpp_conv = cpp_conv
        self.pwr_clamp = pwr_clamp
        self.rot_iq = rot_iq
        self.excitation: int = int(excitation)


class SPSOneTurnFeedback(BirksCavityFeedback):
    r"""The SPS one-turn delay feedback and feedforward model in BLonD for a single cavity type.

    Parameters
    ----------
    _parent_cavity : class
        An RFStation type class
    profile : class
        A Profile type class
    n_sections : int
        Number of sections of the traveling wave cavity
    n_cavities : int
        Number of traveling wave cavities of this type; default is 4
    V_part : float
        Partitioning of the total voltage onto this cavity type; default is 4/9
    G_ff : float
        Feedforward gain; default is 1
    G_llrf : float
        Low-level RF gain; default is 10
    G_tx : float
        Transmitter gain; default is 1
    a_comb : float
        Comb filter coefficient; default is 63/64
    df : float
        Change of the TWC central frequency in Hz from the 2021 measurement; default is 0 Hz
    commissioning : class
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    """

    def __init__(
        self,
        _parent_cavity: MultiHarmonicCavity,
        profile: StaticProfile,
        n_sections: int,
        n_cavities: int = 4,
        V_part: float = 4 / 9,
        G_ff: float = 1,
        G_llrf: float = 10,
        G_tx: float = 1,
        a_comb: float = 63 / 64,
        df: float = 0,
        commissioning: Optional[SPSCavityLoopCommissioning] = None,
        harmonic_index: int = 0,
    ):
        self.V_set: LateInit[NumpyArray] = None
        self.n_delay: LateInit[int] = None

        super().__init__(
            _parent_cavity=_parent_cavity,
            profile=profile,
            n_cavities=n_cavities,
            n_periods_coarse=1,
            harmonic_index=harmonic_index,
        )

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        if commissioning is None:
            commissioning = SPSCavityLoopCommissioning()

        # Commissioning options
        self.open_loop = commissioning.open_loop
        if self.open_loop == 0:  # Open Loop
            self.logger.debug("Opening overall OTFB loop")
        elif self.open_loop == 1:
            self.logger.debug("Closing overall OTFB loop")
        self.open_fb = commissioning.open_fb
        if self.open_fb == 0:  # Open Feedback
            self.logger.debug("Opening feedback of drive correction")
        elif self.open_fb == 1:
            self.logger.debug("Closing feedback of drive correction")
        self.open_drive = commissioning.open_drive
        if self.open_drive == 0:  # Open Drive
            self.logger.debug("Opening drive to generator")
        elif self.open_drive == 1:
            self.logger.debug("Closing drive to generator")
        self.open_ff = commissioning.open_ff
        if self.open_ff == 0:  # Open Feedforward
            self.logger.debug("Opening feed-forward on beam current")
        elif self.open_ff == 1:
            self.logger.debug("Closing feed-forward on beam current")
        self.V_SET = commissioning.V_SET
        if self.V_SET is None:  # Vset as array or not
            self.set_point_modulation = False
        else:
            self.set_point_modulation = True

        self.cpp_conv = commissioning.cpp_conv
        self.rot_iq = commissioning.rot_iq
        self.excitation = commissioning.excitation

        self.n_sections = int(n_sections)

        self.V_part = float(V_part)
        if self.V_part * (1 - self.V_part) < 0:
            raise RuntimeError(
                "ERROR in SPSOneTurnFeedback: V_part should be in range (0,1)!"
            )

        # Gain settings
        self.G_ff = float(G_ff)
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx)

        # 200 MHz travelling wave cavity (TWC) model
        if n_sections in [3, 4, 5]:
            self.TWC = eval(
                "SPS" + str(n_sections) + "Section200MHzTWC(" + str(df) + ")"
            )
            if self.open_ff == 1:
                # Feed-forward filter
                self.coeff_ff = getattr(
                    sys.modules[__name__], "feedforward_filter_TWC" + str(n_sections)
                )
                self.n_ff = len(self.coeff_ff)  # Number of coefficients for FF
                self.n_ff_delay = round(
                    0.5 * (self.n_ff - 1) + 0.5 * self.TWC.tau / self.T_s / 5
                )

                self.logger.debug("Feed-forward delay in samples %d", self.n_ff_delay)

                # Multiply gain by normalisation factors from filter and
                # beam-to generator current
                self.G_ff *= self.TWC.R_beam / (self.TWC.R_gen * np.sum(self.coeff_ff))

        else:
            raise RuntimeError(
                "ERROR in SPSOneTurnFeedback: argument n_sections has invalid value!"
            )
        self.logger.debug(
            "SPS OTFB cavities: %d, sections: %d, voltage partition %.2f, gain: %.2e",
            self.n_cavities,
            n_sections,
            self.V_part,
            self.G_tx,
        )

        # Switch between convolution methods
        if self.cpp_conv:
            self.conv = getattr(self, "call_conv")
        else:
            self.conv = getattr(self, "matr_conv")

        # TWC resonant frequency
        self.omega_c = self.TWC.omega_r
        # Length of arrays in LLRF
        self.n_coarse_ff = int(self.n_coarse / 5)
        # Initialize turn-by-turn variables
        self.dphi_mod = 0

        # Check array length for set point modulation
        if self.set_point_modulation:
            if self.V_SET.shape[0] != 2 * self.n_coarse:
                raise RuntimeError("V_SET length should be %d" % (2 * self.n_coarse))
            self.set_point = getattr(self, "set_point_mod")
        else:
            self.set_point = getattr(self, "set_point_std")
            self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)

        # Array to hold the bucket-by-bucket voltage with length LLRF
        self.DV_GEN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.logger.debug("Length of arrays on coarse grid 2x %d", self.n_coarse)

        # Array if noise is being injected
        self.NOISE = np.zeros(2 * self.n_coarse, dtype=complex)

        # LLRF MODEL ARRAYS
        # Initialize comb filter
        self.DV_COMB_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.a_comb = float(a_comb)

        # Initialize the delayed signal
        self.DV_DELAYED = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize modulated signal (to fr)
        self.DV_MOD_FR = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize moving average
        self.n_mov_av = round(
            self.TWC.tau / self._parent_cavity.t_rf[self.harmonic_index]
        )
        self.DV_MOV_AVG = np.zeros(2 * self.n_coarse, dtype=complex)
        self.logger.debug("Moving average over %d points", self.n_mov_av)
        if self.n_mov_av < 2:
            raise RuntimeError(
                "ERROR in SPSOneTurnFeedback: profile has to"
                " have at least 12.5 ns resolution!"
            )

        # GENERATOR MODEL ARRAYS
        # Initialize modulated signal (to frf)
        self.DV_MOD_FRF = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialize induced voltage on coarse grid
        self.V_IND_COARSE_GEN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.CONV_RES = np.zeros(2 * self.n_coarse, dtype=complex)
        self.CONV_PREV = np.zeros(self.n_coarse, dtype=complex)

        # BEAM MODEL ARRAYS
        # Initialize induced beam voltage coarse and fine
        self.V_IND_FINE_BEAM = np.zeros(self.profile.n_bins, dtype=complex)
        self.V_IND_COARSE_BEAM = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialise feed-forward; sampled every fifth bucket
        if self.open_ff == 1:
            self.logger.debug("Feed-forward active")
            self.I_BEAM_COARSE_FF = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_BEAM_COARSE_FF_MOD = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_FF_CORR_MOD = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_FF_CORR_DEL = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_FF_CORR = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.V_FF_CORR = np.zeros(2 * self.n_coarse_ff, dtype=complex)

        # Update global cavity loop variables before tracking
        self.update_rf_variables()
        self.update_fb_variables()
        self.logger.info("Class initialized")

        self.V_ANT_START: LateInit[NumpyArray] = None
        self.V_ANT_FINE_START: LateInit[NumpyArray] = None
        self.phi_mod_0: LateInit[Any] = None

    def circuit_track(self, no_beam: bool = False):
        r"""Tracking the SPS CL internally."""

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_carrier, self.rf_centers)
        self.TWC.impulse_response_beam(
            self.omega_carrier, self.profile.hist_x, self.rf_centers
        )

        if not no_beam:
            # Beam-induced voltage from beam profile
            self.beam_model()

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.gen_model()

        # Sum generator- and beam-induced voltages for coarse grid
        self.V_ANT_START = np.copy(self.V_ANT_COARSE)
        self.V_ANT_COARSE[: self.n_coarse] = self.V_ANT_COARSE[-self.n_coarse :]
        self.V_ANT_COARSE[-self.n_coarse :] = (
            self.V_IND_COARSE_GEN[-self.n_coarse :]
            + self.V_IND_COARSE_BEAM[-self.n_coarse :]
        )

        # Obtain generator-induced voltage on the fine grid by interpolation
        self.V_ANT_FINE_START = np.copy(self.V_ANT_FINE)
        self.V_ANT_FINE[: self.profile.n_bins] = self.V_ANT_FINE[-self.profile.n_bins :]
        self.V_ANT_FINE[-self.profile.n_bins :] = self.V_IND_FINE_BEAM[
            -self.profile.n_bins :
        ] + np.interp(
            self.profile.hist_x,
            self.rf_centers,
            self.V_IND_COARSE_GEN[-self.n_coarse :],
        )
        self.V_ANT_FINE[-self.profile.n_bins :] = (
            self.n_cavities * self.V_ANT_FINE[-self.profile.n_bins :]
        )

    def llrf_model(self):
        r"""The LLRF model of the SPSOneTurnFeedback. This function calles the functions related
        to the LLRF part of the model in the correct order."""

        # Track all the modules of the LLRF-part of the model
        self.set_point()
        self.error_and_gain()
        self.comb()
        self.one_turn_delay()
        self.mod_to_fr()
        self.mov_avg()

    def gen_model(self):
        r"""The Generator model of the SPSOneTurnFeedback. This function calles the functions related
        to the generator part of the model in the correct order."""

        # Track all the modules for the generator part of the model
        self.mod_to_frf()
        self.sum_and_gain()
        self.gen_response()

    def beam_model(self):
        r"""The Beam model of the SPSOneTurnFeedback. This function find the RF beam current from the Profile-
        object, applies the cavity response towards the beam and the feed-forward correction if engaged.
        """

        # Rotate the RF beam current
        self.I_BEAM_FINE = self.rot_iq * self.I_BEAM_FINE
        self.I_BEAM_COARSE[-self.n_coarse :] = (
            self.rot_iq * self.I_BEAM_COARSE[-self.n_coarse :]
        )

        # Beam-induced voltage
        self.beam_response(coarse=False)
        self.beam_response(coarse=True)

        # Feed-forward
        if self.open_ff == 1:
            # Calculate correction based on previous turn on coarse grid

            # Resample RF beam current to FF sampling frequency
            self.I_BEAM_COARSE_FF[: self.n_coarse_ff] = self.I_BEAM_COARSE_FF[
                -self.n_coarse_ff :
            ]
            I_COARSE_BEAM_RESHAPED = np.copy(self.I_BEAM_COARSE[-self.n_coarse :])
            I_COARSE_BEAM_RESHAPED = I_COARSE_BEAM_RESHAPED.reshape(
                (self.n_coarse_ff, self.n_coarse // self.n_coarse_ff)
            )
            self.I_BEAM_COARSE_FF[-self.n_coarse_ff :] = (
                np.sum(I_COARSE_BEAM_RESHAPED, axis=1) / 5
            )

            # Do a down-modulation to the resonant frequency of the TWC
            self.I_BEAM_COARSE_FF_MOD[: self.n_coarse_ff] = self.I_BEAM_COARSE_FF_MOD[
                -self.n_coarse_ff :
            ]
            self.I_BEAM_COARSE_FF_MOD[-self.n_coarse_ff :] = modulator(
                self.I_BEAM_COARSE_FF[-self.n_coarse_ff :],
                omega_i=self.omega_carrier,
                omega_f=self.omega_c,
                T_sampling=5 * self.T_s,
                phi_0=self.dphi_mod,
                dt=self.dT,
            )

            self.I_FF_CORR[: self.n_coarse_ff] = self.I_FF_CORR[-self.n_coarse_ff :]
            self.I_FF_CORR[-self.n_coarse_ff :] = np.zeros(
                self.n_coarse_ff, dtype=complex
            )
            for ind in range(self.n_coarse_ff, 2 * self.n_coarse_ff):
                for k in range(self.n_ff):
                    self.I_FF_CORR[ind] += (
                        self.coeff_ff[k] * self.I_BEAM_COARSE_FF_MOD[ind - k]
                    )

            # Do a down-modulation to the resonant frequency of the TWC
            phi_delay = (
                self.n_ff_delay * self.T_s * 5 * (self.omega_c - self.omega_carrier)
            )
            self.I_FF_CORR_MOD[: self.n_coarse_ff] = self.I_FF_CORR_MOD[
                -self.n_coarse_ff :
            ]
            self.I_FF_CORR_MOD[-self.n_coarse_ff :] = modulator(
                self.I_FF_CORR[-self.n_coarse_ff :],
                omega_i=self.omega_c,
                omega_f=self.omega_carrier,
                T_sampling=5 * self.T_s,
                phi_0=-(self.dphi_mod + phi_delay),
                dt=self.dT,
            )

            # Compensate for FIR filter delay
            self.I_FF_CORR_DEL[: self.n_coarse_ff] = self.I_FF_CORR_DEL[
                -self.n_coarse_ff :
            ]
            self.I_FF_CORR_DEL[-self.n_coarse_ff :] = self.I_FF_CORR_MOD[
                self.n_ff_delay : self.n_ff_delay - self.n_coarse_ff
            ]

    # BEAM MODEL
    def beam_response(self, coarse: bool = False):
        r"""Computes the beam-induced voltage on the fine- and coarse-grid by convolving
        the RF beam current with the cavity response towards the beam. The voltage is
        multiplied by the number of cavities to find the total."""
        self.logger.debug("Matrix convolution for V_ind")

        if coarse:
            self.V_IND_COARSE_BEAM[: self.n_coarse] = self.V_IND_COARSE_BEAM[
                -self.n_coarse :
            ]
            self.V_IND_COARSE_BEAM[-self.n_coarse :] = (
                self.matr_conv(self.I_BEAM_COARSE, self.TWC.h_beam_coarse)[
                    -self.n_coarse :
                ]
                * self.T_s
            )
        else:
            # Only convolve the slices for the current turn because the fine grid points can be less
            # than one turn in length
            self.V_IND_FINE_BEAM[-self.profile.n_bins :] = (
                self.matr_conv(
                    self.I_BEAM_FINE[-self.profile.n_bins :], self.TWC.h_beam
                )[-self.profile.n_bins :]
                * self.profile.hist_step
            )

    # INDIVIDUAL COMPONENTS ---------------------------------------------------
    # LLRF MODEL

    def set_point_std(self):
        r"""Computes the desired set point voltage in I/Q."""

        self.logger.debug("Entering %s function" % sys._getframe(0).f_code.co_name)
        # Read RF voltage from rf object
        self.V_set = self.set_point_from_rfstation()
        self.V_set = (
            self.V_part
            * self.V_set
            * np.exp(1j * (-0.5 * np.pi + np.angle(self.rot_iq)))
        )

        # Convert to array
        self.V_SET[: self.n_coarse] = self.V_SET[-self.n_coarse :]
        self.V_SET[-self.n_coarse :] = self.V_set

    def set_point_mod(self):
        r"""This function is called instead of set_point_std if a modulated set point is used.
        That is, if the set point is non-constant over a turn with the periodicity of a turn.
        """

        self.logger.debug("Entering %s function" % sys._getframe(0).f_code.co_name)
        pass

    def error_and_gain(self):
        r"""This function computes the difference between the set point and the antenna voltage
        and amplifies it with the LLRF gain."""

        # Store last turn error signal and update for current turn
        self.DV_GEN[: self.n_coarse] = self.DV_GEN[-self.n_coarse :]
        self.DV_GEN[-self.n_coarse :] = self.G_llrf * (
            self.V_SET[-self.n_coarse :]
            - self.open_loop
            * (
                self.V_IND_COARSE_GEN[-self.n_coarse :]
                + self.V_IND_COARSE_BEAM[-self.n_coarse :]
            )
            + self.excitation * self.NOISE[-self.n_coarse :]
        )
        self.logger.debug(
            "In %s, average set point voltage %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.V_SET)),
        )
        self.logger.debug(
            "In %s, average antenna voltage %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.V_ANT_COARSE)),
        )
        self.logger.debug(
            "In %s, average voltage error %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.DV_GEN)),
        )

    def comb(self):
        r"""This function applies the comb filter to the error signal."""

        # Shuffle present data to previous data
        self.DV_COMB_OUT[: self.n_coarse] = self.DV_COMB_OUT[-self.n_coarse :]
        # Update present data
        self.DV_COMB_OUT[-self.n_coarse :] = comb_filter(
            self.DV_COMB_OUT[: self.n_coarse],
            self.DV_GEN[-self.n_coarse :],
            self.a_comb,
        )

    def one_turn_delay(self):
        r"""This function applies the complementary delay such that the correction is applied
        with exactly the delay of one turn."""

        # Store last turn delayed signal and compute current turn error signal
        self.DV_DELAYED[: self.n_coarse] = self.DV_DELAYED[-self.n_coarse :]
        self.DV_DELAYED[-self.n_coarse :] = self.DV_COMB_OUT[
            self.n_coarse - self.n_delay : -self.n_delay
        ]

    def mod_to_fr(self):
        r"""This function modulates the error signal to the resonant frequency of the cavity."""

        # Store last turn modulated signal
        self.DV_MOD_FR[: self.n_coarse] = self.DV_MOD_FR[-self.n_coarse :]
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        self.DV_MOD_FR[-self.n_coarse :] = modulator(
            self.DV_DELAYED[-self.n_coarse :],
            self.omega_carrier,
            self.omega_c,
            self.T_s,
            phi_0=self.dphi_mod,
            dt=self.dT,
        )

    def mov_avg(self):
        r"""This function applies the cavity filter, modelled as a moving average, to the modulated
        error signal."""

        # Store last turn moving average signal
        self.DV_MOV_AVG[: self.n_coarse] = self.DV_MOV_AVG[-self.n_coarse :]
        # Apply moving average filter for current turn
        self.DV_MOV_AVG[-self.n_coarse :] = moving_average(
            self.DV_MOD_FR[-self.n_mov_av - self.n_coarse + 1 :],
            self.n_mov_av,
        )

    # GENERATOR MODEL

    def mod_to_frf(self):
        r"""This function modulates the error signal from the resonant frequency of the cavity to the
        original carrier frequency, the RF frequency."""

        # Store last turn modulated signal
        self.DV_MOD_FRF[: self.n_coarse] = self.DV_MOD_FRF[-self.n_coarse :]
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        dphi_demod = (self.omega_c - self.omega_carrier) * self.TWC.tau
        self.DV_MOD_FRF[-self.n_coarse :] = self.open_fb * modulator(
            self.DV_MOV_AVG[-self.n_coarse :],
            self.omega_c,
            self.omega_carrier,
            self.T_s,
            phi_0=-(self.dphi_mod + dphi_demod),
            dt=self.dT,
        )

    def sum_and_gain(self):
        r"""Summing of the error signal from the LLRF-part of the model and the set point voltage.
        The generator current is then found by multiplying by the transmitter gain and R_gen. The feed-forward
        current will also be added to the generator current if enabled."""

        # Store generator current signal from the last turn
        self.I_GEN_COARSE[: self.n_coarse] = self.I_GEN_COARSE[-self.n_coarse :]
        # Compute current turn generator current
        self.I_GEN_COARSE[-self.n_coarse :] = (
            self.DV_MOD_FRF[-self.n_coarse :]
            + self.open_drive * self.V_SET[-self.n_coarse :]
        )
        # Apply amplifier gain
        self.I_GEN_COARSE[-self.n_coarse :] *= self.G_tx / self.TWC.R_gen
        if self.open_ff == 1:
            self.I_GEN_COARSE[-self.n_coarse :] = self.I_GEN_COARSE[
                -self.n_coarse :
            ] + self.G_ff * np.interp(
                self.rf_centers,
                self.rf_centers[::5],
                self.I_FF_CORR_DEL[-self.n_coarse_ff :],
            )

    def gen_response(self):
        r"""Generator current is convolved with cavity response towards the generator to get the
        generator-induced voltage. Multiplied by the number of cavities to find the total generator-
        induced voltage."""

        # Store generator-induced from last turn
        self.V_IND_COARSE_GEN[: self.n_coarse] = self.V_IND_COARSE_GEN[-self.n_coarse :]
        # Compute current turn generator-induced voltage
        self.V_IND_COARSE_GEN[-self.n_coarse :] = (
            self.matr_conv(self.I_GEN_COARSE, self.TWC.h_gen)[-self.n_coarse :]
            * self.T_s
        )

    def matr_conv(self, I: NumpyArray, h: NumpyArray) -> NumpyArray:
        r"""Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        return fftconvolve(I, h, mode="full")[: I.shape[0]]

    def call_conv(self, signal, kernel):
        r"""Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1, dtype=complex)
        np.convolve(signal, kernel, result=result, mode="full")

        return result

    def update_fb_variables(self):
        r"""Update variables in the feedback"""

        # TODO REMWORK/REMOVE
        t_rev = float(
            (2 * np.pi * self._parent_cavity.harmonic[self.harmonic_index])
            / self._parent_cavity._omega[self.harmonic_index]
        )
        # TODO REMWORK/REMOVE
        t_rf = t_rev / float(self._parent_cavity.harmonic[self.harmonic_index])

        # Phase offset at the end of a 1-turn modulated signal (for demodulated, multiply by -1 as c and r reversed)
        self.phi_mod_0 = (
            (self.omega_carrier_prev - self.omega_c)
            * (self.T_s_prev * self.n_coarse)
            % (2 * np.pi)
        )
        self.dphi_mod += self.phi_mod_0
        self.dphi_mod = self.dphi_mod % (2 * np.pi)

        # Present delay time
        self.n_mov_av = int(self.TWC.tau / t_rf)
        self.n_delay = self.n_coarse - self.n_mov_av

        if self.open_ff == 1:
            self.n_ff_delay = round(
                0.5 * (self.n_ff - 1) + 0.5 * self.TWC.tau / t_rf / 5
            )

    # Power related functions
    def calc_power(self):
        r"""Method to compute the generator power"""

        return get_power_gen_i(np.copy(self.I_GEN_COARSE), 50)

    def wo_clamping(self):
        pass

    def w_clamping(self):
        pass


class SPSCavityFeedback:
    """Class determining the turn-by-turn total RF voltage and phase correction
    originating from the individual cavity feedbacks. Assumes two 4-section and
    two 5-section travelling wave cavities in the pre-LS2 scenario and four
    3-section and two 4-section cavities in the post-LS2 scenario. The voltage
    partitioning is proportional to the number of sections.

    Parameters
    ----------
    _parent_cavity : class
        An RFStation type class
    profile : class
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
        same convention as G_ff; default is 0
    commissioning : class
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    """

    def __init__(
        self,
        _parent_cavity: MultiHarmonicCavity,
        profile: StaticProfile,
        G_ff: float | list = 1,
        G_llrf: float | list = 10,
        G_tx: list[float, list] = 0.5,
        a_comb: Optional[float] = None,
        turns: int = 1000,
        post_LS2: bool = True,
        V_part: Optional[float] = None,
        df: list[float] = 0,
        commissioning: Optional[list | SPSCavityLoopCommissioning] = None,
        n_h: int = 0,
    ):
        # Options for commissioning the feedback
        self.alpha_sum: LateInit[NumpyArray] = None
        self.V_sum: LateInit[NumpyArray] = None
        self.V_corr: LateInit[NumpyArray] = None

        if commissioning is None:
            commissioning = SPSCavityLoopCommissioning()

        self.rf_station = _parent_cavity

        # Parse input for gains
        if hasattr(G_ff, "__iter__"):
            G_ff_1 = G_ff[0]
            G_ff_2 = G_ff[1]
        else:
            G_ff_1 = G_ff
            G_ff_2 = G_ff

        if hasattr(G_llrf, "__iter__"):
            G_llrf_1 = G_llrf[0]
            G_llrf_2 = G_llrf[1]
        else:
            G_llrf_1 = G_llrf
            G_llrf_2 = G_llrf

        if hasattr(G_tx, "__iter__"):
            G_tx_1 = G_tx[0]
            G_tx_2 = G_tx[1]
        else:
            G_tx_1 = G_tx
            G_tx_2 = G_tx

        if hasattr(df, "__iter__"):
            df_1 = df[0]
            df_2 = df[1]
        else:
            df_1 = df
            df_2 = df

        if hasattr(commissioning, "__iter__"):
            commissioning_1 = commissioning[0]
            commissioning_2 = commissioning[1]
        else:
            commissioning_1 = commissioning
            commissioning_2 = commissioning

        # Voltage partitioning has to be a fraction
        if V_part and V_part * (1 - V_part) < 0:
            raise RuntimeError(
                "SPS cavity feedback: voltage partitioning has to be in the range (0,1)!"
            )

        # Voltage partition proportional to the number of sections
        if post_LS2:
            if not a_comb:
                a_comb = 63 / 64

            if V_part is None:
                V_part = 6 / 10
            self.OTFB_1 = SPSOneTurnFeedback(
                _parent_cavity=_parent_cavity,
                profile=profile,
                n_sections=3,
                n_cavities=4,
                V_part=V_part,
                G_ff=float(G_ff_1),
                G_llrf=float(G_llrf_1),
                G_tx=float(G_tx_1),
                a_comb=float(a_comb),
                df=float(df_1),
                commissioning=commissioning_1,
                harmonic_index=n_h,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                _parent_cavity=_parent_cavity,
                profile=profile,
                n_sections=4,
                n_cavities=2,
                V_part=1 - V_part,
                G_ff=float(G_ff_2),
                G_llrf=float(G_llrf_2),
                G_tx=float(G_tx_2),
                a_comb=float(a_comb),
                df=float(df_2),
                commissioning=commissioning_2,
                harmonic_index=n_h,
            )
        else:
            if not a_comb:
                a_comb = 15 / 16

            if V_part is None:
                V_part = 4 / 9
            self.OTFB_1 = SPSOneTurnFeedback(
                _parent_cavity=_parent_cavity,
                profile=profile,
                n_sections=4,
                n_cavities=2,
                V_part=V_part,
                G_ff=float(G_ff_1),
                G_llrf=float(G_llrf_1),
                G_tx=float(G_tx_1),
                a_comb=float(a_comb),
                df=float(df_1),
                commissioning=commissioning_1,
                harmonic_index=n_h,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                _parent_cavity=_parent_cavity,
                profile=profile,
                n_sections=5,
                n_cavities=2,
                V_part=1 - V_part,
                G_ff=float(G_ff_2),
                G_llrf=float(G_llrf_2),
                G_tx=float(G_tx_2),
                a_comb=float(a_comb),
                df=float(df_2),
                commissioning=commissioning_2,
                harmonic_index=n_h,
            )

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Initialise OTFB without beam
        self.turns = int(turns)
        if turns < 1:
            # FeedbackError
            raise RuntimeError(
                "ERROR in SPSCavityFeedback: 'turns' has to be a positive integer!"
            )
        self.track_init(debug=commissioning_1.debug)

        self.logger.info("Class initialized")

    def track(self):
        r"""Main tracking method for the SPSCavityFeedback. This tracks both cavity types
        with beam."""

        # Track the feedbacks for the two TWC types
        self.OTFB_1.track()
        self.OTFB_2.track()

        # Sum the fine-grid antenna voltage from the TWC types
        self.V_sum = (
            self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_bins :]
            + self.OTFB_2.V_ANT_FINE[-self.OTFB_2.profile.n_bins :]
        )

        # Convert to amplitude and phase modulation
        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.OTFB_1._parent_cavity.voltage[self.OTFB_1.harmonic_index]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.OTFB_1.profile.hist_x,
                self.OTFB_1.rf_centers,
                self.OTFB_1.V_SET[-self.OTFB_1.n_coarse :],
            )
        )

    def track_init(self, debug: bool = False):
        r"""Tracking of the SPSCavityFeedback without beam."""

        if debug:
            cmap = plt.get_cmap("jet")
            colors = cmap(np.linspace(0, 1, self.turns))
            plt.figure("Pre-tracking without beam")
            ax = plt.axes([0.18, 0.1, 0.8, 0.8])
            ax.grid()
            ax.set_ylabel("Voltage [V]")

        for i in range(self.turns):
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_1.track_no_beam()
            if debug:
                ax.plot(
                    self.OTFB_1.profile.hist_x * 1e6,
                    np.abs(self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_bins :]),
                    color=colors[i],
                )
                ax.plot(
                    self.OTFB_1.rf_centers * 1e6,
                    self.OTFB_1.n_cavities
                    * np.abs(self.OTFB_1.V_ANT_COARSE[-self.OTFB_1.n_coarse :]),
                    color=colors[i],
                    linestyle="",
                    marker=".",
                )
            self.OTFB_2.track_no_beam()
        if debug:
            plt.show()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            self.OTFB_1.profile.hist_x,
            self.OTFB_1.rf_centers,
            self.OTFB_1.n_cavities
            * self.OTFB_1.V_IND_COARSE_GEN[-self.OTFB_1.n_coarse :]
            + self.OTFB_2.n_cavities
            * self.OTFB_2.V_IND_COARSE_GEN[-self.OTFB_2.n_coarse :],
        )

        # Convert to amplitude and phase
        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.OTFB_1._parent_cavity.voltage[self.OTFB_1.harmonic_index]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.OTFB_1.profile.hist_x,
                self.OTFB_1.rf_centers,
                self.OTFB_1.V_SET[-self.OTFB_1.n_coarse :],
            )
        )
