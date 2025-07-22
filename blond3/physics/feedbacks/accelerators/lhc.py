from __future__ import annotations

import logging
from typing import Any, Optional, Optional as LateInit

import numpy as np
from numpy import random as rnd
from numpy._typing import NDArray as NumpyArray
from scipy.interpolate import interp1d

from blond3 import StaticProfile
from ..base import LocalFeedback, GlobalFeedback
from ..cavity_feedback import BirksCavityFeedback
from .lhc_helpers import (
    smooth_step,
    cavity_response_sparse_matrix,
    fir_filter_lhc_otfb_coeff,
)
from ...cavities import SingleHarmonicCavity, MultiHarmonicCavity
from ...profiles import ProfileBaseClass


class LhcBeamFeedBack(GlobalFeedback):
    def __init__(self, profile: ProfileBaseClass, section_index: int = 0):
        super().__init__(
            profile=profile,
            section_index=section_index,
        )


class LhcRfFeedback(LocalFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        cavity: SingleHarmonicCavity | MultiHarmonicCavity,
        section_index: int = 0,
    ):
        super().__init__(
            profile=profile,
            cavity=cavity,
            section_index=section_index,
        )


class LHCCavityLoopCommissioning:
    r"""RF Feedback settings for LHC ACS cavity loop.

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
    mu : float
        Coefficient for the tuner algorithm determining time scale; default is -0.0001
    power_thres : float
        Available RF power in the klystron; default is 300 kW
    open_drive : bool
        Open (True) or closed (False) cavity loop at drive; default is False
    open_loop : bool
        Open (True) or closed (False) cavity loop at RFFB; default is False
    open_otfb : bool
        Open (true) or closed (False) one-turn feedback; default is False
    open_rffb : bool
        Open (True) or closed (False) RFFB; default is False
    open_tuner : bool
        Open (True) or closed (False) tuner control; default is False
    clamping : bool
        Simulate clamping (True) or not (False); default is False
    excitation : bool
        Perform BBNA measurement of the feedback (True); default is False
    """

    def __init__(
        self,
        alpha: float = 15 / 16,
        d_phi_ad: float = 0,
        G_a: float = 0.00001,
        G_d: float = 10,
        G_o: float = 10,
        tau_a: float = 170e-6,
        tau_d: float = 400e-6,
        tau_o: float = 110e-6,
        mu: float = -0.0001,
        power_thres: float = 300e3,
        open_drive: bool = False,
        open_loop: bool = False,
        open_otfb: bool = False,
        open_rffb: bool = False,
        open_tuner: bool = False,
        clamping: bool = False,
        excitation: bool = False,
        excitation_otfb_1: bool = False,
        excitation_otfb_2: bool = False,
        seed1: Any = 1234,
        seed2: Any = 7564,
    ):
        # Import variables
        self.alpha = alpha
        self.d_phi_ad = d_phi_ad * np.pi / 180
        self.G_a = G_a
        self.G_d = G_d
        self.G_o = G_o
        self.tau_a = tau_a
        self.tau_d = tau_d
        self.tau_o = tau_o
        self.mu = mu
        self.power_thres = power_thres
        self.excitation = excitation
        self.excitation_otfb_1 = excitation_otfb_1
        self.excitation_otfb_2 = excitation_otfb_2
        self.seed1 = seed1
        self.seed2 = seed2

        # Multiply with zeros if open == True
        self.open_drive = 0 if open_drive else 1
        self.open_drive_inv = 0 if self.open_drive else 1
        self.open_loop = 0 if open_loop else 1
        self.open_otfb = 0 if open_otfb else 1
        self.open_rffb = 0 if open_rffb else 1
        self.open_tuner = 0 if open_tuner else 1

        self.clamping = clamping

    def generate_white_noise(self, n_points: int):
        r"""Generates white noise"""

        rnd.seed(self.seed1)
        r1 = rnd.random_sample(n_points)
        rnd.seed(self.seed2)
        r2 = rnd.random_sample(n_points)

        return np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))


class LHCCavityLoop(BirksCavityFeedback):
    r"""Cavity loop to regulate the RF voltage in the LHC ACS cavities.
    The loop contains a generator, a switch-and-protect device, an RF FB and a
    OTFB. The arrays of the LLRF system cover one turn with exactly one tenth
    of the harmonic (i.e.\ the typical sampling time is about 25 ns).

    Parameters
    ----------
    _parent_cavity : class
        An RFStation type class
    profile : class
        Beam profile object
    n_cavities : int
        Number of cavities per beam; default is 8
    f_c : float
        Central cavity frequency [Hz]; default is 400.789e6 Hz
    G_gen : float
        Overall driver chain gain [1]; default is 1
    I_gen_offset : float
        Generator current offset [A]; default is 0
    n_pretrack : int
        Number of turns to pre-track without beam; default is 200
    Q_L : float
        Cavity loaded quality factor; default is 20000
    R_over_Q : float
        Cavity R/Q [Ohm]; default is 45 Ohms
    tau_loop : float
        Total loop delay [s]; default is 650e-9 s
    tau_otfb : float
        Total loop delay as seen by OTFB [s]; default is 1472e-9 s
    RFFB : class
        LHCCavityLoopCommissioning type class containing RF FB gains and delays. If this parameter is None, a new
        LHCCavityLoopCommissioning is used.
    """

    def __init__(
        self,
        _parent_cavity: MultiHarmonicCavity,
        profile: StaticProfile,
        n_cavities: int = 8,
        f_c: float = 400.789e6,
        G_gen: float = 1,
        I_gen_offset: float = 0,
        n_pretrack: int = 200,
        Q_L: float = 20000,
        R_over_Q: float = 45,
        tau_loop: float = 650e-9,
        tau_otfb: float = 1472e-9,
        RFFB: Optional[LHCCavityLoopCommissioning] = None,
        harmonic_index: int = 0,
    ):
        super().__init__(
            _parent_cavity=_parent_cavity,
            profile=profile,
            n_cavities=n_cavities,
            n_periods_coarse=10,
            harmonic_index=harmonic_index,
        )
        # variables that are declared later
        self.samples: LateInit[float] = None
        self.n_delay: LateInit[int] = None
        self.n_fir: LateInit[int] = None
        self.n_otfb: LateInit[int] = None
        self.ind: LateInit[int] = None
        self.samples_fine: LateInit[float] = None
        self.detuning: LateInit[float] = None
        self.d_omega: LateInit[float] = None
        self.omega_c: LateInit[float] = None

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Options for commissioning the feedback
        if RFFB is None:
            RFFB = LHCCavityLoopCommissioning()

        # Import classes and parameters
        self.RFFB = RFFB
        self.I_gen_offset = I_gen_offset
        self.G_gen = G_gen
        self.n_pretrack = n_pretrack
        self.omega_c = 2 * np.pi * f_c
        # TODO: implement optimum loaded Q
        self.Q_L = Q_L
        self.R_over_Q = R_over_Q
        self.tau_loop = tau_loop
        self.tau_otfb = tau_otfb
        self.logger.debug("Cavity loaded Q is %.0f", self.Q_L)

        # Import RF FB properties
        self.open_drive = self.RFFB.open_drive
        self.open_drive_inv = self.RFFB.open_drive_inv
        self.open_loop = self.RFFB.open_loop
        self.open_otfb = self.RFFB.open_otfb
        self.open_rffb = self.RFFB.open_rffb
        self.open_tuner = self.RFFB.open_tuner
        self.clamping = self.RFFB.clamping
        self.alpha = self.RFFB.alpha
        self.d_phi_ad = self.RFFB.d_phi_ad
        self.G_a = self.RFFB.G_a
        self.G_d = self.RFFB.G_d
        self.G_o = self.RFFB.G_o
        self.tau_a = self.RFFB.tau_a
        self.tau_d = self.RFFB.tau_d
        self.tau_o = self.RFFB.tau_o
        self.mu = self.RFFB.mu
        self.power_thres = self.RFFB.power_thres
        self.v_swap_thres = (
            np.sqrt(2 * self.power_thres / (self.R_over_Q * self.Q_L)) / self.G_gen
        )
        self.excitation = self.RFFB.excitation
        self.excitation_otfb_1 = self.RFFB.excitation_otfb_1
        self.excitation_otfb_2 = self.RFFB.excitation_otfb_2

        self.logger.debug("Length of arrays in generator path %d", self.n_coarse)

        # Initialise FIR filter for OTFB
        self.fir_n_taps = 63
        self.fir_coeff = fir_filter_lhc_otfb_coeff(n_taps=self.fir_n_taps)
        self.logger.debug("Sum of FIR coefficients %.4e" % np.sum(self.fir_coeff))

        self.update_rf_variables()
        self.update_fb_variables()
        self.logger.debug("Relative detuning is %.4e", self.detuning)

        # Arrays
        self.V_EXC = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_FB_IN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_AC_IN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_AN_IN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_AN_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_DI_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_OTFB = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_OTFB_INT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_FIR_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_FB_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_SWAP_OUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_TEST = np.zeros(2 * self.n_coarse, dtype=complex)
        self.TUNER_INPUT = np.zeros(2 * self.n_coarse, dtype=complex)
        self.TUNER_INTEGRATED = np.zeros(2 * self.n_coarse, dtype=complex)

        self.V_ANT_FINE = np.zeros(self.profile.n_bins + 1, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_bins + 1, dtype=complex)

        # Pre-track without beam
        self.logger.debug("Track without beam for %d turns", self.n_pretrack)
        if self.excitation:
            self.excitation_otfb = False
            self.logger.debug("Injecting noise in voltage set point")
            self.track_no_beam_excitation(self.n_pretrack)
        elif self.excitation_otfb_1 or self.excitation_otfb_2:
            self.excitation_otfb = True
            self.logger.debug("Injecting noise at OTFB output")
            self.track_no_beam_excitation_otfb(self.n_pretrack)
        else:
            self.excitation_otfb = False
            self.logger.debug("Pre-tracking without beam")
            self.track_no_beam(self.n_pretrack)

        self.logger.info("LHCCavityLoop class initialized")

        self.V_EXC_IN: LateInit = None
        self.V_EXC_OUT: LateInit = None
        # self.xxx: LateInit = None

    def circuit_track(self, no_beam: bool = False):
        r"""Track the feedback model"""
        if not no_beam:
            self.I_BEAM_FINE *= -1j * np.exp(1j * self._parent_cavity.phi_s)
            self.I_BEAM_COARSE[-self.n_coarse :] *= -1j * np.exp(
                1j * self._parent_cavity.phi_s
            )

        # Track the different parts of the model
        self.update_arrays()
        self.update_set_point()
        self.track_one_turn()

        if not no_beam:
            # Resample generator current to the fine-grid
            self.I_GEN_FINE = np.interp(
                np.concatenate(
                    (
                        np.array([self.profile.hist_x[0] - self.profile.hist_step]),
                        self.profile.hist_x,
                    )
                ),
                self.rf_centers,
                self.I_GEN_COARSE[-self.n_coarse :],
            )

            # Compute the fine-grid antenna voltage through solving a sparse matrix equation
            self.cavity_response_fine_matrix()

            # Apply the tuner correction
            self.tuner()

    def cavity_response(self, samples: float):
        r"""ACS cavity reponse model"""

        self.V_ANT_COARSE[self.ind] = (
            self.I_GEN_COARSE[self.ind - 1] * self.R_over_Q * samples
            + self.V_ANT_COARSE[self.ind - 1]
            * (1 - 0.5 * samples / self.Q_L + 1j * self.detuning * samples)
            - self.I_BEAM_COARSE[self.ind - 1] * 0.5 * self.R_over_Q * samples
        )

    def cavity_response_fine_matrix(self):
        r"""ACS cavity response model in matrix form on the fine-grid"""

        # Number of samples on fine grid
        self.samples_fine = self.omega_rf * self.profile.hist_step

        # Find initial value of antenna voltage and generator current
        t_at_init = self.profile.hist_x[0] - self.profile.hist_x
        V_A_init = interp1d(
            np.concatenate(
                (self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)
            ),
            self.V_ANT_COARSE,
            fill_value="extrapolate",
        )(t_at_init)
        I_gen_init = interp1d(
            np.concatenate(
                (self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)
            ),
            self.I_BEAM_COARSE,
            fill_value="extrapolate",
        )(t_at_init)

        self.V_ANT_FINE = cavity_response_sparse_matrix(
            I_beam=self.I_BEAM_FINE,
            I_gen=self.I_GEN_FINE,
            n_samples=self.profile.n_bins,
            V_ant_init=V_A_init,
            I_gen_init=I_gen_init,
            samples_per_rf=self.samples_fine,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            detuning=self.detuning,
        )

        self.V_ANT_FINE[-self.profile.n_bins :] = (
            self.n_cavities * self.V_ANT_FINE[-self.profile.n_bins :]
        )

    def generator_current(self):
        r"""Generator response

        Attributes
        I_TEST : complex array
            Test point for open loop measurements (when injecting a generator
            offset)
        """

        # From V_swap_out in closed loop, constant in open loop
        # TODO: missing terms for changing voltage and beam current
        self.I_TEST[self.ind] = self.G_gen * self.V_SWAP_OUT[self.ind]
        self.I_GEN_COARSE[self.ind] = (
            self.open_drive * self.I_TEST[self.ind]
            + self.open_drive_inv * self.I_gen_offset
        )

    def generator_power(self) -> NumpyArray:
        r"""Calculation of generator power from generator current"""

        return 0.5 * self.R_over_Q * self.Q_L * np.absolute(self.I_GEN_COARSE) ** 2

    def one_turn_feedback(self, T_s: float):
        r"""Apply effect of the OTFB on the analog branch"""

        # OTFB itself
        self.V_OTFB_INT[self.ind] = (
            self.alpha * self.V_OTFB_INT[self.ind - self.n_coarse]
            + self.G_o
            * (1 - self.alpha)
            * self.V_AC_IN[self.ind - self.n_coarse + self.n_otfb]
        )

        # FIR filter
        self.V_FIR_OUT[self.ind] = self.fir_coeff[0] * self.V_OTFB_INT[self.ind]
        for k in range(1, self.fir_n_taps):
            self.V_FIR_OUT[self.ind] += (
                self.fir_coeff[k] * self.V_OTFB_INT[self.ind - k]
            )

        # AC coupling at output
        self.V_OTFB[self.ind] = (
            (1 - T_s / self.tau_o) * self.V_OTFB[self.ind - 1]
            + self.V_FIR_OUT[self.ind]
            - self.V_FIR_OUT[self.ind - 1]
        )

    def rf_feedback(self, T_s: float):
        r"""Analog and digital RF feedback response"""

        # Calculate voltage difference to act on
        self.V_FB_IN[self.ind] = (
            self.V_SET[self.ind - self.n_delay]
            - self.open_loop * self.V_ANT_COARSE[self.ind - self.n_delay]
        )

        # On the analog branch, OTFB can contribute
        self.V_AC_IN[self.ind] = (
            (1 - T_s / self.tau_o) * self.V_AC_IN[self.ind - 1]
            + self.V_FB_IN[self.ind]
            - self.V_FB_IN[self.ind - 1]
        )
        self.one_turn_feedback(T_s=T_s)

        self.V_AN_IN[self.ind] = (
            self.V_FB_IN[self.ind]
            + self.open_otfb * self.V_OTFB[self.ind]
            + int(bool(self.excitation_otfb)) * self.V_EXC[self.ind]
        )

        # Output of analog feedback (separate branch)
        self.V_AN_OUT[self.ind] = self.V_AN_OUT[self.ind - 1] * (
            1 - T_s / self.tau_a
        ) + self.G_a * (self.V_AN_IN[self.ind] - self.V_AN_IN[self.ind - 1])

        # Output of digital feedback (separate branch)
        self.V_DI_OUT[self.ind] = (
            self.V_DI_OUT[self.ind - 1] * (1 - T_s / self.tau_d)
            + T_s
            / self.tau_d
            * self.G_a
            * self.G_d
            * np.exp(1j * self.d_phi_ad)
            * self.V_FB_IN[self.ind - 1]
        )

        # Total output: sum of analog and digital feedback
        self.V_FB_OUT[self.ind] = self.open_rffb * (
            self.V_AN_OUT[self.ind] + self.V_DI_OUT[self.ind]
        )

    def update_set_point(self):
        r"""Updates the set point for the next turn based on the design RF
        voltage."""
        coeff = np.polyfit(
            [0, self.n_coarse + 1],
            [self.V_SET[-self.n_coarse], self.set_point_from_rfstation()[0]],
            1,
        )
        poly = np.poly1d(coeff)
        v_set_prev = poly(np.linspace(0, self.n_coarse, self.n_coarse))

        self.V_SET = np.concatenate((v_set_prev, self.set_point_from_rfstation()))

    def swap(self):
        r"""Model of the Switch and Protect module: clamping of the output
        power above a given input power."""

        # TODO: check implementation
        if self.clamping:
            self.V_SWAP_OUT[self.ind] = (
                self.v_swap_thres
                * smooth_step(
                    np.abs(self.V_FB_OUT[self.ind]), x_max=self.v_swap_thres, N=0
                )
                * np.exp(1j * np.angle(self.V_FB_OUT[self.ind]))
            )
        else:
            self.V_SWAP_OUT[self.ind] = self.V_FB_OUT[self.ind]

    def tuner(self):
        r"""Model of the tuner algorithm."""

        # Compute the detuning factor for the current turn
        dtune = (
            -(self.mu / 2)
            * (
                np.min(self.TUNER_INTEGRATED[-self.n_coarse :].imag)
                + np.max(self.TUNER_INTEGRATED[-self.n_coarse :].imag)
            )
            / (self._parent_cavity.voltage[self.harmonic_index] / self.n_cavities) ** 2
        )

        # Propagate the corrections to the detuning two the global parameters
        self.detuning = self.detuning + dtune * self.open_tuner
        self.d_omega = self.detuning * self.omega_c
        self.omega_c = self.omega_rf + self.d_omega

    def tuner_input(self):
        r"""Gathering data for the detuning algorithm"""

        # Calculating input signal
        self.TUNER_INPUT[self.ind] = self.I_GEN_COARSE[self.ind] * np.conj(
            self.V_ANT_COARSE[self.ind]
        )

        # Apply CIC-component
        self.TUNER_INTEGRATED[self.ind] = (
            (1 / 64)
            * (
                self.TUNER_INPUT[self.ind]
                - 2 * self.TUNER_INPUT[self.ind - 8]
                + self.TUNER_INPUT[self.ind - 16]
            )
            + 2 * self.TUNER_INTEGRATED[self.ind - 1]
            - self.TUNER_INTEGRATED[self.ind - 2]
        )

    def track_one_turn(self):
        r"""Single-turn tracking, index by index."""

        for i in range(self.n_coarse):
            T_s = self.T_s
            self.ind = i + self.n_coarse
            self.cavity_response(samples=T_s * self.omega_rf)
            self.rf_feedback(T_s=T_s)
            self.swap()
            self.generator_current()
            self.tuner_input()

    def update_arrays(self):
        r"""Moves the array indices by one turn (n_coarse points) from the
        present turn to prepare the next turn. All arrays except for V_SET."""

        self.V_ANT_COARSE = np.concatenate(
            (self.V_ANT_COARSE[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_FB_IN = np.concatenate(
            (self.V_FB_IN[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_AC_IN = np.concatenate(
            (self.V_AC_IN[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_AN_IN = np.concatenate(
            (self.V_AN_IN[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_AN_OUT = np.concatenate(
            (self.V_AN_OUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_DI_OUT = np.concatenate(
            (self.V_DI_OUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_OTFB = np.concatenate(
            (self.V_OTFB[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_OTFB_INT = np.concatenate(
            (self.V_OTFB_INT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_FIR_OUT = np.concatenate(
            (self.V_FIR_OUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_FB_OUT = np.concatenate(
            (self.V_FB_OUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.V_SWAP_OUT = np.concatenate(
            (self.V_SWAP_OUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.I_GEN_COARSE = np.concatenate(
            (self.I_GEN_COARSE[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.I_TEST = np.concatenate(
            (self.I_TEST[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.TUNER_INPUT = np.concatenate(
            (self.TUNER_INPUT[self.n_coarse :], np.zeros(self.n_coarse, dtype=complex))
        )
        self.TUNER_INTEGRATED = np.concatenate(
            (
                self.TUNER_INTEGRATED[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )

    def update_fb_variables(self):
        r"""Update counter and frequency-dependent variables in a given turn"""

        # Delay time
        self.n_delay = round(self.tau_loop / self.T_s)
        self.n_fir = round(0.5 * (self.fir_n_taps - 1))
        self.n_otfb = round(self.tau_otfb / self.T_s) + self.n_fir

        # Present detuning
        self.d_omega = self.omega_c - self.omega_rf

        # Dimensionless quantities
        self.samples = self.omega_rf * self.T_s
        self.detuning = self.d_omega / self.omega_c

    def update_set_point_excitation(self, excitation: NumpyArray, turn: int):
        r"""Updates the set point for the next turn based on the excitation to
        be injected."""

        self.V_SET = np.concatenate(
            (
                self.V_SET[self.n_coarse :],
                excitation[turn * self.n_coarse : (turn + 1) * self.n_coarse],
            )
        )

    def track_no_beam_excitation(self, n_turns: int):
        r"""Pre-tracking for n_turns turns, without beam. With excitation; set
        point from white noise. V_EXC_IN and V_EXC_OUT can be used to measure
        the transfer function of the system at set point.

        Notes
        -----
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """

        self.V_EXC_IN = 1000 * self.RFFB.generate_white_noise(self.n_coarse * n_turns)
        self.V_EXC_OUT = np.zeros(self.n_coarse * n_turns, dtype=complex)
        self.V_SET = np.concatenate(
            (np.zeros(self.n_coarse, dtype=complex), self.V_EXC_IN[0 : self.n_coarse])
        )
        self.track_one_turn()
        self.V_EXC_OUT[0 : self.n_coarse] = self.V_ANT_COARSE[
            self.n_coarse : 2 * self.n_coarse
        ]
        for n in range(1, n_turns):
            self.update_arrays()
            self.update_set_point_excitation(self.V_EXC_IN, n)
            self.track_one_turn()
            self.V_EXC_OUT[n * self.n_coarse : (n + 1) * self.n_coarse] = (
                self.V_ANT_COARSE[self.n_coarse : 2 * self.n_coarse]
            )

    def track_no_beam_excitation_otfb(self, n_turns: int):
        r"""Pre-tracking for n_turns turns, without beam. With excitation; set
        point from white noise. V_EXC_IN and V_EXC_OUT can be used to measure
        the transfer function of the system at otfb.

        Notes
        -----
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """

        self.V_EXC_IN = 10000 * self.RFFB.generate_white_noise(self.n_coarse * n_turns)
        self.V_EXC_OUT = np.zeros(self.n_coarse * n_turns, dtype=complex)
        self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_EXC = np.concatenate(
            (np.zeros(self.n_coarse, dtype=complex), self.V_EXC_IN[0 : self.n_coarse])
        )

        self.track_one_turn()
        if self.excitation_otfb_1:
            self.V_EXC_OUT[: self.n_coarse] = self.V_FB_IN[
                self.n_coarse : 2 * self.n_coarse
            ]
        elif self.excitation_otfb_2:
            self.V_EXC_OUT[: self.n_coarse] = self.V_OTFB[self.ind]
        for n in range(1, n_turns):
            self.update_arrays()
            self.V_EXC = np.concatenate(
                (
                    np.zeros(self.n_coarse, dtype=complex),
                    self.V_EXC_IN[n * self.n_coarse : (n + 1) * self.n_coarse],
                )
            )

            for i in range(self.n_coarse):
                self.ind = i + self.n_coarse
                self.cavity_response(self.T_s * self.omega_rf)
                self.rf_feedback(self.T_s)
                self.swap()
                self.generator_current()
                if self.excitation_otfb_1:
                    self.V_EXC_OUT[n * self.n_coarse + i] = self.V_FB_IN[
                        self.n_coarse + i
                    ]
                elif self.excitation_otfb_2:
                    self.V_EXC_OUT[n * self.n_coarse + i] = self.V_OTFB[self.ind]

    @staticmethod
    def half_detuning(imag_peak_beam_current, R_over_Q, rf_frequency, voltage):
        """Optimum detuning for half-detuning scheme

        Parameters
        ----------
        imag_peak_beam_current : float
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
        """

        return -0.25 * R_over_Q * imag_peak_beam_current / voltage * rf_frequency

    @staticmethod
    def half_detuning_power(peak_beam_current, voltage):
        """RF power consumption half-detuning scheme with optimum detuning

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
        """

        return 0.125 * peak_beam_current * voltage

    @staticmethod
    def optimum_Q_L(detuning, rf_frequency):
        """Optimum loaded Q when no real part of RF beam current is present

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
        """

        return np.fabs(0.5 * rf_frequency / detuning)

    @staticmethod
    def optimum_Q_L_beam(R_over_Q, real_peak_beam_current, voltage):
        """Optimum loaded Q when a real part of RF beam current is present

        Parameters
        ----------
        real_peak_beam_current : float
            Peak RF beam current
        R_over_Q : float
            Cavity R/Q
        voltage : float
            Cavity voltage

        Returns
        -------
        float
            Optimum loaded Q
        """

        return voltage / (R_over_Q * real_peak_beam_current)
