# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Parent class to develop cavity feedback models and various cavity loops for the CERN machines**

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy.signal
from scipy.interpolate import interp1d

from .impulse_response import (  # noqa
    SPS3Section200MHzTWC,
    SPS4Section200MHzTWC,
    SPS5Section200MHzTWC,
)

# Import SPS3Section and feedforward_filter for eval(..) below
# noqa statement is used to block autoformatter from
# removing seemingly unused import statements
from .impulse_response import (  # noqa
    SPS3Section200MHzTWC,
    SPS4Section200MHzTWC,
    SPS5Section200MHzTWC,
)
from .impulse_response import cavity_response_sparse_matrix
from .signal_processing import (
    cartesian_to_polar,
    comb_filter,
    modulator,
    moving_average,
    get_power_gen_i,
    polar_to_cartesian,
    rf_beam_current,
    fir_filter_lhc_otfb_coeff,
    smooth_step,
)

# Import SPS3Section and feedforward_filter for eval(..) below
# noqa statement is used to block autoformatter from
# removing seemingly unused import statements
from .signal_processing import (  # noqa
    feedforward_filter_TWC3,
    feedforward_filter_TWC4,
    feedforward_filter_TWC5,
)
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional, Any

    from numpy import float64
    from numpy.typing import NDArray as NumpyArray

    from ..input_parameters.rf_parameters import RFStation
    from ..beam.profile import Profile


class CavityFeedback:
    r"""Parent class for implementing cavity feedback models interfacing with BLonD

    Parameters
    ----------
    rf_station : class
        An RFStation type class
    profile : class
        A Profile type class
    n_cavities : int
        Number of cavities the feedback is acting with
    n_s : int
        The number of RF periods the coarse grid sampling period corresponds to
    n_h : int
        The index of the harmonic in the RFStation that the feedback should act on
    """

    def __init__(
        self,
        rf_station: RFStation,
        profile: Profile,
        n_cavities: int,
        n_s: int,
        n_h: int,
        lpf: bool = False,
    ):
        # BLonD classes the feedback should get information from
        self.rf_station: RFStation = rf_station
        self.profile: Profile = profile
        self.counter = self.rf_station.counter[0]

        # Number of cavities the feedback is working on
        self.n_cavities = int(n_cavities)
        if self.n_cavities < 1:
            raise RuntimeError(
                "ERROR in CavityFeedback: argument"
                " n_cavities has invalid value!"
            )

        # Apply a low-pass filter to the RF beam current
        self.lpf = lpf

        # The harmonic index the cavity feedback is working on
        self.n_h = int(n_h)
        if self.n_h > self.rf_station.n_rf - 1:
            raise RuntimeError(
                "ERROR in CavityFeedback: argument"
                " n_h is greater than the number of n_rf in RFStation"
            )

        # Sampling time in the model and the number of samples per turn
        self.n_s = int(n_s)
        self.T_s = (
            self.n_s
            * 2
            * np.pi
            / self.rf_station.omega_rf[self.n_h, self.rf_station.counter[0]]
        )
        self.n_coarse = round(self.rf_station.t_rev[0] / self.T_s)
        self.omega_carrier = (
            self.rf_station.omega_rf[self.n_h, self.rf_station.counter[0]]
            / self.n_s
        )
        self.omega_rf = self.rf_station.omega_rf[
            self.n_h, self.rf_station.counter[0]
        ]
        self.dT = 0

        # The least amount of arrays needed to feedback to the tracker object
        self.rf_centers = (
            np.arange(self.n_coarse) * self.T_s
            + 0.5 * self.rf_station.t_rf[self.n_h, self.rf_station.counter[0]]
        )
        self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_BEAM_FINE = np.zeros(self.profile.n_slices, dtype=complex)
        self.V_ANT_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_ANT_FINE = np.zeros(self.profile.n_slices, dtype=complex)
        self.I_GEN_COARSE = np.zeros(2 * self.n_coarse, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_slices, dtype=complex)

    @property
    def rfstation(self):
        from warnings import warn

        warn(
            "rfstation is deprecated, use ring",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rf_station

    @rfstation.setter
    def rfstation(self, val):
        from warnings import warn

        warn(
            "rfstation is deprecated, use ring",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rf_station = val

    @abstractmethod
    def circuit_track(self, no_beam: bool = False):
        r"""Method to track circuit of the feedback. This is meant to be implemented in the child class by the user.
        The only requirement for this method is that it has to update the V_ANT_FINE and V_SET arrays turn-by-turn.
        """
        pass

    @abstractmethod
    def update_fb_variables(self):
        r"""Method to update the variables specific to the feedback.
        This is meant to be implemented in the child class by the user."""
        pass

    def track_no_beam(self, n_pretrack: Optional[int] = None):
        r"""Tracking method of the cavity feedback without beam in the accelerator"""

        self.update_rf_variables()
        self.update_fb_variables()
        if n_pretrack is None:
            self.circuit_track(no_beam=True)
        else:
            for i in range(n_pretrack):
                self.circuit_track(no_beam=True)

    def track(self):
        r"""Tracking method of the cavity feedback"""
        # Update parameters from rest of BLonD classes
        self.update_rf_variables()
        self.update_fb_variables()

        # Get rf beam current
        self.rf_beam_current(lpf=self.lpf)

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.V_corr, self.alpha_sum = cartesian_to_polar(
            self.V_ANT_FINE[-self.profile.n_slices :]
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf_station.voltage[
            self.n_h, self.rf_station.counter[0]
        ]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.profile.bin_centers,
                self.rf_centers,
                self.V_SET[-self.n_coarse :],
            )
        )

    def rf_beam_current(self, lpf: bool = False):
        r"""Calculate RF beam current from beam profile"""

        # Beam current from profile
        self.I_BEAM_COARSE[: self.n_coarse] = self.I_BEAM_COARSE[
            -self.n_coarse :
        ]
        self.I_BEAM_FINE, self.I_BEAM_COARSE[-self.n_coarse :] = (
            rf_beam_current(
                self.profile,
                self.omega_rf,
                self.rf_station.t_rev[self.counter],
                lpf=lpf,
                downsample={"Ts": self.T_s, "points": self.n_coarse},
                external_reference=True,
                dT=self.dT,
            )
        )

        # Convert RF beam currents to be in units of Amperes
        self.I_BEAM_FINE = self.I_BEAM_FINE / self.profile.bin_size
        self.I_BEAM_COARSE[-self.n_coarse :] = (
            self.I_BEAM_COARSE[-self.n_coarse :] / self.T_s
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""Computes the setpoint in I/Q based on the RF voltage in the RFStation"""

        V_set = polar_to_cartesian(
            self.rf_station.voltage[self.n_h, self.counter] / self.n_cavities,
            0,
        )

        return V_set * np.ones(self.n_coarse)

    def update_rf_variables(self):
        r"""Updating variables from the other BLonD classes"""

        # Present time step
        self.counter = self.rf_station.counter[0]

        # Present RF angular frequency
        self.omega_rf = self.rf_station.omega_rf[self.n_h, self.counter]

        # Present carrier frequency: main RF frequency
        self.omega_carrier_prev = self.omega_carrier
        self.omega_carrier = self.omega_rf / self.n_s

        # Present sampling time
        self.T_s_prev = self.T_s
        self.T_s = self.n_s * 2 * np.pi / self.omega_rf

        # Update the coarse grid sampling
        self.n_coarse = round(
            self.rf_station.t_rev[self.rf_station.counter[0]] / self.T_s
        )

        # Present coarse grid and save previous turn coarse grid
        self.rf_centers_prev = np.copy(self.rf_centers)

        # Residual part of last turn entering the current turn due to non-integer harmonic number
        self.dT = (
            -self.rf_station.phi_rf[self.n_h, self.counter] / self.omega_rf
        )
        self.rf_centers = (
            np.arange(self.n_coarse) + 0.5 / self.n_s
        ) * self.T_s + self.dT


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
    V_SET : complex array
        Array set point voltage; default is False
    cpp_conv : bool
        Enable (True) or disable (False) convolutions using a C++ implementation; default is False
    pwr_clamp : bool
        Enable (True) or disable (False) power clamping; default is False
    rot_IQ : complex
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


class SPSOneTurnFeedback(CavityFeedback):
    r"""The SPS one-turn delay feedback and feedforward model in BLonD for a single cavity type.

    Parameters
    ----------
    rf_station : class
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
    Commissioning : class
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    """

    def __init__(
        self,
        rf_station: RFStation,
        profile: Profile,
        n_sections: int,
        n_cavities: int = 4,
        V_part: float = 4 / 9,
        G_ff: float = 1,
        G_llrf: float = 10,
        G_tx: float = 1,
        a_comb: float = 63 / 64,
        df: float = 0,
        commissioning: Optional[SPSCavityLoopCommissioning] = None,
        n_h: int = 0,
    ):
        super().__init__(
            rf_station=rf_station,
            profile=profile,
            n_cavities=n_cavities,
            n_s=1,
            n_h=n_h,
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
                    sys.modules[__name__],
                    "feedforward_filter_TWC" + str(n_sections),
                )
                self.n_ff = len(self.coeff_ff)  # Number of coefficients for FF
                self.n_ff_delay = round(
                    0.5 * (self.n_ff - 1) + 0.5 * self.TWC.tau / self.T_s / 5
                )

                self.logger.debug(
                    "Feed-forward delay in samples %d", self.n_ff_delay
                )

                # Multiply gain by normalisation factors from filter and
                # beam-to generator current
                self.G_ff *= self.TWC.R_beam / (
                    self.TWC.R_gen * np.sum(self.coeff_ff)
                )

        else:
            raise RuntimeError(
                "ERROR in SPSOneTurnFeedback: argument"
                " n_sections has invalid value!"
            )
        self.logger.debug(
            "SPS OTFB cavities: %d, sections: %d, voltage"
            " partition %.2f, gain: %.2e",
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
                raise RuntimeError(
                    "V_SET length should be %d" % (2 * self.n_coarse)
                )
            self.set_point = getattr(self, "set_point_mod")
        else:
            self.set_point = getattr(self, "set_point_std")
            self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)

        # Array to hold the bucket-by-bucket voltage with length LLRF
        self.DV_GEN = np.zeros(2 * self.n_coarse, dtype=complex)
        self.logger.debug(
            "Length of arrays on coarse grid 2x %d", self.n_coarse
        )

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
            self.TWC.tau
            / self.rf_station.t_rf[self.n_h, self.rf_station.counter[0]]
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
        self.V_IND_FINE_BEAM = np.zeros(self.profile.n_slices, dtype=complex)
        self.V_IND_COARSE_BEAM = np.zeros(2 * self.n_coarse, dtype=complex)

        # Initialise feed-forward; sampled every fifth bucket
        if self.open_ff == 1:
            self.logger.debug("Feed-forward active")
            self.I_BEAM_COARSE_FF = np.zeros(
                2 * self.n_coarse_ff, dtype=complex
            )
            self.I_BEAM_COARSE_FF_MOD = np.zeros(
                2 * self.n_coarse_ff, dtype=complex
            )
            self.I_FF_CORR_MOD = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_FF_CORR_DEL = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.I_FF_CORR = np.zeros(2 * self.n_coarse_ff, dtype=complex)
            self.V_FF_CORR = np.zeros(2 * self.n_coarse_ff, dtype=complex)

        # Update global cavity loop variables before tracking
        self.update_rf_variables()
        self.update_fb_variables()
        self.logger.info("Class initialized")

    def circuit_track(self, no_beam: bool = False):
        r"""Tracking the SPS CL internally."""

        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_carrier, self.rf_centers)
        self.TWC.impulse_response_beam(
            self.omega_carrier, self.profile.bin_centers, self.rf_centers
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
        self.V_ANT_COARSE[: self.n_coarse] = self.V_ANT_COARSE[
            -self.n_coarse :
        ]
        self.V_ANT_COARSE[-self.n_coarse :] = (
            self.V_IND_COARSE_GEN[-self.n_coarse :]
            + self.V_IND_COARSE_BEAM[-self.n_coarse :]
        )

        # Obtain generator-induced voltage on the fine grid by interpolation
        self.V_ANT_FINE_START = np.copy(self.V_ANT_FINE)
        self.V_ANT_FINE[: self.profile.n_slices] = self.V_ANT_FINE[
            -self.profile.n_slices :
        ]
        self.V_ANT_FINE[-self.profile.n_slices :] = self.V_IND_FINE_BEAM[
            -self.profile.n_slices :
        ] + np.interp(
            self.profile.bin_centers,
            self.rf_centers,
            self.V_IND_COARSE_GEN[-self.n_coarse :],
        )
        self.V_ANT_FINE[-self.profile.n_slices :] = (
            self.n_cavities * self.V_ANT_FINE[-self.profile.n_slices :]
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
            I_COARSE_BEAM_RESHAPED = np.copy(
                self.I_BEAM_COARSE[-self.n_coarse :]
            )
            I_COARSE_BEAM_RESHAPED = I_COARSE_BEAM_RESHAPED.reshape(
                (self.n_coarse_ff, self.n_coarse // self.n_coarse_ff)
            )
            self.I_BEAM_COARSE_FF[-self.n_coarse_ff :] = (
                np.sum(I_COARSE_BEAM_RESHAPED, axis=1) / 5
            )

            # Do a down-modulation to the resonant frequency of the TWC
            self.I_BEAM_COARSE_FF_MOD[: self.n_coarse_ff] = (
                self.I_BEAM_COARSE_FF_MOD[-self.n_coarse_ff :]
            )
            self.I_BEAM_COARSE_FF_MOD[-self.n_coarse_ff :] = modulator(
                self.I_BEAM_COARSE_FF[-self.n_coarse_ff :],
                omega_i=self.omega_carrier,
                omega_f=self.omega_c,
                T_sampling=5 * self.T_s,
                phi_0=self.dphi_mod,
                dt=self.dT,
            )

            self.I_FF_CORR[: self.n_coarse_ff] = self.I_FF_CORR[
                -self.n_coarse_ff :
            ]
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
                self.n_ff_delay
                * self.T_s
                * 5
                * (self.omega_c - self.omega_carrier)
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
            self.V_IND_FINE_BEAM[-self.profile.n_slices :] = (
                self.matr_conv(
                    self.I_BEAM_FINE[-self.profile.n_slices :], self.TWC.h_beam
                )[-self.profile.n_slices :]
                * self.profile.bin_size
            )

    # INDIVIDUAL COMPONENTS ---------------------------------------------------
    # LLRF MODEL

    def set_point_std(self):
        r"""Computes the desired set point voltage in I/Q."""

        self.logger.debug(
            "Entering %s function" % sys._getframe(0).f_code.co_name
        )
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

        self.logger.debug(
            "Entering %s function" % sys._getframe(0).f_code.co_name
        )
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
            self.DV_MOD_FR[-self.n_mov_av - self.n_coarse + 1 :], self.n_mov_av
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
        self.I_GEN_COARSE[: self.n_coarse] = self.I_GEN_COARSE[
            -self.n_coarse :
        ]
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
        self.V_IND_COARSE_GEN[: self.n_coarse] = self.V_IND_COARSE_GEN[
            -self.n_coarse :
        ]
        # Compute current turn generator-induced voltage
        self.V_IND_COARSE_GEN[-self.n_coarse :] = (
            self.matr_conv(self.I_GEN_COARSE, self.TWC.h_gen)[-self.n_coarse :]
            * self.T_s
        )

    def matr_conv(self, I: NumpyArray, h: NumpyArray) -> NumpyArray:
        r"""Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        return scipy.signal.fftconvolve(I, h, mode="full")[: I.shape[0]]

    def call_conv(self, signal, kernel):
        r"""Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1, dtype=complex)
        bm.convolve(signal, kernel, result=result, mode="full")

        return result

    def update_fb_variables(self):
        r"""Update variables in the feedback"""

        # Phase offset at the end of a 1-turn modulated signal (for demodulated, multiply by -1 as c and r reversed)
        self.phi_mod_0 = (
            (self.omega_carrier_prev - self.omega_c)
            * (self.T_s_prev * self.n_coarse)
            % (2 * np.pi)
        )
        self.dphi_mod += self.phi_mod_0
        self.dphi_mod = self.dphi_mod % (2 * np.pi)

        # Present delay time
        self.n_mov_av = int(
            self.TWC.tau / self.rf_station.t_rf[self.n_h, self.counter]
        )
        self.n_delay = self.n_coarse - self.n_mov_av

        if self.open_ff == 1:
            self.n_ff_delay = round(
                0.5 * (self.n_ff - 1)
                + 0.5
                * self.TWC.tau
                / self.rfstation.t_rf[self.n_h, self.counter]
                / 5
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
    rf_station : class
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
    Commissioning : class
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        rf_station: RFStation,
        profile: Profile,
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
        if commissioning is None:
            commissioning = SPSCavityLoopCommissioning()

        self.rf_station = rf_station

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
                rf_station=rf_station,
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
                n_h=n_h,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                rf_station=rf_station,
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
                n_h=n_h,
            )
        else:
            if not a_comb:
                a_comb = 15 / 16

            if V_part is None:
                V_part = 4 / 9
            self.OTFB_1 = SPSOneTurnFeedback(
                rf_station=rf_station,
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
                n_h=n_h,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                rf_station=rf_station,
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
                n_h=n_h,
            )

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Initialise OTFB without beam
        self.turns = int(turns)
        if turns < 1:
            # FeedbackError
            raise RuntimeError(
                "ERROR in SPSCavityFeedback: 'turns' has to"
                " be a positive integer!"
            )
        self.track_init(debug=commissioning_1.debug)

        self.logger.info("Class initialized")

    @property
    def rfstation(self):
        from warnings import warn

        warn(
            "rfstation is deprecated, use rf_station",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rf_station

    @rfstation.setter
    def rfstation(self, val):
        from warnings import warn

        warn(
            "rfstation is deprecated, use rf_station",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rf_station = val

    def track(self):
        r"""Main tracking method for the SPSCavityFeedback. This tracks both cavity types
        with beam."""

        # Track the feedbacks for the two TWC types
        self.OTFB_1.track()
        self.OTFB_2.track()

        # Sum the fine-grid antenna voltage from the TWC types
        self.V_sum = (
            self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_slices :]
            + self.OTFB_2.V_ANT_FINE[-self.OTFB_2.profile.n_slices :]
        )

        # Convert to amplitude and phase modulation
        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf_station.voltage[
            self.OTFB_1.n_h, self.rf_station.counter[0]
        ]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.OTFB_1.profile.bin_centers,
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
                    self.OTFB_1.profile.bin_centers * 1e6,
                    np.abs(
                        self.OTFB_1.V_ANT_FINE[-self.OTFB_1.profile.n_slices :]
                    ),
                    color=colors[i],
                )
                ax.plot(
                    self.OTFB_1.rf_centers * 1e6,
                    self.OTFB_1.n_cavities
                    * np.abs(
                        self.OTFB_1.V_ANT_COARSE[-self.OTFB_1.n_coarse :]
                    ),
                    color=colors[i],
                    linestyle="",
                    marker=".",
                )
            self.OTFB_2.track_no_beam()
        if debug:
            plt.show()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            self.OTFB_1.profile.bin_centers,
            self.OTFB_1.rf_centers,
            self.OTFB_1.n_cavities
            * self.OTFB_1.V_IND_COARSE_GEN[-self.OTFB_1.n_coarse :]
            + self.OTFB_2.n_cavities
            * self.OTFB_2.V_IND_COARSE_GEN[-self.OTFB_2.n_coarse :],
        )

        # Convert to amplitude and phase
        self.V_corr, self.alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf_station.voltage[
            self.OTFB_1.n_h, self.rf_station.counter[0]
        ]
        self.phi_corr = self.alpha_sum - np.angle(
            np.interp(
                self.OTFB_1.profile.bin_centers,
                self.OTFB_1.rf_centers,
                self.OTFB_1.V_SET[-self.OTFB_1.n_coarse :],
            )
        )


class LHCCavityLoop(CavityFeedback):
    r"""Cavity loop to regulate the RF voltage in the LHC ACS cavities.
    The loop contains a generator, a switch-and-protect device, an RF FB and a
    OTFB. The arrays of the LLRF system cover one turn with exactly one tenth
    of the harmonic (i.e.\ the typical sampling time is about 25 ns).

    Parameters
    ----------
    rf_station : class
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
        rf_station: RFStation,
        profile: Profile,
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
        n_h: int = 0,
    ):
        super().__init__(
            rf_station=rf_station,
            profile=profile,
            n_cavities=n_cavities,
            n_s=10,
            n_h=n_h,
        )

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
            np.sqrt(2 * self.power_thres / (self.R_over_Q * self.Q_L))
            / self.G_gen
        )
        self.excitation = self.RFFB.excitation
        self.excitation_otfb_1 = self.RFFB.excitation_otfb_1
        self.excitation_otfb_2 = self.RFFB.excitation_otfb_2

        self.logger.debug(
            "Length of arrays in generator path %d", self.n_coarse
        )

        # Initialise FIR filter for OTFB
        self.fir_n_taps = 63
        self.fir_coeff = fir_filter_lhc_otfb_coeff(n_taps=self.fir_n_taps)
        self.logger.debug(
            "Sum of FIR coefficients %.4e" % np.sum(self.fir_coeff)
        )

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

        self.V_ANT_FINE = np.zeros(self.profile.n_slices + 1, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_slices + 1, dtype=complex)

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

    def circuit_track(self, no_beam: bool = False):
        r"""Track the feedback model"""
        if not no_beam:
            self.I_BEAM_FINE *= -1j * np.exp(
                1j * (self.rf_station.phi_s[self.rf_station.counter[0]])
            )
            self.I_BEAM_COARSE[-self.n_coarse :] *= -1j * np.exp(
                1j * (self.rf_station.phi_s[self.rf_station.counter[0]])
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
                        np.array(
                            [
                                self.profile.bin_centers[0]
                                - self.profile.bin_size
                            ]
                        ),
                        self.profile.bin_centers,
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
        self.samples_fine = self.omega_rf * self.profile.bin_size

        # Find initial value of antenna voltage and generator current
        t_at_init = self.profile.bin_centers[0] - self.profile.bin_size
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
            n_samples=self.profile.n_slices,
            V_ant_init=V_A_init,
            I_gen_init=I_gen_init,
            samples_per_rf=self.samples_fine,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            detuning=self.detuning,
        )

        self.V_ANT_FINE[-self.profile.n_slices :] = (
            self.n_cavities * self.V_ANT_FINE[-self.profile.n_slices :]
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

        return (
            0.5
            * self.R_over_Q
            * self.Q_L
            * np.absolute(self.I_GEN_COARSE) ** 2
        )

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
        self.V_FIR_OUT[self.ind] = (
            self.fir_coeff[0] * self.V_OTFB_INT[self.ind]
        )
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

        self.V_SET = np.concatenate(
            (v_set_prev, self.set_point_from_rfstation())
        )

    def swap(self):
        r"""Model of the Switch and Protect module: clamping of the output
        power above a given input power."""

        # TODO: check implementation
        if self.clamping:
            self.V_SWAP_OUT[self.ind] = (
                self.v_swap_thres
                * smooth_step(
                    np.abs(self.V_FB_OUT[self.ind]),
                    x_max=self.v_swap_thres,
                    N=0,
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
            / (
                self.rf_station.voltage[self.n_h, self.counter]
                / self.n_cavities
            )
            ** 2
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
            (
                self.V_ANT_COARSE[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_FB_IN = np.concatenate(
            (
                self.V_FB_IN[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_AC_IN = np.concatenate(
            (
                self.V_AC_IN[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_AN_IN = np.concatenate(
            (
                self.V_AN_IN[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_AN_OUT = np.concatenate(
            (
                self.V_AN_OUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_DI_OUT = np.concatenate(
            (
                self.V_DI_OUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_OTFB = np.concatenate(
            (
                self.V_OTFB[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_OTFB_INT = np.concatenate(
            (
                self.V_OTFB_INT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_FIR_OUT = np.concatenate(
            (
                self.V_FIR_OUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_FB_OUT = np.concatenate(
            (
                self.V_FB_OUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.V_SWAP_OUT = np.concatenate(
            (
                self.V_SWAP_OUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.I_GEN_COARSE = np.concatenate(
            (
                self.I_GEN_COARSE[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.I_TEST = np.concatenate(
            (
                self.I_TEST[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
        )
        self.TUNER_INPUT = np.concatenate(
            (
                self.TUNER_INPUT[self.n_coarse :],
                np.zeros(self.n_coarse, dtype=complex),
            )
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

        Attributes
        ----------
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """

        self.V_EXC_IN = 1000 * self.RFFB.generate_white_noise(
            self.n_coarse * n_turns
        )
        self.V_EXC_OUT = np.zeros(self.n_coarse * n_turns, dtype=complex)
        self.V_SET = np.concatenate(
            (
                np.zeros(self.n_coarse, dtype=complex),
                self.V_EXC_IN[0 : self.n_coarse],
            )
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

        Attributes
        ----------
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """

        self.V_EXC_IN = 10000 * self.RFFB.generate_white_noise(
            self.n_coarse * n_turns
        )
        self.V_EXC_OUT = np.zeros(self.n_coarse * n_turns, dtype=complex)
        self.V_SET = np.zeros(2 * self.n_coarse, dtype=complex)
        self.V_EXC = np.concatenate(
            (
                np.zeros(self.n_coarse, dtype=complex),
                self.V_EXC_IN[0 : self.n_coarse],
            )
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
                    self.V_EXC_OUT[n * self.n_coarse + i] = self.V_OTFB[
                        self.ind
                    ]

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

        return (
            -0.25 * R_over_Q * imag_peak_beam_current / voltage * rf_frequency
        )

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
