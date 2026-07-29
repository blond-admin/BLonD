# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the LHC cavity control.

Notes
-----
Authors:
Birk Emil Karlsen-Bæck
Helga Timko
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from typing import Optional as LateInit

import numpy as np
from numpy import random as rnd
from scipy.interpolate import interp1d
from scipy.signal import firwin

from blond import Simulation, StaticProfile
from blond.core.ring.helpers import requires
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)

from .helpers import (
    cavity_response_sparse_matrix,
    fir_filter_lhc_otfb_coeff,
    smooth_step,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass


class LHCCavityFeedbackCommissioning:
    r"""
    RF Feedback settings for LHC ACS cavity loop.

    Parameters
    ----------
    alpha
        One-turn feedback memory parameter; default is 15/16.
    d_phi_ad
        Phase misalignment of digital FB w.r.t. analog FB [deg].
    g_a
        Analog FB gain [1].
    g_d
        Digital FB gain, w.r.t. analog gain [1].
    g_o
        One-turn feedback gain.
    tau_a
        Analog FB delay time [s].
    tau_d
        Digital FB delay time [s].
    tau_o
        AC-coupling delay time of one-turn feedback [s].
    mu
        Coefficient for the tuner algorithm determining timescale; default is -0.0001.
    power_thres
        Available RF power in the klystron; default is 300 kW.
    klystron_bandwidth
        Bandwidth of the klystron [Hz].
    open_drive
        Open (True) or closed (False) cavity loop at drive; default is False.
    open_loop
        Open (True) or closed (False) cavity loop at RFFB; default is False.
    open_otfb
        Open (true) or closed (False) one-turn feedback; default is False.
    open_rffb
        Open (True) or closed (False) RFFB; default is False.
    open_tuner
        Open (True) or closed (False) tuner control; default is False.
    clamping
        Simulate clamping (True) or not (False); default is False.
    enable_klystron
        Flag to enable or disable the group-delay and bandwidth of the klystron.
    excitation
        Perform BBNA measurement of the feedback (True); default is False.
    excitation_otfb_1
        Perform BBNA measurement of the feedback (True); default is False.
        This version injects the noise via the OTFB.
    excitation_otfb_2
        Perform BBNA measurement of the feedback (True); default is False.
        This version injects the noise via the OTFB.
    seed1
        Seed for the generation of the white noise.
    seed2
        Second seed for the generation of the white noise.
    """

    def __init__(
        self,
        alpha: float = 15 / 16,
        d_phi_ad: float = 0,
        g_a: float = 0.00001,
        g_d: float = 10,
        g_o: float = 10,
        tau_a: float = 170e-6,
        tau_d: float = 400e-6,
        tau_o: float = 110e-6,
        mu: float = -0.0001,
        power_thres: float = 300e3,
        klystron_bandwidth: float = 1.7e6,
        open_drive: bool = False,
        open_loop: bool = False,
        open_otfb: bool = False,
        open_rffb: bool = False,
        open_tuner: bool = False,
        clamping: bool = False,
        enable_klystron: bool = False,
        excitation: bool = False,
        excitation_otfb_1: bool = False,
        excitation_otfb_2: bool = False,
        seed1: Any = 1234,
        seed2: Any = 7564,
    ):
        # Import variables
        self.alpha = alpha
        self.d_phi_ad = d_phi_ad * np.pi / 180
        self.g_a = g_a
        self.g_d = g_d
        self.g_o = g_o
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
        self.klystron_bandwidth = klystron_bandwidth

        # Multiply with zeros if open == True
        self.open_drive = not open_drive
        self.open_drive_inv = not self.open_drive
        self.open_loop = not open_loop
        self.open_otfb = not open_otfb
        self.open_rffb = not open_rffb
        self.open_tuner = not open_tuner
        self.enable_klystron = enable_klystron

        self.clamping = clamping

    def generate_white_noise(self, n_points: int):
        """
        Generate white noise.

        Parameters
        ----------
        n_points
            Number of points to generate the white noise for.

        Returns
        -------
        white_noise
            Array containing the generated white noise.
        """
        r1 = rnd.default_rng(self.seed1)
        r1 = r1.uniform(low=0.0, high=1.0, size=n_points)

        r2 = rnd.default_rng(self.seed2)
        r2 = r2.uniform(low=0.0, high=1.0, size=n_points)

        return np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))


class LHCCavityFeedback(IQCavityFeedback):
    r"""
    Model of the cavity loop regulating the RF voltage in the LHC ACS cavities.

    The loop contains a generator, a switch-and-protect device, an RF FB and a
    OTFB. The arrays of the LLRF system cover one turn with exactly one tenth
    of the harmonic (i.e.\ the typical sampling time is about 25 ns).

    Parameters
    ----------
    profile
        Beam profile object.
    n_cavities
        Number of cavities per beam; default is 8.
    f_c
        Central cavity frequency [Hz]; default is 400.789e6 Hz.
    g_gen
        Overall driver chain gain [1]; default is 1.
    i_gen_offset
        Generator current offset [A]; default is 0.
    n_pretrack
        Number of turns to pre-track without beam; default is 200.
    q_l
        Cavity loaded quality factor; default is 20000.
    r_over_q
        Cavity R/Q [Ohm]; default is 45 Ohms.
    tau_loop
        Total loop delay [s]; default is 650e-9 s.
    tau_otfb
        Total loop delay as seen by OTFB [s]; default is 1472e-9 s.
    rffb
        LHCCavityLoopCommissioning type class containing RF FB gains and delays.
        If this parameter is None, a new LHCCavityLoopCommissioning is used.
    harmonic_index
        Index of the harmonic the loop is regulating on.
    """

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int = 8,
        f_c: float = 400.789e6,
        g_gen: float = 1,
        i_gen_offset: float = 0,
        n_pretrack: int = 200,
        q_l: float = 20000,
        r_over_q: float = 45,
        tau_loop: float = 650e-9,
        tau_otfb: float = 1472e-9,
        rffb: LHCCavityFeedbackCommissioning | None = None,
        harmonic_index: int = 0,
    ):
        super().__init__(
            profile=profile,
            n_cavities=n_cavities,
            n_periods_coarse=10,
            harmonic_index=harmonic_index,
        )
        # variables that are declared later
        self.samples: float | None = None
        self.n_delay: int | None = None
        self.n_fir: int | None = None
        self.n_otfb: int | None = None
        self.ind: int | None = None
        self.samples_fine: float | None = None
        self.detuning: float | None = None
        self.d_omega: float | None = None
        self.omega_c: float | None = None

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Options for commissioning the feedback
        if rffb is None:
            rffb = LHCCavityFeedbackCommissioning()

        # Import classes and parameters
        self.rffb = rffb
        self.I_gen_offset = i_gen_offset
        self.G_gen = g_gen
        self.n_pretrack = n_pretrack
        self.omega_c = 2 * np.pi * f_c
        # TODO: implement optimum loaded Q
        self.Q_L = q_l
        self.R_over_Q = r_over_q
        self.tau_loop = tau_loop
        self.tau_otfb = tau_otfb
        self.logger.debug(f"Cavity loaded Q is {self.Q_L:.0f}")

        # Import RF FB properties
        self.open_drive = self.rffb.open_drive
        self.open_drive_inv = self.rffb.open_drive_inv
        self.open_loop = self.rffb.open_loop
        self.open_otfb = self.rffb.open_otfb
        self.open_rffb = self.rffb.open_rffb
        self.open_tuner = self.rffb.open_tuner
        self.enable_klystron = self.rffb.enable_klystron
        self.clamping = self.rffb.clamping
        self.alpha = self.rffb.alpha
        self.d_phi_ad = self.rffb.d_phi_ad
        self.G_a = self.rffb.g_a
        self.G_d = self.rffb.g_d
        self.G_o = self.rffb.g_o
        self.tau_a = self.rffb.tau_a
        self.tau_d = self.rffb.tau_d
        self.tau_o = self.rffb.tau_o
        self.mu = self.rffb.mu
        self.power_thres = self.rffb.power_thres
        self.v_swap_thres = (
            np.sqrt(2 * self.power_thres / (self.R_over_Q * self.Q_L))
            / self.G_gen
        )
        self.klystron_bandwidth = self.rffb.klystron_bandwidth
        self.excitation = self.rffb.excitation
        self.excitation_otfb_1 = self.rffb.excitation_otfb_1
        self.excitation_otfb_2 = self.rffb.excitation_otfb_2

        self.disable_fine_grid = False

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        pass

    @requires(["RFStationBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(simulation, beam, n_turns, **kwargs)
        self.logger.debug(
            f"Length of arrays in generator path {self.n_coarse}"
        )

        # Initialise FIR filter for OTFB
        self.fir_n_taps = 63
        self.fir_coeff = fir_filter_lhc_otfb_coeff(n_taps=self.fir_n_taps)
        self.logger.debug(
            f"Sum of FIR coefficients {np.sum(self.fir_coeff):.4e}"
        )

        self.update_rf_variables()
        self.update_fb_variables()
        self.logger.debug(f"Relative detuning is {self.detuning:.4e}")

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
        self.I_GEN_GAIN = np.zeros(2 * self.n_coarse, dtype=complex)

        self.V_ANT_FINE = np.zeros(self.profile.n_bins + 1, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_bins + 1, dtype=complex)

        # Bandwidth of klystron
        num_taps = round(2 * self.tau_loop / self.T_s + 1)
        self.klystron_fir = firwin(
            num_taps,
            self.rffb.klystron_bandwidth,
            fs=1 / self.T_s,
            pass_zero="lowpass",
        )

        # Pre-track without beam
        self.logger.debug(f"Track without beam for {self.n_pretrack} turns")
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

    def set_hardware_commissioning(self, omega_rf: float, harmonic: int):
        """
        Method to prepare the cavity feedback model for transfer function measurements.

        This is meant to set the necessary feedback parameters to run the model
        standalone, e.g. to perform transfer function measurements.

        Parameters
        ----------
        omega_rf
            Angular frequency of the RF system.
        harmonic
            Harmonic number of the RF system.
        """
        super().set_hardware_commissioning(
            omega_rf=omega_rf, harmonic=harmonic
        )
        self.logger.debug(
            f"Length of arrays in generator path {self.n_coarse}"
        )

        # Initialise FIR filter for OTFB
        self.fir_n_taps = 63
        self.fir_coeff = fir_filter_lhc_otfb_coeff(n_taps=self.fir_n_taps)
        self.logger.debug(
            f"Sum of FIR coefficients {np.sum(self.fir_coeff):.4e}"
        )

        self.update_rf_variables(omega_rf=omega_rf, harmonic=harmonic)
        self.update_fb_variables()
        self.logger.debug(f"Relative detuning is {self.detuning:.4e}")

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
        self.I_GEN_GAIN = np.zeros(2 * self.n_coarse, dtype=complex)

        self.V_ANT_FINE = np.zeros(self.profile.n_bins, dtype=complex)
        self.I_GEN_FINE = np.zeros(self.profile.n_bins + 1, dtype=complex)

        # Bandwidth of klystron
        num_taps = round(2 * self.tau_loop / self.T_s + 1)
        self.klystron_fir = firwin(
            num_taps,
            self.rffb.klystron_bandwidth,
            fs=1 / self.T_s,
            pass_zero="lowpass",
        )

        self.V_EXC_IN: LateInit = None
        self.V_EXC_OUT: LateInit = None

        # Pre-track without beam
        self.logger.debug(f"Track without beam for {self.n_pretrack} turns")
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

        # self.xxx: LateInit = None

    def circuit_track(self, no_beam: bool = False):
        """
        Method to track circuit of the feedback.

        Parameters
        ----------
        no_beam
            Optional argument to track without calculating the
            beam-induced voltage. Flag used for pre-tracking of the model.
        """
        if not no_beam:
            phi_s = np.pi  # TODO: change with changing phi_s
            self.I_BEAM_FINE *= -1j * np.exp(1j * phi_s)
            self.I_BEAM_COARSE[-self.n_coarse :] *= -1j * np.exp(1j * phi_s)

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
                            [self.profile.hist_x[0] - self.profile.hist_step]
                        ),
                        self.profile.hist_x,
                    )
                ),
                self.rf_centers,
                self.I_GEN_COARSE[-self.n_coarse :],
            )

            if not self.disable_fine_grid:
                # Compute the fine-grid antenna voltage through solving a sparse matrix equation
                self.cavity_response_fine_matrix()

            # Apply the tuner correction
            self.tuner()

    def cavity_response(self, samples: float):
        """
        ACS cavity reponse model.

        Parameters
        ----------
        samples
            Samples per RF period.
        """
        self.V_ANT_COARSE[self.ind] = (
            self.I_GEN_COARSE[self.ind - 1] * self.R_over_Q * samples
            + self.V_ANT_COARSE[self.ind - 1]
            * (1 - 0.5 * samples / self.Q_L + 1j * self.detuning * samples)
            - self.I_BEAM_COARSE[self.ind - 1] * 0.5 * self.R_over_Q * samples
        )

    def cavity_response_fine_matrix(self):
        """ACS cavity response model in matrix form on the fine-grid."""
        # Number of samples on fine grid
        self.samples_fine = self.omega_rf * self.profile.hist_step

        # Find initial value of antenna voltage and generator current
        t_at_init = self.profile.hist_x[0] - self.profile.hist_step
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
            self.I_GEN_COARSE,
            fill_value="extrapolate",
        )(t_at_init)

        self.V_ANT_FINE = cavity_response_sparse_matrix(
            i_beam=self.I_BEAM_FINE,
            i_gen=self.I_GEN_FINE,
            n_samples=self.profile.n_bins,
            v_ant_init=V_A_init,
            i_gen_init=I_gen_init,
            samples_per_rf=self.samples_fine,
            r_over_q=self.R_over_Q,
            q_l=self.Q_L,
            detuning=self.detuning,
        )

        self.V_ANT_FINE[-self.profile.n_bins :] = (
            self.n_cavities * self.V_ANT_FINE[-self.profile.n_bins :]
        )

    def generator_current(self):
        """Calculate generator response."""
        # From V_swap_out in closed loop, constant in open loop
        # TODO: missing terms for changing voltage and beam current
        self.I_TEST[self.ind] = self.G_gen * self.V_SWAP_OUT[self.ind]
        self.I_GEN_GAIN[self.ind] = (
            self.open_drive * self.I_TEST[self.ind]
            + self.open_drive_inv * self.I_gen_offset
        )

        # FIR filter
        if self.enable_klystron:
            self.I_GEN_COARSE[self.ind] = (
                self.klystron_fir[0] * self.I_GEN_GAIN[self.ind]
            )
            for k in range(1, len(self.klystron_fir)):
                self.I_GEN_COARSE[self.ind] += (
                    self.klystron_fir[k] * self.I_GEN_GAIN[self.ind - k]
                )
        else:
            self.I_GEN_COARSE[self.ind] = self.I_GEN_GAIN[self.ind]

    def generator_power(self) -> NumpyArray:
        """
        Calculate of generator power from generator current.

        Returns
        -------
        generator_power
            RF power [W] calculated from the forward generator current.
        """
        return (
            0.5
            * self.R_over_Q
            * self.Q_L
            * np.absolute(self.I_GEN_COARSE) ** 2
        )

    def one_turn_feedback(self, t_s: float):
        """
        Apply effect of the OTFB on the analog branch.

        Parameters
        ----------
        t_s
            Sampling time on the coarse-grid.
        """
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
            (1 - t_s / self.tau_o) * self.V_OTFB[self.ind - 1]
            + self.V_FIR_OUT[self.ind]
            - self.V_FIR_OUT[self.ind - 1]
        )

    def rf_feedback(self, t_s: float):
        """
        Compute analog and digital RF feedback response.

        Parameters
        ----------
        t_s
            Sampling time on the coarse-grid.
        """
        # Calculate voltage difference to act on
        if self.enable_klystron:
            self.V_FB_IN[self.ind] = (
                self.V_SET[self.ind]
                - self.open_loop * self.V_ANT_COARSE[self.ind]
            )
        else:
            self.V_FB_IN[self.ind] = (
                self.V_SET[self.ind - self.n_delay]
                - self.open_loop * self.V_ANT_COARSE[self.ind - self.n_delay]
            )

        # On the analog branch, OTFB can contribute
        self.V_AC_IN[self.ind] = (
            (1 - t_s / self.tau_o) * self.V_AC_IN[self.ind - 1]
            + self.V_FB_IN[self.ind]
            - self.V_FB_IN[self.ind - 1]
        )
        self.one_turn_feedback(t_s=t_s)

        self.V_AN_IN[self.ind] = (
            self.V_FB_IN[self.ind]
            + self.open_otfb * self.V_OTFB[self.ind]
            + int(bool(self.excitation_otfb)) * self.V_EXC[self.ind]
        )

        # Output of analog feedback (separate branch)
        self.V_AN_OUT[self.ind] = self.V_AN_OUT[self.ind - 1] * (
            1 - t_s / self.tau_a
        ) + self.G_a * (self.V_AN_IN[self.ind] - self.V_AN_IN[self.ind - 1])

        # Output of digital feedback (separate branch)
        self.V_DI_OUT[self.ind] = (
            self.V_DI_OUT[self.ind - 1] * (1 - t_s / self.tau_d)
            + t_s
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
        """Update the set point for the next turn based on the design RF voltage."""
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
        """Model of the Switch and Protect module: clamping of the output power above a given input power."""
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
        """Model of the tuner algorithm."""
        # Compute the detuning factor for the current turn
        volt = self.get_voltage_from_parent_rf_station()
        dtune = (
            -(self.mu / 2)
            * (
                np.min(self.TUNER_INTEGRATED[-self.n_coarse :].imag)
                + np.max(self.TUNER_INTEGRATED[-self.n_coarse :].imag)
            )
            / (volt / self.n_cavities) ** 2
        )

        # Propagate the corrections to the detuning two the global parameters
        self.detuning = self.detuning + dtune * self.open_tuner
        self.d_omega = self.detuning * self.omega_c
        self.omega_c = self.omega_rf + self.d_omega

    def tuner_input(self):
        """Gather data for the detuning algorithm."""
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
        """Single-turn tracking, index by index."""
        for i in range(self.n_coarse):
            T_s = self.T_s
            self.ind = i + self.n_coarse
            self.cavity_response(samples=T_s * self.omega_rf)
            self.rf_feedback(t_s=T_s)
            self.swap()
            self.generator_current()
            self.tuner_input()

    def update_arrays(self):
        """
        Move arrays indices by one turn on the coarse grid.

        Moves the array indices by one turn (n_coarse points) from the
        present turn to prepare the next turn. All arrays except for V_SET.
        """
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
        self.I_GEN_GAIN = np.concatenate(
            (
                self.I_GEN_GAIN[self.n_coarse :],
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
        """Update counter and frequency-dependent variables in a given turn."""
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
        """
        Update the set point for the next turn based on the excitation to be injected.

        Parameters
        ----------
        excitation
            Array containing excitation noise.
        turn
            The index of the current turn.
        """
        self.V_SET = np.concatenate(
            (
                self.V_SET[self.n_coarse :],
                excitation[turn * self.n_coarse : (turn + 1) * self.n_coarse],
            )
        )

    def track_no_beam_excitation(self, n_turns: int):
        """
        Pre-tracking for n_turns turns, without beam.

        With excitation; setpoint from white noise. V_EXC_IN
        and V_EXC_OUT can be used to measure the transfer function
        of the system at set point.

        Parameters
        ----------
        n_turns
            Number of turns to track.

        Notes
        -----
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """
        self.V_EXC_IN = 1000 * self.rffb.generate_white_noise(
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
        """
        Pre-tracking for n_turns turns, without beam.

        With excitation; set point from white noise. V_EXC_IN
        and V_EXC_OUT can be used to measure the transfer function
        of the system at otfb.

        Parameters
        ----------
        n_turns
            Number of turns to track.

        Notes
        -----
        V_EXC_IN : complex array
            Noise being played in set point; n_coarse * n_turns elements
        V_EXC_OUT : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """
        self.V_EXC_IN = 10000 * self.rffb.generate_white_noise(
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
    def half_detuning(imag_peak_beam_current, r_over_q, rf_frequency, voltage):
        """
        Optimum detuning for half-detuning scheme.

        Parameters
        ----------
        imag_peak_beam_current
            Peak RF beam current.
        r_over_q
            Cavity R/Q.
        rf_frequency
            RF frequency.
        voltage
            RF voltage amplitude in the cavity.

        Returns
        -------
        float
            Optimum detuning (revolution) frequency in the half-detuning scheme.
        """
        return (
            -0.25 * r_over_q * imag_peak_beam_current / voltage * rf_frequency
        )

    @staticmethod
    def half_detuning_power(peak_beam_current, voltage):
        """
        RF power consumption half-detuning scheme with optimum detuning.

        Parameters
        ----------
        peak_beam_current
            Peak RF beam current.
        voltage
            Cavity voltage.

        Returns
        -------
        float
            Optimum detuning (revolution) frequency in the half-detuning scheme.
        """
        return 0.125 * peak_beam_current * voltage

    @staticmethod
    def optimum_Q_L(detuning, rf_frequency):
        """
        Optimum loaded Q when no real part of RF beam current is present.

        Parameters
        ----------
        detuning
            Detuning frequency.
        rf_frequency
            RF frequency.

        Returns
        -------
        float
            Optimum loaded Q.
        """
        return np.fabs(0.5 * rf_frequency / detuning)

    @staticmethod
    def optimum_Q_L_beam(r_over_q, real_peak_beam_current, voltage):
        """
        Optimum loaded Q when a real part of RF beam current is present.

        Parameters
        ----------
        r_over_q
            Cavity R/Q.
        real_peak_beam_current
            Peak RF beam current.
        voltage
            Cavity voltage.

        Returns
        -------
        float
            Optimum loaded Q.
        """
        return voltage / (r_over_q * real_peak_beam_current)
