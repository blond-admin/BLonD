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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from typing import Optional as LateInit

import numpy as np
from numpy import random as rnd
from scipy.signal import firwin

from blond import Simulation, StaticProfile
from blond.core.ring.helpers import requires
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.buffers import (
    OneTurnBufferBase,
    TwoTurnArray,
    TwoTurnBufferBase,
)
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)

from .helpers import (
    cavity_response_sparse_matrix,
    fir_filter_lhc_otfb_coeff,
    ideal_switch_and_limit,
    klystron_saturation_curve,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass


@dataclass
class LHCCavityFeedbackCoarseBuffers(TwoTurnBufferBase):
    """
    Container class for the coarse-grid signal buffers for the SPS one-turn feedback.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    v_excitation
        Buffer containing white noise for transfer function measurements.
    v_feedback_in
        Input signal [V] for the RF feedback.
    v_analog_in
        Input signal [V] for the analog feedback model.
    i_analog_out
        Output signal [A] from the analog feedback branch.
    i_digital_out
        Output signal [A] from the digital feedback branch.
    i_feedback_out
        Total output signal [A] from the RF feedback model.
    v_otfb_ac_in
        Output signal [V] from the AC coupler at the input of the one-turn delay feedback.
    v_otfb_comb
        Output signal [V] after the comb-filter of the one-turn delay feedback.
    v_otfb_fir_out
        Output signal [V] from the FIR filter of the one-turn delay feedback.
    v_otfb_out
        The output signal [V] from the one-turn delay feedback.
    i_swap_out
        The output signal [A] from the switch and protect model.
    i_gen_test
        The output signal [A] after the generator gain.
    i_gen_predrive
        The signal [A] going into the klystron saturation and bandwidth models.
    tuner_in
        Input signal for the tuner loop.
    tuner_integrated
        The tuner loop signal after the CIC-filter.
    """

    v_excitation: TwoTurnArray = field(init=False)

    # Fast RF feedback signals
    v_feedback_in: TwoTurnArray = field(init=False)
    v_analog_in: TwoTurnArray = field(init=False)
    i_analog_out: TwoTurnArray = field(init=False)
    i_digital_out: TwoTurnArray = field(init=False)
    i_feedback_out: TwoTurnArray = field(init=False)

    # OTFB signals
    v_otfb_ac_in: TwoTurnArray = field(init=False)
    v_otfb_comb: TwoTurnArray = field(init=False)
    v_otfb_fir_out: TwoTurnArray = field(init=False)
    v_otfb_out: TwoTurnArray = field(init=False)

    # High-power signals
    i_swap_out: TwoTurnArray = field(init=False)
    i_gen_test: TwoTurnArray = field(init=False)
    i_gen_predrive: TwoTurnArray = field(init=False)

    # Tuner signals
    tuner_in: TwoTurnArray = field(init=False)
    tuner_integrated: TwoTurnArray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        super().__post_init__()
        self.v_excitation = self._make_array(dtype=complex)

        self.v_feedback_in = self._make_array(dtype=complex)
        self.v_analog_in = self._make_array(dtype=complex)
        self.i_analog_out = self._make_array(dtype=complex)
        self.i_digital_out = self._make_array(dtype=complex)
        self.i_feedback_out = self._make_array(dtype=complex)

        self.v_otfb_ac_in = self._make_array(dtype=complex)
        self.v_otfb_comb = self._make_array(dtype=complex)
        self.v_otfb_fir_out = self._make_array(dtype=complex)
        self.v_otfb_out = self._make_array(dtype=complex)

        self.i_swap_out = self._make_array(dtype=complex)
        self.i_gen_test = self._make_array(dtype=complex)
        self.i_gen_predrive = self._make_array(dtype=complex)

        self.tuner_in = self._make_array(dtype=complex)
        self.tuner_integrated = self._make_array(dtype=complex)


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
    saturation
        Simulate the saturation effect of the LHC klystrons; default is False.
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
        saturation: bool = False,
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
        self.saturation = saturation
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


class LHCCavityFeedback(
    IQCavityFeedback[LHCCavityFeedbackCoarseBuffers, OneTurnBufferBase]
):
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

    buffer_cls_coarse = LHCCavityFeedbackCoarseBuffers
    buffer_cls_fine = OneTurnBufferBase

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
        self.q_l = q_l
        self.r_over_q = r_over_q
        self.tau_loop = tau_loop
        self.tau_otfb = tau_otfb
        self.logger.debug(f"Cavity loaded Q is {self.q_l:.0f}")

        # Import RF FB properties
        self.open_drive = self.rffb.open_drive
        self.open_drive_inv = self.rffb.open_drive_inv
        self.open_loop = self.rffb.open_loop
        self.open_otfb = self.rffb.open_otfb
        self.open_rffb = self.rffb.open_rffb
        self.open_tuner = self.rffb.open_tuner
        self.enable_klystron = self.rffb.enable_klystron
        self.saturation = self.rffb.saturation
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
        self.i_swap_threshold = (
            np.sqrt(2 * self.power_thres / (self.r_over_q * self.q_l))
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

        self.buffers_fine.i_gen = np.zeros(
            self.profile.n_bins + 1, dtype=complex
        )

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

        self.v_excitation_in: LateInit = None
        self.v_excitation_out: LateInit = None

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

        self.buffers_fine.i_gen = np.zeros(
            self.profile.n_bins + 1, dtype=complex
        )

        # Bandwidth of klystron
        num_taps = round(2 * self.tau_loop / self.T_s + 1)
        self.klystron_fir = firwin(
            num_taps,
            self.rffb.klystron_bandwidth,
            fs=1 / self.T_s,
            pass_zero="lowpass",
        )

        self.v_excitation_in: LateInit = None
        self.v_excitation_out: LateInit = None

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
            self.buffers_fine.i_beam *= -1j * np.exp(1j * phi_s)
            self.buffers_coarse.i_beam.curr *= -1j * np.exp(1j * phi_s)

        # Track the different parts of the model
        self.update_set_point()
        self.track_one_turn()

        if not no_beam:
            # Resample generator current to the fine-grid
            prof_hist_x = copy_to_cpu(self.profile.hist_x)
            self.buffers_fine.i_gen = np.interp(
                np.concatenate(
                    (
                        np.array([prof_hist_x[0] - self.profile.hist_step]),
                        prof_hist_x,
                    )
                ),
                self.rf_centers,
                self.buffers_coarse.i_gen.curr,
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
        self.buffers_coarse.v_ant[self.ind] = (
            self.buffers_coarse.i_gen[self.ind - 1] * self.r_over_q * samples
            + self.buffers_coarse.v_ant[self.ind - 1]
            * (1 - 0.5 * samples / self.q_l + 1j * self.detuning * samples)
            - self.buffers_coarse.i_beam[self.ind - 1]
            * 0.5
            * self.r_over_q
            * samples
        )

    def cavity_response_fine_matrix(self):
        """ACS cavity response model in matrix form on the fine-grid."""
        # Number of samples on fine grid
        self.samples_fine = self.omega_rf * self.profile.hist_step

        # Find initial value of antenna voltage and generator current
        t_at_init = float(self.profile.hist_x[0]) - self.profile.hist_step
        x_grid = np.concatenate(
            (self.rf_centers - self.T_s * self.n_coarse, self.rf_centers)
        )
        V_A_init = self._linear_interp_scalar(
            x_grid, self.buffers_coarse.v_ant.full, t_at_init
        )
        I_gen_init = self._linear_interp_scalar(
            x_grid, self.buffers_coarse.i_gen.full, t_at_init
        )

        self.buffers_fine.v_ant = cavity_response_sparse_matrix(
            i_beam=self.buffers_fine.i_beam,
            i_gen=self.buffers_fine.i_gen,
            n_samples=self.profile.n_bins,
            v_ant_init=V_A_init,
            i_gen_init=I_gen_init,
            samples_per_rf=self.samples_fine,
            r_over_q=self.r_over_q,
            q_l=self.q_l,
            detuning=self.detuning,
        )

        self.buffers_fine.v_ant[-self.profile.n_bins :] = (
            self.n_cavities * self.buffers_fine.v_ant[-self.profile.n_bins :]
        )

    def generator_current(self):
        """Calculate generator response."""
        # From V_swap_out in closed loop, constant in open loop
        # TODO: missing terms for changing voltage and beam current
        self.buffers_coarse.i_gen_test[self.ind] = (
            self.G_gen * self.buffers_coarse.i_swap_out[self.ind]
        )
        self.buffers_coarse.i_gen_predrive[self.ind] = (
            self.open_drive * self.buffers_coarse.i_gen_test[self.ind]
            + self.open_drive_inv * self.I_gen_offset
        )

        if self.saturation:
            self.buffers_coarse.i_gen_predrive[self.ind] = (
                klystron_saturation_curve(
                    predrive=np.abs(
                        self.buffers_coarse.i_gen_predrive[self.ind]
                    ),
                    zero_gain_current=self.i_swap_threshold,
                    maximum_current=None,
                    onset=0.8 * self.i_swap_threshold,
                )
                * np.exp(
                    1j * np.angle(self.buffers_coarse.i_gen_predrive[self.ind])
                )
            )

        # FIR filter
        if self.enable_klystron:
            window = self.buffers_coarse.i_gen_predrive.get_window(
                self.ind, len(self.klystron_fir)
            )[::-1]
            self.buffers_coarse.i_gen[self.ind] = np.dot(
                self.klystron_fir, window
            )
        else:
            self.buffers_coarse.i_gen[self.ind] = (
                self.buffers_coarse.i_gen_predrive[self.ind - self.n_delay]
            )

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
            * self.r_over_q
            * self.q_l
            * np.absolute(self.buffers_coarse.i_gen.full) ** 2
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
        self.buffers_coarse.v_otfb_comb[self.ind] = (
            self.alpha
            * self.buffers_coarse.v_otfb_comb[self.ind - self.n_coarse]
            + self.G_o
            * (1 - self.alpha)
            * self.buffers_coarse.v_otfb_ac_in[
                self.ind - self.n_coarse + self.n_otfb
            ]
        )

        # FIR filter
        window = self.buffers_coarse.v_otfb_comb.get_window(
            self.ind, self.fir_n_taps
        )[::-1]
        self.buffers_coarse.v_otfb_fir_out[self.ind] = np.dot(
            self.fir_coeff, window
        )

        # AC coupling at output
        self.buffers_coarse.v_otfb_out[self.ind] = (
            (1 - t_s / self.tau_o)
            * self.buffers_coarse.v_otfb_out[self.ind - 1]
            + self.buffers_coarse.v_otfb_fir_out[self.ind]
            - self.buffers_coarse.v_otfb_fir_out[self.ind - 1]
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
            self.buffers_coarse.v_feedback_in[self.ind] = (
                self.buffers_coarse.v_setpoint[self.ind]
                - self.open_loop * self.buffers_coarse.v_ant[self.ind]
            )
        else:
            self.buffers_coarse.v_feedback_in[self.ind] = (
                self.buffers_coarse.v_setpoint[self.ind]
                - self.open_loop * self.buffers_coarse.v_ant[self.ind]
            )

        # On the analog branch, OTFB can contribute
        self.buffers_coarse.v_otfb_ac_in[self.ind] = (
            (1 - t_s / self.tau_o)
            * self.buffers_coarse.v_otfb_ac_in[self.ind - 1]
            + self.buffers_coarse.v_feedback_in[self.ind]
            - self.buffers_coarse.v_feedback_in[self.ind - 1]
        )
        self.one_turn_feedback(t_s=t_s)

        self.buffers_coarse.v_analog_in[self.ind] = (
            self.buffers_coarse.v_feedback_in[self.ind]
            + self.open_otfb * self.buffers_coarse.v_otfb_out[self.ind]
            + int(bool(self.excitation_otfb))
            * self.buffers_coarse.v_excitation[self.ind]
        )

        # Output of analog feedback (separate branch)
        self.buffers_coarse.i_analog_out[self.ind] = (
            self.buffers_coarse.i_analog_out[self.ind - 1]
            * (1 - t_s / self.tau_a)
            + self.G_a
            * (
                self.buffers_coarse.v_analog_in[self.ind]
                - self.buffers_coarse.v_analog_in[self.ind - 1]
            )
        )

        # Output of digital feedback (separate branch)
        self.buffers_coarse.i_digital_out[self.ind] = (
            self.buffers_coarse.i_digital_out[self.ind - 1]
            * (1 - t_s / self.tau_d)
            + t_s
            / self.tau_d
            * self.G_a
            * self.G_d
            * np.exp(1j * self.d_phi_ad)
            * self.buffers_coarse.v_feedback_in[self.ind - 1]
        )

        # Total output: sum of analog and digital feedback
        self.buffers_coarse.i_feedback_out[self.ind] = self.open_rffb * (
            self.buffers_coarse.i_analog_out[self.ind]
            + self.buffers_coarse.i_digital_out[self.ind]
        )

    def update_set_point(self):
        """Update the set point for the next turn based on the design RF voltage."""
        coeff = np.polyfit(
            [0, self.n_coarse + 1],
            [
                self.buffers_coarse.v_setpoint.prev[-1],
                self.set_point_from_rfstation()[0],
            ],
            1,
        )
        poly = np.poly1d(coeff)
        v_set_prev = poly(np.linspace(0, self.n_coarse, self.n_coarse))

        self.buffers_coarse.v_setpoint.prev = v_set_prev
        self.buffers_coarse.v_setpoint.curr = self.set_point_from_rfstation()

    def swap(self):
        """Model of the Switch and Protect module: clamping of the output power above a given input power."""
        # TODO: check implementation
        if self.clamping:
            self.buffers_coarse.i_swap_out[self.ind] = ideal_switch_and_limit(
                signal=np.abs(self.buffers_coarse.i_feedback_out[self.ind]),
                limit=self.i_swap_threshold,
            ) * np.exp(
                1j * np.angle(self.buffers_coarse.i_feedback_out[self.ind])
            )
        else:
            self.buffers_coarse.i_swap_out[self.ind] = (
                self.buffers_coarse.i_feedback_out[self.ind]
            )

    def tuner(self):
        """Model of the tuner algorithm."""
        # Compute the detuning factor for the current turn
        volt = self.get_voltage_from_parent_rf_station()
        dtune = (
            -(self.mu / 2)
            * (
                np.min(self.buffers_coarse.tuner_integrated.curr.imag)
                + np.max(self.buffers_coarse.tuner_integrated.curr.imag)
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
        self.buffers_coarse.tuner_in[self.ind] = self.buffers_coarse.i_gen[
            self.ind
        ] * np.conj(self.buffers_coarse.v_ant[self.ind])

        # Apply CIC-component
        self.buffers_coarse.tuner_integrated[self.ind] = (
            (1 / 64)
            * (
                self.buffers_coarse.tuner_in[self.ind]
                - 2 * self.buffers_coarse.tuner_in[self.ind - 8]
                + self.buffers_coarse.tuner_in[self.ind - 16]
            )
            + 2 * self.buffers_coarse.tuner_integrated[self.ind - 1]
            - self.buffers_coarse.tuner_integrated[self.ind - 2]
        )

    def track_one_turn(self):
        """Single-turn tracking, index by index."""
        for i in range(self.n_coarse):
            T_s = self.T_s
            self.ind = i
            self.cavity_response(samples=T_s * self.omega_rf)
            self.rf_feedback(t_s=T_s)
            self.swap()
            self.generator_current()
            self.tuner_input()

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
        self.buffers_coarse.v_setpoint.curr = excitation[
            turn * self.n_coarse : (turn + 1) * self.n_coarse
        ]

    def track_no_beam_excitation(self, n_turns: int):
        """
        Pre-tracking for n_turns turns, without beam.

        With excitation; setpoint from white noise. v_excitation_in
        and v_excitation_out can be used to measure the transfer function
        of the system at set point.

        Parameters
        ----------
        n_turns
            Number of turns to track.

        Notes
        -----
        v_excitation_in : complex array
            Noise being played in set point; n_coarse * n_turns elements
        v_excitation_out : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """
        self.v_excitation_in = 1000 * self.rffb.generate_white_noise(
            self.n_coarse * n_turns
        )
        self.v_excitation_out = np.zeros(
            self.n_coarse * n_turns, dtype=complex
        )

        self.buffers_coarse.v_setpoint.prev = np.zeros(
            self.n_coarse, dtype=complex
        )
        self.buffers_coarse.v_setpoint.curr = self.v_excitation_in[
            0 : self.n_coarse
        ]

        self.track_one_turn()
        self.v_excitation_out[0 : self.n_coarse] = (
            self.buffers_coarse.v_ant.curr
        )
        for n in range(1, n_turns):
            self.buffers_coarse.shift()
            self.update_set_point_excitation(self.v_excitation_in, n)
            self.track_one_turn()
            self.v_excitation_out[
                n * self.n_coarse : (n + 1) * self.n_coarse
            ] = self.buffers_coarse.v_ant.curr

    def track_no_beam_excitation_otfb(self, n_turns: int):
        """
        Pre-tracking for n_turns turns, without beam.

        With excitation; set point from white noise. v_excitation_in
        and v_excitation_out can be used to measure the transfer function
        of the system at otfb.

        Parameters
        ----------
        n_turns
            Number of turns to track.

        Notes
        -----
        v_excitation_in : complex array
            Noise being played in set point; n_coarse * n_turns elements
        v_excitation_out : complex array
            System reaction to noise (accumulated from V_ANT); n_coarse * n_turns
            elements
        """
        self.v_excitation_in = 10000 * self.rffb.generate_white_noise(
            self.n_coarse * n_turns
        )
        self.v_excitation_out = np.zeros(
            self.n_coarse * n_turns, dtype=complex
        )

        self.buffers_coarse.v_setpoint.prev = np.zeros(
            self.n_coarse, dtype=complex
        )
        self.buffers_coarse.v_setpoint.curr = np.zeros(
            self.n_coarse, dtype=complex
        )

        self.buffers_coarse.v_excitation.prev = np.zeros(
            self.n_coarse, dtype=complex
        )
        self.buffers_coarse.v_excitation.curr = self.v_excitation_in[
            0 : self.n_coarse
        ]

        self.track_one_turn()
        if self.excitation_otfb_1:
            self.v_excitation_out[: self.n_coarse] = (
                self.buffers_coarse.v_feedback_in.curr
            )
        elif self.excitation_otfb_2:
            self.v_excitation_out[: self.n_coarse] = (
                self.buffers_coarse.v_otfb_out.curr
            )

        for n in range(1, n_turns):
            self.buffers_coarse.shift()

            self.buffers_coarse.v_excitation.prev = np.zeros(
                self.n_coarse, dtype=complex
            )
            self.buffers_coarse.v_excitation.curr = self.v_excitation_in[
                n * self.n_coarse : (n + 1) * self.n_coarse
            ]

            for i in range(self.n_coarse):
                self.ind = i
                self.cavity_response(self.T_s * self.omega_rf)
                self.rf_feedback(self.T_s)
                self.swap()
                self.generator_current()
                if self.excitation_otfb_1:
                    self.v_excitation_out[n * self.n_coarse + i] = (
                        self.buffers_coarse.v_feedback_in[self.n_coarse + i]
                    )
                elif self.excitation_otfb_2:
                    self.v_excitation_out[n * self.n_coarse + i] = (
                        self.buffers_coarse.v_otfb_out[self.ind]
                    )

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

    @staticmethod
    def _linear_interp_scalar(
        x: NumpyArray, y: NumpyArray, t: float
    ) -> complex:
        """
        Evaluate a piecewise-linear interpolant of (x, y) at a single scalar t.

        The method linearly extrapolates beyond the array bounds using the edge segment's
        slope — equivalent to interp1d(x, y, fill_value="extrapolate")(t) for a
        scalar t, but without building an interpolator object.

        Parameters
        ----------
        x
            Sorted (ascending) sample locations.
        y
            Sample values corresponding to x.
        t
            The scalar location to evaluate at.

        Returns
        -------
        complex
            Interpolated (or extrapolated) value at t.
        """
        if t <= x[0]:
            i0, i1 = 0, 1
        elif t >= x[-1]:
            i0, i1 = -2, -1
        else:
            # locate the bracketing segment
            i1 = np.searchsorted(x, t)
            i0 = i1 - 1

        slope = (y[i1] - y[i0]) / (x[i1] - x[i0])
        return y[i0] + slope * (t - x[i0])
