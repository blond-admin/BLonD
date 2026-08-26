# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the SPS cavity control.

Notes
-----
Authors:
Birk Emil Karlsen-Bæck
Helga Timko
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import fftconvolve

from blond import Simulation, backend
from blond.core.ring.helpers import requires
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.accelerators.sps.helpers import (
    comb_filter,
    feedforward_filter_generator,
    get_power_from_current,
    modulator,
    moving_average,
)
from blond.physics.feedbacks.accelerators.sps.impulse_response import (  # NOQA
    SPS3Section200MHzTWC,
    SPS4Section200MHzTWC,
    SPS5Section200MHzTWC,
)
from blond.physics.feedbacks.buffers import (
    OneTurnBufferBase,
    TwoTurnArray,
    TwoTurnBufferBase,
)
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)
from blond.physics.feedbacks.helpers import (
    cartesian_to_polar,
    generate_white_noise,
)
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.physics.cavities import (
        MultiHarmonicRFStation,
        SingleHarmonicRFStation,
    )


@dataclass
class SPSOneTurnFeedbackCoarseBuffers(TwoTurnBufferBase):
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
    dv_llrf
        Buffer containing the difference between the antenna voltage and setpoint.
    noise
        Array with white noise, if it being injected.
    dv_comb_out
        Comb-filter output signal.
    dv_delayed
        Signal after the one-turn delay.
    dv_mod_fr
        LLRF signal after the modulation to the TWS central frequency.
    dv_mod_averaged
        The signal after the TWS FIR response.
    dv_mod_frf
        Signal modulated back to the RF frequency.
    v_generator_induced
        The voltage induced by the generator inside the TWS.
    v_beam_induced
        The beam-induced voltage inside the TWS.
    v_ant_start
        Buffer containing the total antenna voltage at the start of the turn.
    """

    dv_llrf: TwoTurnArray = field(init=False)
    noise: TwoTurnArray = field(init=False)

    # LLRF MODEL ARRAYS
    dv_comb_out: TwoTurnArray = field(init=False)
    dv_delayed: TwoTurnArray = field(init=False)
    dv_mod_fr: TwoTurnArray = field(init=False)
    dv_mod_averaged: TwoTurnArray = field(init=False)

    # GENERATOR MODEL ARRAYS
    dv_mod_frf: TwoTurnArray = field(init=False)
    v_generator_induced: TwoTurnArray = field(init=False)

    # BEAM MODEL ARRAYS
    # Initialize induced beam voltage coarse and fine
    v_beam_induced: TwoTurnArray = field(init=False)
    v_ant_start: TwoTurnArray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        super().__post_init__()
        self.dv_llrf = self._make_array(dtype=complex)
        self.noise = self._make_array(dtype=complex)

        self.dv_comb_out = self._make_array(dtype=complex)
        self.dv_delayed = self._make_array(dtype=complex)
        self.dv_mod_fr = self._make_array(dtype=complex)
        self.dv_mod_averaged = self._make_array(dtype=complex)
        self.dv_mod_frf = self._make_array(dtype=complex)

        self.v_generator_induced = self._make_array(dtype=complex)
        self.v_beam_induced = self._make_array(dtype=complex)

        self.v_ant_start = self._make_array(dtype=complex)


@dataclass
class SPSOneTurnFeedbackFineBuffers(OneTurnBufferBase):
    """
    Container class for the fine-grid signal buffers for the SPS one-turn feedback.

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
    v_beam_induced
        The beam-induced voltage inside the TWS.
    v_ant_start
        Buffer containing the total antenna voltage at the start of the turn.
    """

    v_beam_induced: np.ndarray = field(init=False)
    v_ant_start: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        super().__post_init__()
        self.v_beam_induced = self._make_array(dtype=complex)
        self.v_ant_start = self._make_array(dtype=complex)


@dataclass
class SPSFeedForwardCoarseBuffers(TwoTurnBufferBase):
    """
    Container class for the coarse-grid signal buffers for the SPS feedforward.

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
    i_beam_modulated
        Beam current modulated from the RF frequency to the central frequency of the TWC.
    i_ffwd_filtered
        The feedforward correction signal after applying the FFWD FIR filter.
    i_ffwd_modulated
        The feedforward correction signal modulated back to the RF frequency.
    i_ffwd_correction
        The final correction signal from the feedforward.
    """

    i_beam_modulated: TwoTurnArray = field(init=False)
    i_ffwd_filtered: TwoTurnArray = field(init=False)
    i_ffwd_modulated: TwoTurnArray = field(init=False)
    i_ffwd_correction: TwoTurnArray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        super().__post_init__()

        self.i_beam_modulated = self._make_array(dtype=complex)
        self.i_ffwd_filtered = self._make_array(dtype=complex)
        self.i_ffwd_modulated = self._make_array(dtype=complex)
        self.i_ffwd_correction = self._make_array(dtype=complex)


class SPSCavityFeedbackCommissioning:
    """
    Class containing commissioning settings for the cavity feedback.

    Parameters
    ----------
    open_loop
        Open (True) or closed (False) cavity loop; default is False.
    open_fb
        Open (True) or closed (False) feedback; default is False.
    open_drive
        Open (True) or closed (False) drive; default is False.
    open_ff
        Open (True) or closed (False) feed-forward; default is True.
    v_set
        Array set point voltage; default is False.
    cpp_conv
        Enable (True) or disable (False) convolutions using a C++ implementation; default is False.
    pwr_clamp
        Enable (True) or disable (False) power clamping; default is False.
    rot_iq
        Option to rotate the set point and beam induced voltages in the complex plane.
    excitation
        Excite the model with white noise to perform BBNA measurements.
    seed1
        Seed for the generation of the white noise.
    seed2
        Second seed for the generation of the white noise.
    """

    def __init__(
        self,
        open_loop: bool = False,
        open_fb: bool = False,
        open_drive: bool = False,
        open_ff: bool = True,
        v_set: NumpyArray | None = None,
        cpp_conv: bool = False,
        pwr_clamp: bool = False,
        rot_iq: complex = 1,
        excitation: bool = False,
        seed1: int = 1234,
        seed2: int = 5678,
    ):
        self.open_loop = 0 if open_loop else 1
        self.open_fb = 0 if open_fb else 1
        self.open_drive = 0 if open_drive else 1
        self.open_ff = 0 if open_ff else 1
        self.v_set = v_set
        self.cpp_conv = cpp_conv
        self.pwr_clamp = pwr_clamp
        self.rot_iq: complex = rot_iq
        self.excitation: int = int(excitation)
        self.seed1 = seed1
        self.seed2 = seed2


class SPSOneTurnFeedback(
    IQCavityFeedback[
        SPSOneTurnFeedbackCoarseBuffers, SPSOneTurnFeedbackFineBuffers
    ]
):
    """
    The SPS one-turn delay feedback and feedforward model in BLonD for a single cavity type.

    Parameters
    ----------
    profile
        A Profile type class.
    n_sections
        Number of sections of the traveling wave cavity.
    n_cavities
        Number of traveling wave cavities of this type; default is 4.
    v_part
        Partitioning of the total voltage onto this cavity type; default is 4/9.
    g_ff
        Feedforward gain; default is 1.
    g_llrf
        Low-level RF gain; default is 10.
    g_tx
        Transmitter gain; default is 1.
    a_comb
        Comb filter coefficient; default is 63/64.
    df
        Change of the TWC central frequency in Hz from the 2021 measurement; default is 0 Hz.
    n_pretrack
        Number of turns to pre-track without beam.
    commissioning
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    harmonic_index
        Index of the harmonic the feedback should be working on.
    """

    buffer_cls_coarse = SPSOneTurnFeedbackCoarseBuffers
    buffer_cls_fine = SPSOneTurnFeedbackFineBuffers

    def __init__(
        self,
        profile: StaticProfile,
        n_sections: int,
        n_cavities: int = 4,
        v_part: float = 4 / 9,
        g_ff: float = 1,
        g_llrf: float = 10,
        g_tx: float = 1,
        a_comb: float = 63 / 64,
        df: float = 0,
        n_pretrack: int = 1000,
        commissioning: SPSCavityFeedbackCommissioning | None = None,
        harmonic_index: int = 0,
    ):
        self.V_set: NumpyArray | None = None
        self.n_delay: int | None = None

        if commissioning is None:
            commissioning = SPSCavityFeedbackCommissioning()

        super().__init__(
            profile=profile,
            n_cavities=n_cavities,
            n_periods_coarse=1,
            harmonic_index=harmonic_index,
        )

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Commissioning options
        self.open_loop = commissioning.open_loop
        self.open_fb = commissioning.open_fb
        self.open_drive = commissioning.open_drive
        self.open_ff = commissioning.open_ff

        self.custom_setpoint = commissioning.v_set
        self.set_point_modulation = True
        if self.custom_setpoint is None:  # Vset as array or not
            self.set_point_modulation = False

        self.cpp_conv = commissioning.cpp_conv
        self.rot_iq: complex = commissioning.rot_iq
        self.excitation = commissioning.excitation
        self.excitation_seed1 = commissioning.seed1
        self.excitation_seed2 = commissioning.seed2

        self.n_sections = int(n_sections)
        self.df = df
        self.n_pretrack = n_pretrack

        self.V_part = float(v_part)
        if self.V_part * (1 - self.V_part) < 0:
            raise ValueError(
                "ERROR in SPSOneTurnFeedback: V_part should be in range (0,1)!"
            )

        # Gain settings
        self.G_ff = float(g_ff)
        self.G_llrf = float(g_llrf)
        self.G_tx = float(g_tx)
        self.a_comb = float(a_comb)

        # 200 MHz travelling wave cavity (TWC) model
        if self.n_sections in [3, 4, 5]:
            self.TWC = eval(
                "SPS"
                + str(self.n_sections)
                + "Section200MHzTWC("
                + str(self.df)
                + ")"
            )

            # TWC resonant frequency
            self.omega_c = self.TWC.omega_r
        else:
            raise ValueError(
                "ERROR in SPSOneTurnFeedback: argument n_sections has invalid value!"
            )

        self.dphi_mod = 0

        # Switch between convolution methods
        if self.cpp_conv:
            self.conv = self.call_conv
        else:
            self.conv = self.matr_conv

        self.buffers_ffwd: SPSFeedForwardCoarseBuffers | None = None
        self.phi_mod_0: Any | None = None
        self.v_excitation_in: NumpyArray | None = None
        self.v_excitation_out: NumpyArray | None = None

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
        if self.open_loop == 0:  # pragma: no cover
            self.logger.debug("Opening overall OTFB loop")
        else:
            self.logger.debug("Closing overall OTFB loop")

        if self.open_fb == 0:  # pragma: no cover
            self.logger.debug("Opening feedback of drive correction")
        else:
            self.logger.debug("Closing feedback of drive correction")

        if self.open_drive == 0:  # pragma: no cover
            self.logger.debug("Opening drive to generator")
        else:
            self.logger.debug("Closing drive to generator")

        if self.open_ff == 0:  # pragma: no cover
            self.logger.debug("Opening feed-forward on beam current")
        else:
            self.logger.debug("Closing feed-forward on beam current")

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
        super().on_run_simulation(
            simulation=simulation, beam=beam, n_turns=n_turns, **kwargs
        )

        self.setup_feedback()

        # Update global cavity loop variables before tracking
        self.update_rf_variables()
        self.update_fb_variables()

        if self.n_pretrack > 0:
            self.track_no_beam(n_pretrack=self.n_pretrack)

        self.logger.info("Class initialized")

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

        self.setup_feedback()

        # Update global cavity loop variables before tracking
        self.update_rf_variables(omega_rf=omega_rf, harmonic=harmonic)
        self.update_fb_variables()
        self.logger.info("Class initialized")

        if self.n_pretrack > 0:
            self.track_no_beam(n_pretrack=self.n_pretrack)
            if self.excitation:
                self.track_no_beam_excitation(n_turns=self.n_pretrack)

    def setup_feedback(self):
        """Method to setup the cavity feedback model."""
        # 200 MHz travelling wave cavity (TWC) model
        if self.open_ff == 1:
            # Feed-forward filter
            self.coeff_ff = feedforward_filter_generator(self.n_sections)
            self.n_ff = len(self.coeff_ff)  # Number of coefficients for FF
            self.n_ff_delay = round(
                0.5 * (self.n_ff - 1) + 0.5 * self.TWC.tau / self.T_s / 5
            )

            self.logger.debug(
                f"Feed-forward delay in samples {self.n_ff_delay}"
            )

            # Multiply gain by normalisation factors from filter and
            # beam-to generator current
            self.G_ff *= self.TWC.R_beam / (
                self.TWC.R_gen * np.sum(self.coeff_ff)
            )

        self.logger.debug(
            f"SPS OTFB cavities: {self.n_cavities}, sections: {self.n_sections}, "
            f"voltage partition {self.V_part:.2f}, gain: {self.G_tx:.2e}",
        )

        # Length of arrays in LLRF
        self.n_coarse_ff = int(self.n_coarse / 5)
        # Initialize turn-by-turn variables

        # Check array length for set point modulation
        if self.set_point_modulation:
            if self.custom_setpoint.shape[0] != 2 * self.n_coarse:
                raise RuntimeError(
                    f"V_SET length should be {(2 * self.n_coarse)}"
                )
            self.set_point = self.set_point_mod
            self.buffers_coarse.v_setpoint.prev = self.custom_setpoint[
                : self.n_coarse
            ]
            self.buffers_coarse.v_setpoint.curr = self.custom_setpoint[
                -self.n_coarse :
            ]
        else:
            self.set_point = self.set_point_std

        # Array to hold the bucket-by-bucket voltage with length LLRF
        self.logger.debug(
            f"Length of arrays on coarse grid 2x {self.n_coarse}"
        )

        # Initialize moving average
        self.n_mov_av = round(self.TWC.tau / self.T_s)
        self.logger.debug(f"Moving average over {self.n_mov_av} points")

        n_mov_av_thres = 2
        if self.n_mov_av < n_mov_av_thres:
            raise ValueError(
                "ERROR in SPSOneTurnFeedback: profile has to"
                " have at least 12.5 ns resolution!"
            )

        # Initialise feed-forward; sampled every fifth bucket
        if self.open_ff == 1:
            self.logger.debug("Feed-forward active")
            self.buffers_ffwd = SPSFeedForwardCoarseBuffers(
                samples_per_turn=self.n_coarse_ff
            )

    def circuit_track(self, no_beam: bool = False):
        """
        Method to track circuit of the feedback.

        Parameters
        ----------
        no_beam
            Optional argument to track without calculating the
            beam-induced voltage. Flag used for pre-tracking of the model.
        """
        # Update the impulse response at present carrier frequency
        self.TWC.impulse_response_gen(self.omega_carrier, self.rf_centers)
        self.TWC.impulse_response_beam(
            self.omega_carrier,
            copy_to_cpu(self.profile.hist_x),
            self.rf_centers,
        )

        if not no_beam:
            # Beam-induced voltage from beam profile
            self.beam_model()

        # On current measured (I,Q) voltage, apply LLRF model
        self.llrf_model()

        # Generator-induced voltage from generator current
        self.gen_model()

        # Sum generator- and beam-induced voltages for coarse grid
        self.buffers_coarse.v_ant_start.prev = np.copy(
            self.buffers_coarse.v_ant.prev
        )
        self.buffers_coarse.v_ant_start.curr = np.copy(
            self.buffers_coarse.v_ant.curr
        )
        self.buffers_coarse.v_ant.curr = (
            self.buffers_coarse.v_generator_induced.curr
            + self.buffers_coarse.v_beam_induced.curr
        )

        # Obtain generator-induced voltage on the fine grid by interpolation
        self.buffers_fine.v_ant_start = np.copy(self.buffers_fine.v_ant)
        self.buffers_fine.v_ant = self.buffers_fine.v_beam_induced + np.interp(
            copy_to_cpu(self.profile.hist_x),
            self.rf_centers,
            self.buffers_coarse.v_generator_induced.curr,
        )
        self.buffers_fine.v_ant = self.n_cavities * self.buffers_fine.v_ant

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
        self.v_excitation_in = 1000 * generate_white_noise(
            self.n_coarse * n_turns
        )
        self.v_excitation_out = np.zeros(
            self.n_coarse * n_turns, dtype=complex
        )

        for i in range(n_turns):
            self.buffers_coarse.noise.curr = self.v_excitation_in[
                self.n_coarse * i : self.n_coarse * (i + 1)
            ]
            self.track_no_beam()
            self.v_excitation_out[
                self.n_coarse * i : self.n_coarse * (i + 1)
            ] = self.buffers_coarse.v_generator_induced.curr

    def llrf_model(self):
        """
        The LLRF model of the SPSOneTurnFeedback.

        This function calles the functions related
        to the LLRF part of the model in the correct order.
        """
        # Track all the modules of the LLRF-part of the model
        self.set_point()
        self.error_and_gain()
        self.comb()
        self.one_turn_delay()
        self.mod_to_fr()
        self.mov_avg()

    def gen_model(self):
        """
        The Generator model of the SPSOneTurnFeedback.

        This function calles the functions related
        to the generator part of the model in the correct order.
        """
        # Track all the modules for the generator part of the model
        self.mod_to_frf()
        self.sum_and_gain()
        self.gen_response()

    def beam_model(self):
        """
        The Beam model of the SPSOneTurnFeedback.

        This function find the RF beam current from the Profile-object,
        applies the cavity response towards the beam and the feed-forward
        correction if engaged.
        """
        # Rotate the RF beam current
        self.buffers_fine.i_beam = self.rot_iq * self.buffers_fine.i_beam
        self.buffers_coarse.i_beam.curr = (
            self.rot_iq * self.buffers_coarse.i_beam.curr
        )

        # Beam-induced voltage
        self.beam_response(coarse=False)
        self.beam_response(coarse=True)

        # Feed-forward
        if self.open_ff == 1:
            # Calculate correction based on previous turn on coarse grid
            self.buffers_ffwd.shift()

            # Resample RF beam current to FF sampling frequency
            i_beam_coarse_reshaped = np.copy(self.buffers_coarse.i_beam.curr)
            i_beam_coarse_reshaped = i_beam_coarse_reshaped.reshape(
                (self.n_coarse_ff, self.n_coarse // self.n_coarse_ff)
            )
            self.buffers_ffwd.i_beam.curr = (
                np.sum(i_beam_coarse_reshaped, axis=1) / 5
            )

            # Do a down-modulation to the resonant frequency of the TWC
            self.buffers_ffwd.i_beam_modulated.curr = modulator(
                self.buffers_ffwd.i_beam.curr,
                omega_i=self.omega_carrier,
                omega_f=self.omega_c,
                t_sampling=5 * self.T_s,
                phi_0=self.dphi_mod,
                dt=self.dT,
            )

            self.buffers_ffwd.i_ffwd_filtered.curr = np.zeros(
                self.n_coarse_ff, dtype=complex
            )
            for ind in range(self.n_coarse_ff):
                for k in range(self.n_ff):
                    self.buffers_ffwd.i_ffwd_filtered.curr[ind] += (
                        self.coeff_ff[k]
                        * self.buffers_ffwd.i_beam_modulated[ind - k]
                    )

            # Do a down-modulation to the resonant frequency of the TWC
            phi_delay = (
                self.n_ff_delay
                * self.T_s
                * 5
                * (self.omega_c - self.omega_carrier)
            )
            self.buffers_ffwd.i_ffwd_modulated.curr = modulator(
                self.buffers_ffwd.i_ffwd_filtered.curr,
                omega_i=self.omega_c,
                omega_f=self.omega_carrier,
                t_sampling=5 * self.T_s,
                phi_0=-(self.dphi_mod + phi_delay),
                dt=self.dT,
            )

            # Compensate for FIR filter delay
            self.buffers_ffwd.i_ffwd_correction.curr = (
                self.buffers_ffwd.i_ffwd_modulated.full[
                    self.n_ff_delay : self.n_ff_delay - self.n_coarse_ff
                ]
            )

    # BEAM MODEL
    def beam_response(self, coarse: bool = False):
        """
        Compute the beam-induced voltage on the fine- and coarse-grid.

        This is done by convolving the RF beam current with the cavity response
        towards the beam. The voltage is multiplied by the number of cavities to find the total.

        Parameters
        ----------
        coarse
            Flag to indicate whether the calculation should be done on the coarse or fine grid.
        """
        self.logger.debug("Matrix convolution for V_ind")

        if coarse:
            self.buffers_coarse.v_beam_induced.curr = (
                self.matr_conv(
                    current=self.buffers_coarse.i_beam.full,
                    transfer_function=self.TWC.h_beam_coarse,
                )[-self.n_coarse :]
                * self.T_s
            )
        else:
            # Only convolve the slices for the current turn because the fine grid points can be less
            # than one turn in length
            self.buffers_fine.v_beam_induced = (
                self.matr_conv(
                    current=self.buffers_fine.i_beam,
                    transfer_function=self.TWC.h_beam,
                )[-self.profile.n_bins :]
                * self.profile.hist_step
            )

    # INDIVIDUAL COMPONENTS ---------------------------------------------------
    # LLRF MODEL

    def set_point_std(self):
        """Compute the desired set point voltage in I/Q."""
        self.logger.debug(
            f"Entering {sys._getframe(0).f_code.co_name} function"
        )
        # Read RF voltage from rf object
        self.V_set = self.set_point_from_rfstation()
        self.V_set = (
            self.V_part
            * self.V_set
            * np.exp(1j * (-0.5 * np.pi + np.angle(self.rot_iq)))
        )

        # Convert to array
        self.buffers_coarse.v_setpoint.curr = self.V_set

    def set_point_mod(self):
        """
        Use the custom modulate setpoint.

        This function is called instead of set_point_std if a modulated set point is used.
        That is, if the set point is non-constant over a turn with the periodicity of a turn.
        """
        self.logger.debug(
            f"Entering {sys._getframe(0).f_code.co_name} function"
        )
        pass

    def error_and_gain(self):
        """
        Compute the error signal and apply the LLRF gain.

        This function computes the difference between the set point and the antenna voltage
        and amplifies it with the LLRF gain.
        """
        # Store last turn error signal and update for current turn
        self.buffers_coarse.dv_llrf.curr = self.G_llrf * (
            self.buffers_coarse.v_setpoint.curr
            - self.open_loop
            * (
                self.buffers_coarse.v_generator_induced.curr
                + self.buffers_coarse.v_beam_induced.curr
            )
            + self.excitation * self.buffers_coarse.noise.curr
        )
        self.logger.debug(
            "In %s, average set point voltage %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.buffers_coarse.v_setpoint.curr)),
        )
        self.logger.debug(
            "In %s, average antenna voltage %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.buffers_coarse.v_ant.curr)),
        )
        self.logger.debug(
            "In %s, average voltage error %.6f MV",
            sys._getframe(0).f_code.co_name,
            1e-6 * np.mean(np.absolute(self.buffers_coarse.dv_llrf.curr)),
        )

    def comb(self):
        """Apply the comb filter to the error signal."""
        # Update present data
        self.buffers_coarse.dv_comb_out.curr = comb_filter(
            self.buffers_coarse.dv_comb_out.prev,
            self.buffers_coarse.dv_llrf.curr,
            self.a_comb,
        )

    def one_turn_delay(self):
        """
        Apply a delay to act with exactly a one-turn delay.

        This function applies the complementary delay such that the correction is applied
        with exactly the delay of one turn.
        """
        # Store last turn delayed signal and compute current turn error signal
        self.buffers_coarse.dv_delayed.curr = (
            self.buffers_coarse.dv_comb_out.full[
                self.n_coarse - self.n_delay : -self.n_delay
            ]
        )

    def mod_to_fr(self):
        """Modulate the error signal to the resonant frequency of the cavity."""
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        self.buffers_coarse.dv_mod_fr.curr = modulator(
            self.buffers_coarse.dv_delayed.curr,
            self.omega_carrier,
            self.omega_c,
            self.T_s,
            phi_0=self.dphi_mod,
            dt=self.dT,
        )

    def mov_avg(self):
        """Apply the cavity filter, modelled as a moving average, to the modulated error signal."""
        # Apply moving average filter for current turn
        self.buffers_coarse.dv_mod_averaged.curr = moving_average(
            self.buffers_coarse.dv_mod_fr.full[
                -self.n_mov_av - self.n_coarse + 1 :
            ],
            self.n_mov_av,
        )

    # GENERATOR MODEL

    def mod_to_frf(self):
        """Modulate the error signal from the resonant frequency of the cavity to the original carrier frequency."""
        # Note here that dphi_rf is already accumulated somewhere else (i.e. in the tracker).
        dphi_demod = (self.omega_c - self.omega_carrier) * self.TWC.tau
        self.buffers_coarse.dv_mod_frf.curr = self.open_fb * modulator(
            self.buffers_coarse.dv_mod_averaged.curr,
            self.omega_c,
            self.omega_carrier,
            self.T_s,
            phi_0=-(self.dphi_mod + dphi_demod),
            dt=self.dT,
        )

    def sum_and_gain(self):
        """
        Sum the correction signal with the setpoint and apply transmitter gain.

        Summing of the error signal from the LLRF-part of the model and the set point voltage.
        The generator current is then found by multiplying by the transmitter gain and R_gen. The feed-forward
        current will also be added to the generator current if enabled.
        """
        # Compute current turn generator current
        self.buffers_coarse.i_gen.curr = (
            self.buffers_coarse.dv_mod_frf.curr
            + self.open_drive * self.buffers_coarse.v_setpoint.curr
        )
        # Apply amplifier gain
        self.buffers_coarse.i_gen.curr *= self.G_tx / self.TWC.R_gen
        if self.open_ff == 1:
            self.buffers_coarse.i_gen.curr = (
                self.buffers_coarse.i_gen.curr
                + self.G_ff
                * np.interp(
                    self.rf_centers,
                    self.rf_centers[::5],
                    self.buffers_ffwd.i_ffwd_correction.curr,
                )
            )

    def gen_response(self):
        """
        Calculate the generator response via a matrix convolution.

        Generator current is convolved with cavity response towards the generator to get the
        generator-induced voltage. Multiplied by the number of cavities to find the total generator-
        induced voltage.
        """
        # Compute current turn generator-induced voltage
        self.buffers_coarse.v_generator_induced.curr = (
            self.matr_conv(self.buffers_coarse.i_gen.full, self.TWC.h_gen)[
                -self.n_coarse :
            ]
            * self.T_s
        )

    def matr_conv(
        self, current: NumpyArray, transfer_function: NumpyArray
    ) -> NumpyArray:
        """
        Convolution of beam current with impulse response.

        The calculation uses a complete matrix with off-diagonal elements.

        Parameters
        ----------
        current
            The current signal array [A].
        transfer_function
            The impulse response.

        Returns
        -------
        voltage
            Calculated voltage [V].
        """
        return fftconvolve(current, transfer_function, mode="full")[
            : current.shape[0]
        ]

    def call_conv(self, current: NumpyArray, transfer_function: NumpyArray):
        """
        Convolution of beam current with impulse response using an optimised C++ convolution.

        Parameters
        ----------
        current
            The current signal array [A].
        transfer_function
            The impulse response.

        Returns
        -------
        voltage
            Calculated voltage [V].
        """
        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(current)
        kernel = np.ascontiguousarray(transfer_function)

        result = np.zeros(len(kernel) + len(signal) - 1, dtype=complex)
        backend.specials.convolve(signal, kernel, result=result, mode="full")

        return result

    def update_fb_variables(self):
        """Update variables in the feedback."""
        t_rf = 2 * np.pi / float(self.omega_rf)

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
        """
        Method to compute the generator power.

        Returns
        -------
        rf_power
            Calculated RF power [W].
        """
        return get_power_from_current(self.buffers_coarse.i_gen.full, 50)


class SPSCavityFeedback:
    """
    Class taking into account the response of the two lengths of TWCs.

    Class determining the turn-by-turn total RF voltage and phase correction
    originating from the individual cavity feedbacks. Assumes two 4-section and
    two 5-section travelling wave cavities in the pre-LS2 scenario and four
    3-section and two 4-section cavities in the post-LS2 scenario. The voltage
    partitioning is proportional to the number of sections.

    Parameters
    ----------
    profile
        A Profile type class.
    g_ff
        FF gain [1]; if passed as a float, both 3- and 4-section (4- and
        5-section) cavities have the same G_ff in the post- (pre-)LS2
        scenario. If passed as a list, the first and second elements correspond
        to the G_ff of the 3- and 4-section (4- and 5-section) cavity
        feedback in the post- (pre-)LS2 scenario; default is 10.
    g_llrf
        LLRF Gain [1]; convention same as G_ff; default is 10.
    g_tx
        Transmitter gain [1] of the cavity feedback; convention same as G_ff;
        default is 0.5.
    a_comb
        Comb filter ratio [1]; default is 15/16.
    n_pretrack
        Number of turns to pre-track without beam.
    post_LS2
        Activates pre-LS2 scenario (False) or post-LS2 scenario (True); default
        is True.
    v_part
        Voltage partitioning of the shorter cavities; has to be in the range
        (0,1). Default is None and will result in 6/10 for the 3-section
        cavities in the post-LS2 scenario and 4/9 for the 4-section cavities in
        the pre-LS2 scenario.
    df
        Frequency difference between measured frequency and desired frequency;
        same convention as G_ff; default is 0.
    commissioning
        A SPSCavityLoopCommissioning type class; default is None. If this parameter is None, a new
        SPSCavityLoopCommissioning is used.
    harmonic_index
        Index of the harmonic the feedback should be working on.
    """

    def __init__(
        self,
        profile: StaticProfile,
        g_ff: float | list = 1,
        g_llrf: float | list = 10,
        g_tx: float | list = 0.5,
        a_comb: float | None = None,
        n_pretrack: int = 1000,
        post_LS2: bool = True,
        v_part: float | None = None,
        df: list[float] = 0,
        commissioning: list | SPSCavityFeedbackCommissioning | None = None,
        harmonic_index: int = 0,
    ):
        # Options for commissioning the feedback
        self.alpha_sum: NumpyArray | None = None
        self.V_sum: NumpyArray | None = None
        self.relative_amplitude_correction: NumpyArray | None = None
        self.phase_correction: NumpyArray | None = None
        self.profile = profile

        if commissioning is None:
            commissioning = SPSCavityFeedbackCommissioning()

        # Parse input for gains
        def to_pair(x):
            if hasattr(x, "__iter__"):
                return x[0], x[1]
            return x, x

        G_ff_1, G_ff_2 = to_pair(g_ff)
        G_llrf_1, G_llrf_2 = to_pair(g_llrf)
        G_tx_1, G_tx_2 = to_pair(g_tx)
        df_1, df_2 = to_pair(df)
        commissioning_1, commissioning_2 = to_pair(commissioning)

        # Voltage partitioning has to be a fraction
        if v_part and v_part * (1 - v_part) < 0:
            raise RuntimeError(
                "SPS cavity feedback: voltage partitioning has to be in the range (0,1)!"
            )

        # Voltage partition proportional to the number of sections
        if post_LS2:
            if not a_comb:
                a_comb = 63 / 64

            if v_part is None:
                v_part = 6 / 10
            self.OTFB_1 = SPSOneTurnFeedback(
                profile=profile,
                n_sections=3,
                n_cavities=4,
                v_part=v_part,
                g_ff=float(G_ff_1),
                g_llrf=float(G_llrf_1),
                g_tx=float(G_tx_1),
                a_comb=float(a_comb),
                df=float(df_1),
                n_pretrack=0,
                commissioning=commissioning_1,
                harmonic_index=harmonic_index,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                profile=profile,
                n_sections=4,
                n_cavities=2,
                v_part=1 - v_part,
                g_ff=float(G_ff_2),
                g_llrf=float(G_llrf_2),
                g_tx=float(G_tx_2),
                a_comb=float(a_comb),
                df=float(df_2),
                n_pretrack=0,
                commissioning=commissioning_2,
                harmonic_index=harmonic_index,
            )
        else:
            if not a_comb:
                a_comb = 15 / 16

            if v_part is None:
                v_part = 4 / 9
            self.OTFB_1 = SPSOneTurnFeedback(
                profile=profile,
                n_sections=4,
                n_cavities=2,
                v_part=v_part,
                g_ff=float(G_ff_1),
                g_llrf=float(G_llrf_1),
                g_tx=float(G_tx_1),
                a_comb=float(a_comb),
                df=float(df_1),
                n_pretrack=0,
                commissioning=commissioning_1,
                harmonic_index=harmonic_index,
            )
            self.OTFB_2 = SPSOneTurnFeedback(
                profile=profile,
                n_sections=5,
                n_cavities=2,
                v_part=1 - v_part,
                g_ff=float(G_ff_2),
                g_llrf=float(G_llrf_2),
                g_tx=float(G_tx_2),
                a_comb=float(a_comb),
                df=float(df_2),
                n_pretrack=0,
                commissioning=commissioning_2,
                harmonic_index=harmonic_index,
            )

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Initialise OTFB without beam
        self.n_pretrack = int(n_pretrack)

        self.logger.info("Class initialized")

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

    @requires(["RFStationBaseClass", "IQCavityFeedback"])
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
        self.gap_voltage_phase = np.zeros(self.OTFB_1.n_coarse)

        if self.n_pretrack < 1:
            # FeedbackError
            raise RuntimeError(
                "ERROR in SPSCavityFeedback: 'n_pretrack' has to be a positive integer!"
            )
        self.track_init()

    def set_parent_rf_station(
        self, rf_station: SingleHarmonicRFStation | MultiHarmonicRFStation
    ):
        """
        Set the parent rf station for the SPSCavityFeedback.

        Parameters
        ----------
        rf_station
            Simulation `SingleHarmonicRFStation` or `MultiHarmonicRFStation` object.
        """
        self.OTFB_1.set_parent_rf_station(rf_station)
        self.OTFB_2.set_parent_rf_station(rf_station)

    def track(self, beam: BeamBaseClass):
        """
        Main tracking method for the SPSCavityFeedback.

        This tracks both cavity types with beam.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        # Track the feedbacks for the two TWC types
        self.OTFB_1.track(beam=beam)
        self.OTFB_2.track(beam=beam)

        # Sum the fine-grid antenna voltage from the TWC types
        self.V_sum = (
            self.OTFB_1.buffers_fine.v_ant + self.OTFB_2.buffers_fine.v_ant
        )

        # Convert to amplitude and phase modulation
        self.relative_amplitude_correction, self.alpha_sum = (
            cartesian_to_polar(self.V_sum)
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_amplitude_correction /= (
            self.OTFB_1._parent_rf_station.voltage[self.OTFB_1.harmonic_index]
        )
        self.phase_correction = self.alpha_sum - np.angle(
            np.mean(self.OTFB_1.buffers_coarse.v_setpoint.curr)
        )

        cav_sum = (
            self.OTFB_1.buffers_coarse.v_ant.curr
            + self.OTFB_2.buffers_coarse.v_ant.curr
        )
        cav_sum_ref = (
            self.OTFB_1.buffers_coarse.v_setpoint.curr
            + self.OTFB_2.buffers_coarse.v_setpoint.curr
        )

        self.gap_voltage_phase = np.angle(cav_sum / cav_sum_ref)

    def track_init(self):
        """Tracking of the SPSCavityFeedback without beam."""
        profile_hist_x = copy_to_cpu(self.OTFB_1.profile.hist_x)

        for i in range(self.n_pretrack):
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            self.OTFB_1.track_no_beam()
            self.OTFB_2.track_no_beam()

        # Interpolate from the coarse mesh to the fine mesh of the beam
        self.V_sum = np.interp(
            profile_hist_x,
            self.OTFB_1.rf_centers,
            self.OTFB_1.n_cavities
            * self.OTFB_1.buffers_coarse.v_generator_induced.curr
            + self.OTFB_2.n_cavities
            * self.OTFB_2.buffers_coarse.v_generator_induced.curr,
        )

        # Convert to amplitude and phase
        self.relative_amplitude_correction, self.alpha_sum = (
            cartesian_to_polar(self.V_sum)
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_amplitude_correction /= (
            self.OTFB_1._parent_rf_station.voltage[self.OTFB_1.harmonic_index]
        )
        self.phase_correction = self.alpha_sum - np.angle(
            np.interp(
                profile_hist_x,
                self.OTFB_1.rf_centers,
                self.OTFB_1.buffers_coarse.v_setpoint.curr,
            )
        )

        self.relative_amplitude_correction = backend.array(
            self.relative_amplitude_correction
        )
        self.phase_correction = backend.array(self.phase_correction)
