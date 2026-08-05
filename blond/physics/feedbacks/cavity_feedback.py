# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base classes for the implementation of cavity feedbacks."""

from __future__ import annotations

# Import the module, not the name: a bare ``deque`` in the module namespace is
# documented by automodule, and on Python 3.14 (the CI doc image) autodoc fails
# to format its C-level signature, which breaks the ``-W`` doc build.
import collections
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import AltersReference, HasPropertyCache
from blond.core.helpers import int_from_float_with_warning
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.cavities import (
    RFStationBaseClass,
    SingleHarmonicRFStation,
)
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.beam_current import (
    rf_beam_current,
    rf_beam_current_partial,
)
from blond.physics.feedbacks.cavity_solvers import (
    cavity_response_sparse_matrix_second_order,
    pretrack_fill_voltage,
)
from blond.physics.feedbacks.envelope_kernel import envelope_pi_scan
from blond.physics.feedbacks.generator_regulation import (
    GeneratorRegulationMixin,
)
from blond.physics.feedbacks.helpers import cavity_response_sparse_matrix
from blond.physics.feedbacks.iq import (
    cartesian_to_polar,
    polar_to_cartesian,
)
from blond.physics.feedbacks.rf_center_grid import RFCenterGridMixin
from blond.physics.feedbacks.rf_center_segment import RFCenterSegment
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.feedbacks.generator_current_controller import (
        GeneratorCurrentController,
    )


class IQCavityFeedbackBase(LocalFeedback, HasPropertyCache):
    """
    Base class to design cavity feedbacks.

    Abstract IQ-envelope cavity feedback: it owns the beam profile, the
    coarse/fine grid arrays, the beam-current demodulation and the accessors
    onto the parent RF station. Concrete feedbacks (the muon-collider
    :class:`IQCavityFeedbackTimingClass`, and the experimental LHC/SPS loops)
    subclass it. The vocabulary is defined in the "Concepts and notation"
    section of :ref:`mucol_cavity_feedback_overview`.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on.
    n_cavities
        Number of cavities the feedback controls.
    n_rf_periods_per_coarse_grid
        Number of periods for the coarse grid.
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback.
    use_lowpass_filter
        Whether to apply a lowpass filter when calculating the beam current.
    name
        Name of the object.

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on.
    use_lowpass_filter
        Apply a low-pass filter to the RF beam current.
    harmonic_index
        The harmonic index the cavity feedback is working on.
    n_rf_periods_per_coarse_grid
        Width of one coarse-grid step in RF periods; sets the coarse sampling
        time and thereby the number of coarse samples per turn.
    """

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int,
        n_rf_periods_per_coarse_grid: int | float,
        harmonic_index: int,
        use_lowpass_filter: bool = False,
        name: str | None = None,
    ):
        assert isinstance(profile, StaticProfile), (
            "IQ cavity feedbacks require static profiles"
        )
        super().__init__(
            profile=profile,
            name=name,
        )

        # Number of cavities the feedback is working on
        assert n_cavities > 0, f"{n_cavities=}, but must be bigger 0."
        self.n_cavities = int_from_float_with_warning(
            n_cavities,
            warning_stacklevel=2,
        )

        # Apply a low-pass filter to the RF beam current
        self.use_lowpass_filter = use_lowpass_filter

        # The harmonic index the cavity feedback is working on
        self.harmonic_index = int_from_float_with_warning(
            harmonic_index,
            warning_stacklevel=2,
        )

        # Ratio between rf periods and coarse grid sampling period.
        # A value in (0, 1) is the sub-stepping mode: several coarse-grid
        # points per RF period, used to keep the forward-Euler cavity step
        # stable for low Q_L (see _check_step_sizes). It is a deliberate
        # configuration and is therefore accepted without warning.
        if n_rf_periods_per_coarse_grid <= 0:
            raise ValueError(f"{n_rf_periods_per_coarse_grid=} must be > 0.")
        # A non-integer number of *whole* RF periods (n >= 1) de-aligns the
        # coarse grid from the RF buckets and can break the coupling between
        # feedback loops, so warn about that case only.
        if (
            n_rf_periods_per_coarse_grid >= 1
            and n_rf_periods_per_coarse_grid
            != int(n_rf_periods_per_coarse_grid)
        ):
            warnings.warn(
                "n_rf_periods_per_coarse_grid is not an integer number of RF "
                "periods; coupling between loops might break",
                stacklevel=2,
            )
        self.n_rf_periods_per_coarse_grid = n_rf_periods_per_coarse_grid
        # TODO: what is the difference to the next one?

        # Update the coarse grid sampling
        self._n_samples_coarse: int | None = None

        self.beam_current_forward_coarse_grid: NumpyArray | None = None
        self.beam_current_fine_grid: NumpyArray | None = None
        self.antenna_voltage_coarse_grid: NumpyArray | None = None
        self.antenna_voltage_fine_grid: NumpyArray | None = None
        self.generator_current_coarse_grid: NumpyArray | None = None
        self.generator_current_fine_grid: NumpyArray | None = None

    @requires(["RFStationBaseClass", "BeamBaseClass"])
    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialisation function at the start of the simulation.

        All array elements are defined based on the parameters of
        the parent rf station, which at this point in time is
        already fully initialised.

        Parameters
        ----------
        simulation
            Simulation object to initialise on.
        beam
            Beam object to initialise on.
        n_turns
            Number of turns in the simulation.
        **kwargs
            Unused in this function.
        """
        self.invalidate_cache()

        # Number of *complete* coarse cells (each sampling_time_coarse wide)
        # that fit in one revolution; a partial trailing cell is dropped
        # (floor), consistent with the arange-based rf_centers of the timing
        # subclass. int() is required because np.zeros() below rejects a float
        # length on numpy >= 2.
        self._n_samples_coarse = int(
            np.floor(self.t_rev / self.sampling_time_coarse)
        )

        self.beam_current_forward_coarse_grid = np.zeros(
            self._n_samples_coarse, dtype=complex
        )
        self.beam_current_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )
        self.antenna_voltage_coarse_grid = np.zeros(
            self._n_samples_coarse, dtype=complex
        )
        self.antenna_voltage_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )
        self.generator_current_coarse_grid = np.zeros(
            self._n_samples_coarse, dtype=complex
        )
        self.generator_current_fine_grid = np.zeros(
            self.profile.n_bins, dtype=complex
        )

        self.invalidate_cache()

    @abstractmethod  # pragma: no cover
    def update_feedback_variables(self) -> None:
        r"""
        Method to update the variables specific to the feedback.

        This is meant to be implemented in the child class by the user.
        """
        pass

    def _resolve_main_harmonic(self, value):
        """
        Reduce a parent RF-station value to the tracked main harmonic.

        A :class:`SingleHarmonicRFStation` carries scalar RF quantities, while
        a multi-harmonic station carries a per-harmonic array that must be
        indexed by :attr:`harmonic_index`. Centralising the dispatch here keeps
        the RF-parameter properties (``omega_rf``, ``phi_rf`` etc.) to one line
        each and confines the ``isinstance`` check to a single place.

        Parameters
        ----------
        value
            The parent station's value: scalar for a single-harmonic station,
            per-harmonic array otherwise.

        Returns
        -------
        resolved
            The value at the tracked harmonic.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return value
        return value[self.harmonic_index]

    def get_voltage_from_parent_rf_station(self) -> float:
        """
        Convenience function to get the voltage from the parent RF station.

        Returns
        -------
        voltage
            Voltage from the parent RF station, either at harmonic_index or the only one.
        """
        return self._resolve_main_harmonic(self._parent_rf_station.voltage)

    @abstractmethod  # pragma: no cover
    # NOTE: the debug helper ``plot_antenna_voltage`` moved to the test
    # plotting module
    # ``unittests/physics/feedbacks/accelerators/mucol/plotting.py``.
    def circuit_track(self, no_beam: bool = False) -> None:
        r"""
        Method to track circuit of the feedback.

        Parameters
        ----------
        no_beam
            Beam dependant parts of the feedback can be skipped if this is True.

        Notes
        -----
        This is meant to be implemented in the child class by the user.
        The only requirement for this method is that it has to update the
        V_ANT_FINE and V_SET arrays turn-by-turn.
        """
        pass

    def track_no_beam(self, n_pretrack: int = 1) -> None:
        r"""
        Tracking method of the cavity feedback without beam in the accelerator.

        Parameters
        ----------
        n_pretrack
            Number of turns to pretrack the feedback.
        """
        self.update_feedback_variables()
        for _ in range(n_pretrack):
            self.circuit_track(no_beam=True)

    def _track(self, beam: BeamBaseClass) -> None:
        r"""
        Tracking method of the cavity feedback.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        self.invalidate_cache()
        # Update parameters from rest of BLonD classes
        self.update_feedback_variables()

        # Get rf beam current
        self.calculate_rf_beam_current(
            beam=beam,
            use_lowpass_filter=self.use_lowpass_filter,
        )

        # Tracking circuit model of feedback
        self.circuit_track()

        # Convert to amplitude and phase
        self.relative_voltage_correction, alpha_sum = cartesian_to_polar(
            IQ_vector=self.antenna_voltage_fine_grid,
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_voltage_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        self.phase_correction = alpha_sum - np.mean(
            np.angle(self.voltage_setpoint)
        )

    def calculate_rf_beam_current(
        self,
        beam: BeamBaseClass,
        use_lowpass_filter: bool = False,
    ) -> None:
        r"""
        Calculate the IQ beam current for the coarse and fine grid.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        use_lowpass_filter
            Usage of low-pass filter in the calculation of the beam current.
        """
        # Beam current from profile
        (
            self.beam_current_fine_grid,
            self.beam_current_forward_coarse_grid[-self.n_samples_coarse :],
        ) = rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.omega_carrier,
            T_rev=self.t_rev,
            use_lowpass_filter=use_lowpass_filter,
            downsample={
                "Ts": self.sampling_time_coarse,
                "points": self.n_samples_coarse,
            },
            external_reference=True,
            dT=self.residual_time_shift_from_last_turn,
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / self.sampling_time_coarse
        )

    def set_point_from_rfstation(self) -> NumpyArray:
        r"""
        Compute the setpoint in I/Q based on the RF voltage in the RFStation.

        Returns
        -------
        V_set
            Voltage setpoint in I/Q frame [V].
        """
        V_set = polar_to_cartesian(
            self.get_voltage_from_parent_rf_station() / self.n_cavities,
            0,
        )

        return V_set * np.ones(self._n_samples_coarse)

    @property
    def n_samples_coarse(self) -> int | None:
        """
        Number of complete coarse-grid cells per revolution.

        Read-only view of the private ``_n_samples_coarse`` set in
        :meth:`on_run_simulation`; ``None`` before the simulation starts.
        Exposed for observers such as
        :class:`~blond.handle_results.observables.IQCavityFeedbackObservation`.

        Returns
        -------
        n_samples_coarse
            Number of coarse-grid samples per turn, or ``None`` if the
            feedback has not been initialised yet.
        """
        return self._n_samples_coarse

    @property
    def harmonic(self) -> float:
        """
        Harmonic number of the parent cavity at harmonic_index.

        Returns
        -------
        harmonic
            Harmonic number of the parent cavity at harmonic_index.
        """
        if isinstance(self._parent_rf_station, SingleHarmonicRFStation):
            return self._parent_rf_station.get_main_harmonic()
        else:
            return self._parent_rf_station.harmonic[self.harmonic_index]

    @property
    def delta_omega_rf(self) -> float:
        """
        Frequency deviation of the main harmonic of the parent cavity at harmonic_index.

        Returns
        -------
        delta_omega_rf
            Frequency deviation of the main harmonic of the parent cavity at harmonic_index.
        """
        return self._resolve_main_harmonic(
            self._parent_rf_station.delta_omega_rf
        )

    @property
    def delta_phi_rf(self) -> float:
        """
        Accumulated RF phase slip of the parent cavity at harmonic_index.

        The parent station's kick clock: the phase slip
        ``int delta_omega_rf dt`` accumulated since the first passage
        (see ``RFStationBaseClass._update_delta_phi_rf_from_beam_feedback``).
        ``0.0`` before the first passage and whenever no RF-frequency offset
        ever acted.

        Returns
        -------
        delta_phi_rf
            Accumulated RF phase slip of the parent cavity at harmonic_index.
        """
        value = self._parent_rf_station.delta_phi_rf
        if value is None:
            return 0.0
        return self._resolve_main_harmonic(value)

    @property
    def omega_rf_design(self) -> float:
        """
        Design RF frequency of the parent cavity at harmonic_index.

        Returns
        -------
        omega_rf_design
            Design RF frequency of the parent cavity at harmonic_index.
        """
        return self._resolve_main_harmonic(
            self._parent_rf_station.omega_rf_design
        )

    @property
    def omega_rf(self) -> float:
        """
        Actual RF frequency of the parent cavity at harmonic_index.

        Returns
        -------
        omega_rf_actual
            Actual RF frequency of the parent cavity at harmonic_index.
        """
        return self._resolve_main_harmonic(self._parent_rf_station.omega_rf)

    @property
    def phi_rf(self) -> float:
        """
        Actual RF phase of the parent cavity at harmonic_index.

        Returns
        -------
        phi_rf_actual
            Actual RF phase of the parent cavity at harmonic_index.
        """
        return self._resolve_main_harmonic(self._parent_rf_station.phi_rf)

    # Names invalidated by invalidate_cache(). The listed members are plain
    # (uncached) properties today, so invalidation is a no-op for them; the
    # list documents which derived values would need invalidation if they
    # were ever converted to functools.cached_property.
    cached_props = (
        "t_rf",
        "omega_carrier",
        "sampling_time_coarse",
        "residual_time_shift_from_last_turn",
        "voltage_setpoint",
    )

    @property
    def t_rf(self) -> float:
        """
        Actual RF period of the parent cavity at harmonic_index.

        Returns
        -------
        t_rf
            Actual RF period of the parent cavity at harmonic_index.
        """
        return 1 / (self.omega_rf / (2 * np.pi))

    @property
    def omega_carrier(self) -> float:
        """
        Feedback carrier frequency.

        Returns
        -------
        omega_carrier
            Feedback carrier frequency.
        """
        return self.omega_rf / self.n_rf_periods_per_coarse_grid

    @property
    def t_rev(self) -> float:
        """
        Revolution time based on the harmonic and the design frequency.

        Returns
        -------
        t_rev
            Revolution time based on the harmonic and the design frequency.
        """
        return float((2 * np.pi * self.harmonic) / self.omega_rf_design)

    @property
    def sampling_time_coarse(self) -> float:
        """
        Sampling time on the coarse grid.

        Returns
        -------
        sampling_time_coarse
            Sampling time based on the number of periods per coarse grid and the design frequency.
        """
        return self.n_rf_periods_per_coarse_grid * 2 * np.pi / self.omega_rf

    @property
    def residual_time_shift_from_last_turn(self) -> float:
        """
        Residual time shift of the RF clock against the turn start [s].

        The parent station's actual RF phase ``phi_rf`` converted to a time,
        ``-phi_rf / omega_rf``. It is passed as the reference-frame shift
        ``dT`` to the beam-current demodulation (see
        :func:`~blond.physics.feedbacks.beam_current.rf_beam_current`), which
        applies the phase correction ``dT * omega_c``. The sign follows the
        reworked (mucol) convention; the LHC comparison path bridges to the
        blond2 convention via ``dT_index_sign`` in ``rf_beam_current``.

        Returns
        -------
        residual_time_shift_from_last_turn
            Residual time shift from the last turn to the current turn [s].
        """
        return -self.phi_rf / self.omega_rf

    @property
    def voltage_setpoint(self) -> NumpyArray:
        """
        Voltage setpoint on the coarse grid [V].

        The parent RF station's design voltage replicated over the coarse
        grid (one value per coarse sample).

        Returns
        -------
        voltage_setpoint
            Voltage setpoint on the coarse grid [V].
        """
        return (
            np.ones_like(self.antenna_voltage_coarse_grid)
            * self.get_voltage_from_parent_rf_station()
        )

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @property."""
        self._invalidate_cache(IQCavityFeedbackBase.cached_props)


class IQCavityFeedbackTimingClass(
    IQCavityFeedbackBase, RFCenterGridMixin, GeneratorRegulationMixin
):
    r"""
    Cavity feedback that tracks the antenna voltage on a coarse time grid.

    New to cavity feedback? The vocabulary used throughout (antenna voltage,
    IQ envelope, coarse vs fine grid, ``R/Q``, ``Q_L``, beam loading, kick,
    the reference clocks) is defined in the "Concepts and notation" section
    of :ref:`mucol_cavity_feedback_overview`.

    The antenna voltage is advanced on a coarse grid (the ``rf_centers``) with
    a forward-Euler discretisation of the cavity ODE; see ``cavity_response``
    and ``_check_step_sizes``.

    By default (no ``controller``) the generator current is a constant value
    (``generator_current_bias``). Passing a
    :class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController`
    instead turns it into a regulated generator current: each coarse-grid
    step the feedback forms the antenna-voltage error ``V_set - V_ant[n]``
    and lets the controller convert it into the generator current (see
    ``_update_generator_current``). All control tuning (gains, loop
    delay, klystron limit) lives on the controller.

    Parameters
    ----------
    profile
        Static profile the feedback should act on.
    R_over_Q
        Geometric shunt impedance of the cavity.
    Q_L
        Loaded quality factor of the cavity.
    generator_current_bias
        Constant generator-current bias [A]: the value the controller
        regulates around, and the generator current itself when no
        controller is attached.
    n_cavities
        Number of cavities connected to the feedback.
    initial_voltage
        Initial voltage [V].
    n_rf_periods_per_coarse_grid
        Width of one coarse-grid step, expressed in RF periods, i.e. the
        sampling period is ``n_rf_periods_per_coarse_grid * t_rf``. An integer
        ``>= 1`` places one coarse point every ``n`` RF periods (the standard
        mode). A fractional value in ``(0, 1)`` is the *sub-stepping* mode:
        several coarse points per RF period (see Notes). Default is 1.
    delta_omega
        Cavity detuning in [rad/s]. Applied to the cavity response as a
        per-step phase rotation, but *not* to the coarse-grid spacing (see
        Notes). Default is 0.
    debug
        Save debugging parameters during runtime.
    second_order_fine_grid_solver_enable
        If True, integrate the fine-grid cavity response with the second-order
        (trapezoidal / Crank-Nicolson) solver instead of the default
        first-order forward-Euler one. The second-order solver is much more
        accurate at coarse profile binning (its error scales as the bin size
        squared rather than linearly). Default is False.
    exponential_coarse_solver_enable
        If True, advance the *coarse* grid with the exact exponential
        propagator ``V_{n+1} = e^{L} V_n + src (e^{L}-1)/L`` (exact in decay
        and detuning rotation, unconditionally stable) instead of the default
        forward-Euler step. Same cost per step; removes the Euler step-size
        cap and the ``(delta_omega dt)^2`` per-step rotation error, so it is
        the accurate alternative to sub-stepping for low ``Q_L`` / large
        detuning. Reduces to the Euler update as the step shrinks. Default is
        False (forward-Euler, bit-unchanged).
    controller
        Optional generator-current controller (a
        :class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController`)
        that converts the antenna-voltage error into the generator current.
        If None, the generator current stays at the constant value
        ``generator_current``.
    voltage_setpoint
        Explicit per-cavity voltage setpoint in the IQ frame [V] used to form
        the error the controller acts on. If None, it is derived from the
        parent rf station. Must be real and positive (phase 0): the station's
        phase correction is referenced to the parent-derived setpoint at
        phase 0, so a rotated setpoint would be regulated but not applied --
        a non-real value raises ``ValueError``. Rotate ``phi_rf`` on the
        station instead.
    n_pretrack
        Feedforward cavity fill budget in turns. If given, the initial antenna
        voltage is seeded (in ``on_run_simulation``) from the constant-current
        fill of the cavity instead of the scalar ``initial_voltage``; see
        :func:`~blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`. The
        fill uses the constant ``generator_current_bias`` only -- the
        controller, if any, acts on the tracked turns after injection. Default
        None (start from ``initial_voltage``).
    injection_voltage
        Target ``|V_ant|`` [V] at injection. When set (requires ``n_pretrack``)
        the seed is the fill transient at the moment ``|V_ant|`` first reaches
        this value, i.e. the beam is injected part-way through the fill.
        Default None (seed from the fill after ``n_pretrack`` turns).

    Notes
    -----
    **Sub-stepping (** ``n_rf_periods_per_coarse_grid`` **< 1).** The
    forward-Euler step in ``cavity_response`` advances the antenna voltage by a
    decay factor ``1 - 0.5 * omega_rf * dt / Q_L`` with
    ``dt = n_rf_periods_per_coarse_grid * t_rf``, so the per-step decay is

        decay_per_step = 0.5 * omega_rf * dt / Q_L
                       = n_rf_periods_per_coarse_grid * pi / Q_L .

    This must stay below the hard cap of 1.0 -- the sign-flip boundary, where
    the Euler decay factor ``1 - decay_per_step`` turns negative and the
    discretized voltage inverts every step, which the exact factor
    ``exp(-omega_rf * dt / (2 * Q_L))``, positive for any step, never does.
    (Beyond ``decay_per_step > 2`` the factor magnitude also exceeds 1 and the
    response diverges outright; in between it inverts yet still contracts,
    which is unphysical all the same.) Ideally the decay is ``<< 1`` for
    accuracy; ``_check_step_sizes`` enforces the cap and warns above 0.1.
    Steps that must stay larger belong on the exact propagator
    (``exponential_coarse_solver_enable=True``), which integrates the decay
    exactly and is exempt from the check. For a low ``Q_L`` even a
    single RF period per step (``n = 1``) can be unstable (``decay = pi/Q_L``),
    so ``n`` is lowered below 1 to sub-divide the RF period and shrink the step
    proportionally. In this mode the coarse grid no longer re-aligns to an RF
    bucket each turn; the centres tile continuously across the turn boundary
    (see ``_generate_rf_centers``).

    **RF-frequency offset.** The coarse-grid geometry (spacing, tiling,
    residuals) *and* the beam-current demodulation carrier both stay on the
    *design* RF clock under a station RF-frequency offset ``delta_omega_rf``
    (see ``forward_tracking_omega_rf``); the offset enters only as an
    explicit phase. Concretely, the beam current is demodulated at the design
    carrier ``forward_tracking_omega_rf`` and rotated by the accumulated slip
    ``int delta_omega_rf dt`` -- the parent station's kick clock
    ``delta_phi_rf`` plus its live end-of-track tail -- carried as a constant
    ``carrier_phase_offset``. The readout applies the identical total (the
    station clock via ``phi_rf``, the tail via ``phase_correction``), so the
    inter-turn slip cancels and the demod/readout chain closes for every
    carried deposit. The only residual is the intra-window mismatch
    ``delta_omega_rf * hist_x`` between the design demodulation carrier and
    the actual RF; ``hist_x`` is the bunch-local profile time (order
    ``t_rf``, reset each turn), so this term is bounded to ~1e-6 rad and does
    not accumulate -- validated at the discretization floor against the
    retuning convolution (``test_multiturn_delta_omega_rf_*``). Everything
    reduces bit-identically to the undetuned behaviour when
    ``delta_omega_rf == 0``. Note this is the RF *frequency* offset of the
    parent station, distinct from the ``delta_omega`` constructor argument
    above (the cavity *resonance* detuning), which enters the cavity response
    as a per-step phase rotation and does not move the grid.
    """

    #: Compile the per-cell coarse-envelope recursion to a numba host kernel
    #: (see :mod:`~blond.physics.feedbacks.envelope_kernel`). The pure-Python
    #: path is kept as the byte-identical reference and the fallback for
    #: degenerate (coincident) coarse steps and klystron-limit saturation. Set
    #: ``False`` on an instance to force the reference path.
    use_numba_envelope_kernel: bool = True

    def __init__(
        self,
        profile,
        R_over_Q: float,
        Q_L: float,
        generator_current_bias: complex,
        n_cavities: int | float,
        initial_voltage: float = 30.0e6,
        n_rf_periods_per_coarse_grid: int = 1,
        delta_omega: float = 0.0,
        debug: bool = False,
        second_order_fine_grid_solver_enable: bool = False,
        exponential_coarse_solver_enable: bool = False,
        controller: GeneratorCurrentController | None = None,
        voltage_setpoint: complex | None = None,
        n_pretrack: int | None = None,
        injection_voltage: float | None = None,
    ):
        super().__init__(
            profile=profile,
            n_cavities=1,
            harmonic_index=1,
            n_rf_periods_per_coarse_grid=n_rf_periods_per_coarse_grid,
        )

        self.R_over_Q = R_over_Q
        self.Q_L = Q_L

        self.delta_omega = delta_omega
        # self._segments is the single source of truth for the per-turn coarse
        # grid; rf_centers / rf_centers_lengths are derived from it (see
        # _rebuild_grid_arrays) so the flat arrays the tracking loop indexes can
        # never desync from the segment list.
        self._segments: list[RFCenterSegment] = []
        self._rf_centers = np.zeros(0)
        self._rf_centers_lengths = np.zeros(0, dtype=int)
        self._residual_time_last_rf_centers_calculation = 0

        self._ring_circumference: float | None = None

        self._reference_altering_elements: (
            tuple[AltersReference, ...] | None
        ) = None
        self._reference_altering_elements_reverse: (
            tuple[AltersReference, ...] | None
        ) = None
        self._own_index_in_reference_list: int | None = None
        self._own_index_in_reference_list_reverse: int | None = None

        self._forward_tracking_omega_rf: float | None = None
        self._forward_tracking_time: float | None = None
        self._tracked_forward_until_element: AltersReference | None = None
        self._last_forward_tracking_freq: float | None = None
        self._residual_taps_last_rf_centers_calculation: int = 0

        self._reverse_tracking_time_array: NumpyArray | None = None
        self._reverse_tracking_omega_list: NumpyArray | None = None

        self._reference_state_until_tracked: ReferenceCoordinates | None = None
        self._reference_turn_offset: int = 0
        self._last_tracked_turn_frwrd: int = 0
        self._last_tracked_beam_state_frwrd: bool | None = None

        self._init_passage_tracking_state()

        self._phase_offset_frwrd_next: float = 0.0
        self._phase_offset_frwrd: float = 0.0

        self._last_val_ant_voltage: float = 0.0
        self._last_val_beam_current: float = 0.0
        self._last_val_generator_current: float = 0.0
        self._last_rf_centers_entry: float | None = None

        self._init_voltage = initial_voltage

        self.n_cavities = n_cavities

        self._debug = debug

        self._second_order_fine_grid_solver_enable = (
            second_order_fine_grid_solver_enable
        )
        self._exponential_coarse_solver_enable = (
            exponential_coarse_solver_enable
        )

        self._generator_current_bias = generator_current_bias

        self._beam_kick_warning_issued = False

        # --- Optional generator-current controller ---
        # When ``controller`` is None the generator current stays at the
        # constant value (pure constant-current drive). Otherwise the
        # controller converts the antenna-voltage error into the generator
        # current; see _update_generator_current. All control tuning lives on
        # the controller, not on this feedback.
        self._controller = controller
        # The station's phase correction is formed against the parent-derived
        # voltage_setpoint property, whose phase is 0 by construction; an
        # explicit setpoint with a non-zero phase would make the PI regulate
        # to a frame the phase correction does not use. Until that frame is
        # unified, only real, positive setpoints are supported.
        if voltage_setpoint is not None and (
            np.imag(voltage_setpoint) != 0.0 or np.real(voltage_setpoint) <= 0
        ):
            raise ValueError(
                f"voltage_setpoint={voltage_setpoint} must be real and "
                "positive (phase 0): the RF station's phase correction is "
                "referenced to the parent-derived setpoint at phase 0, so a "
                "rotated explicit setpoint would be regulated by the "
                "controller but not reflected in the applied kick. Rotate "
                "phi_rf on the station instead."
            )
        self._voltage_setpoint = voltage_setpoint
        self._omega_input_for_pi: float | None = None

        # --- Optional feedforward cavity pre-fill / injection matching ---
        # When n_pretrack is set, on_run_simulation seeds the initial antenna
        # voltage from the constant-current (feedforward) cavity fill instead
        # of the scalar initial_voltage; with injection_voltage the seed is the
        # fill transient at the point |V_ant| reaches that target. The PI
        # controller, if any, only acts on the tracked turns after injection.
        self.n_pretrack = n_pretrack
        self.injection_voltage = injection_voltage
        if self.injection_voltage is not None and self.n_pretrack is None:
            raise ValueError(
                "injection_voltage requires n_pretrack (the cavity fill "
                "budget in turns); set n_pretrack or drop injection_voltage."
            )

    def _init_passage_tracking_state(self) -> None:
        """
        Initialise the per-passage bookkeeping attributes.

        Groups the state written once per ``_track`` call: the
        simultaneous counter-rotating passage detection (the arrival time
        and direction of the previous ``_track`` call, plus the
        coarse-cell width of its forward grid as the coincidence
        tolerance) and the live tail of the RF-frequency-offset phase
        slip (the slip accumulated since the station kick clock's last
        end-of-track tick; ``0.0`` without an offset).
        """
        self._last_track_arrival_time: float | None = None
        self._last_track_is_counter_rotating: bool | None = None
        self._last_forward_cell_width: float | None = None
        self._carrier_slip_gap: float = 0.0
        # Running total of the multi-section grid-vs-carrier registration
        # phase ``sum_k (omega_k - omega_0) T_seg,k`` (see ``_track``).
        # Stays exactly 0.0 for a single section and without acceleration.
        self._grid_carrier_phase: float = 0.0

    @requires(["RFStationBaseClass"])
    def _check_step_sizes(self) -> None:
        """
        Check that the per-step decay is not too large.

        Checks that the per-step decay and detuning-induced phase rotation
        are small, since cavity_response() advances the antenna voltage by
        the forward-Euler factor
        ``(1 - 0.5 * omega * dt / Q_L + 1j * delta_omega * dt)``,
        which is only a good approximation of the exact propagator
        ``exp((-omega / (2 * Q_L) + 1j * delta_omega) * dt)``
        when ``omega * dt / Q_L`` and ``delta_omega * dt`` are << 1.

        Called from ``on_run_simulation`` (not ``on_init_simulation``),
        because ``omega_carrier`` reads the parent RF station's
        ``omega_rf_design``, which is only set once the station is fully
        initialised at the start of the run.
        """
        # The exponential coarse solver integrates the decay and detuning
        # terms exactly (an unconditional, exact propagator), so the
        # forward-Euler step-size caps below -- and their accuracy warnings --
        # do not apply. This is exactly the low-Q_L / large-detuning regime
        # the option exists to enable, so gating it here would defeat its
        # documented purpose. (The separate beam-kick magnitude check is
        # likewise a forward-Euler tripwire and is gated the same way inside
        # _check_beam_kick_magnitude / _check_beam_kicks: the exponential
        # propagator integrates the piecewise-constant drive -- beam included
        # -- exactly, so a large per-step beam kick is not a discretisation
        # error there either.)
        if self._exponential_coarse_solver_enable:
            return

        max_step_angle = 0.1  # rad, heuristic threshold for Euler validity
        # Hard cap on the per-step decay ``d = decay_per_step``. The
        # forward-Euler decay factor (real part, ignoring detuning) is
        # ``1 - d``:
        #   * d < 1:      factor in (0, 1)  -- ordinary contraction.
        #   * 1 < d < 2:  factor in (-1, 0) -- the sign flips every step
        #                 (unphysical), but |factor| < 1 so the voltage
        #                 still contracts (oscillatory decay).
        #   * d > 2:      |factor| > 1      -- the magnitude grows every
        #                 step: a genuinely divergent discretization.
        # The exact factor ``exp(-omega dt / (2 Q_L))`` is positive for every
        # step size, so the whole band d > 1 misrepresents the physics -- the
        # cap deliberately sits at the sign-flip boundary d = 1, not at the
        # divergence boundary d = 2: contracting while inverting every step
        # is still wrong, just not explosively so. Steps that genuinely need
        # d > 1 belong on the exact propagator
        # (``exponential_coarse_solver_enable=True``), which skips this check.
        max_step_angle_hard = 1.0

        # NB: use omega_rf, not omega_carrier. cavity_response() advances the
        # antenna voltage by ``omega_input * delta_t`` with
        # ``omega_input == omega_rf`` and ``delta_t == sampling_time_coarse``,
        # so the actual per-step decay is ``0.5 * omega_rf * dt / Q_L`` and
        # scales with n_rf_periods_per_coarse_grid. Using omega_carrier
        # (== omega_rf / n) would cancel that n-dependence to a constant 2*pi
        # and misjudge the stability of the discretization.
        omega_dt = self.omega_rf * self.sampling_time_coarse
        decay_per_step = 0.5 * omega_dt / self.Q_L
        detuning_phase_per_step = self.delta_omega * self.sampling_time_coarse

        if decay_per_step > max_step_angle_hard:
            raise ValueError(
                f"{decay_per_step=:.3g} > {max_step_angle_hard}: the "
                "forward-Euler decay factor (1 - decay_per_step) used in "
                "cavity_response() is negative, so the discretized cavity "
                "voltage inverts its sign every step -- an unphysical "
                "per-step sign inversion, since the exact decay factor "
                "exp(-omega * dt / (2 * Q_L)) is always positive. (It stays "
                "contracting, |factor| < 1, until decay_per_step exceeds 2, "
                "beyond which the response also diverges.) "
                "Increase Q_L or decrease n_rf_periods_per_coarse_grid; for "
                "steps that must stay this large, set "
                "exponential_coarse_solver_enable=True to use the exact "
                "propagator, which integrates the decay exactly and is not "
                "subject to this check."
            )
        if decay_per_step > max_step_angle:
            warnings.warn(
                f"{decay_per_step=:.3g} is not << 1: the forward-Euler "
                "approximation of the cavity decay "
                "(1 - 0.5 * omega * dt / Q_L) used in cavity_response() "
                "may be inaccurate; consider increasing Q_L or decreasing "
                "n_rf_periods_per_coarse_grid.",
                stacklevel=2,
            )
        if abs(detuning_phase_per_step) > max_step_angle_hard:
            raise ValueError(
                f"{detuning_phase_per_step=:.3g} > {max_step_angle_hard}: "
                "the forward-Euler approximation of the detuning-induced "
                "phase rotation (1 + 1j * delta_omega * dt) used in "
                "cavity_response() rotates the antenna voltage by more "
                "than one step's worth of angle per coarse-grid sample, "
                "i.e. the discretization can no longer track the cavity "
                "phase. Decrease delta_omega or "
                "n_rf_periods_per_coarse_grid."
            )
        if abs(detuning_phase_per_step) > max_step_angle:
            warnings.warn(
                f"{detuning_phase_per_step=:.3g} is not << 1: the "
                "forward-Euler approximation of the detuning-induced phase "
                "rotation (1 + 1j * delta_omega * dt) used in "
                "cavity_response() may be inaccurate; consider decreasing "
                "delta_omega or n_rf_periods_per_coarse_grid.",
                stacklevel=2,
            )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Initialisation function at the start of the simulation.

        All array elements are defined based on the parameters of
        the parent rf station, which at this point in time is
        already fully initialised.

        Parameters
        ----------
        simulation
            Simulation object to initialise on.
        beam
            Beam object to initialise on.
        n_turns
            Number of turns in the simulation.
        **kwargs
            Unused in this function.
        """
        self._reference_altering_elements = (
            simulation.ring.elements.get_elements(AltersReference)
        )

        self._ring_circumference = simulation.ring.circumference
        # Number of RF stations in the ring. The multi-section frame
        # correction in _track only applies with more than one, since it
        # compensates the *other* stations' mid-turn grid re-seeding; a
        # single station re-seeds only at its own passage (no mid-turn
        # frequency mismatch), so the correction must be a no-op there.
        self._n_rf_stations_in_ring = sum(
            isinstance(element, RFStationBaseClass)
            for element in self._reference_altering_elements
        )
        self._own_index_in_reference_list = (
            self._reference_altering_elements.index(self._parent_rf_station)
        )
        self._reference_altering_elements_reverse = (
            self._reference_altering_elements[::-1]
        )
        self._own_index_in_reference_list_reverse = (
            self._reference_altering_elements_reverse.index(
                self._parent_rf_station
            )
        )

        self._reference_state_until_tracked = deepcopy(beam.reference)
        self._phase_offset_frwrd_next = 0
        self._phase_offset_frwrd = 0

        # The parent RF station is fully initialised at this point (see
        # docstring), so the step-size sanity check can read omega_rf.
        self._check_step_sizes()

        # Feedforward cavity pre-fill: seed the initial antenna voltage from
        # the constant-current fill (optionally injection-matched), now that
        # omega_rf / t_rev are available. The PI controller, if attached, only
        # acts on the tracked turns after injection, so the fill stays a pure
        # feedforward (constant generator_current_bias) transient.
        if self.n_pretrack is not None:
            self._init_voltage = pretrack_fill_voltage(
                r_over_q=self.R_over_Q,
                q_l=self.Q_L,
                omega=self.omega_rf,
                delta_omega=self.delta_omega,
                generator_current=self._generator_current_bias,
                n_pretrack=self.n_pretrack,
                t_rev=self.t_rev,
                injection_voltage=self.injection_voltage,
            )

    def circuit_track(
        self,
        omega_input: float,
        no_beam: bool = False,
        start_index: int = 0,
        end_index: int = -1,
    ) -> None:
        """
        Advance the antenna voltage over a coarse-grid segment of rf_centers.

        Parameters
        ----------
        omega_input
            Frequency in the tracked segment.
        no_beam
            No beam in this segment.
        start_index
            Index of self._rf_centers at which to start computing the response.
        end_index
            Index of rf_centers until which to compute the response.
        """
        self._circuit_track_cells(
            omega_input=omega_input,
            no_beam=no_beam,
            start_index=start_index,
            end_index=end_index,
        )

        if not no_beam:
            init_beam_time = self.profile.cut_left
            assert init_beam_time > 0, (
                f"{init_beam_time=} has to be > 0, shift profile."
            )

            # last entry is forward length
            # TODO: check this, might be wrong
            # antenna_voltage_init = interp1d(
            #     self._rf_centers[-self._rf_centers_lengths[-1] :],
            #     self.antenna_voltage_coarse_grid[
            #         -self._rf_centers_lengths[-1] :
            #     ],
            # )(
            #     init_beam_time
            # )  # This is already interpolated between 0 and 100%
            antenna_voltage_init = self.antenna_voltage_coarse_grid[
                -self._rf_centers_lengths[-1] :
            ][0]
            # generator_current_init = interp1d(
            #     self._rf_centers[-self._rf_centers_lengths[-1] :],
            #     self.generator_current_coarse_grid[
            #         -self._rf_centers_lengths[-1] :
            #     ],
            # )(
            #     init_beam_time
            # )  # TODO: this should also be before the bunch arrival time and not interpolated

            generator_current_init = self.generator_current_coarse_grid[
                -self._rf_centers_lengths[-1] :
            ][0]

            samples_per_rf_fine_grid = omega_input * self.profile.hist_step
            # copy_to_cpu: the feedback signal processing is host-side
            # (scipy), so a GPU-backend profile grid must be brought to host.
            self.generator_current_fine_grid = np.interp(
                copy_to_cpu(self.profile.hist_x),
                self._rf_centers[-self._rf_centers_lengths[-1] :],
                self.generator_current_coarse_grid[
                    -self._rf_centers_lengths[-1] :
                ],
            )

            relative_detuning = self.delta_omega / omega_input
            self.cavity_response_fine(
                antenna_voltage_init,
                0,
                generator_current_init,
                samples_per_rf_fine_grid,
                relative_detuning=relative_detuning,
            )

    def _circuit_track_cells(
        self,
        omega_input: float,
        no_beam: bool,
        start_index: int,
        end_index: int,
    ) -> None:
        """
        Advance the coarse-grid recursion over ``[start_index, end_index)``.

        Dispatches to the compiled numba kernel
        (:func:`~blond.physics.feedbacks.envelope_kernel.envelope_pi_scan`)
        when ``use_numba_envelope_kernel`` is set, otherwise to the pure-Python
        per-cell reference. Both produce byte-identical coarse grids; the
        kernel exists only to remove the per-cell interpreter overhead.

        Parameters
        ----------
        omega_input
            Angular frequency of this segment.
        no_beam
            Whether the segment carries no beam.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.
        """
        # The optional PI update recovers the per-step sampling time from
        # ``omega_times_T_s / omega_input``; expose omega_input for it.
        self._omega_input_for_pi = omega_input
        if self.use_numba_envelope_kernel:
            self._circuit_track_cells_kernel(
                omega_input, no_beam, start_index, end_index
            )
        else:
            self._circuit_track_cells_python(
                omega_input, no_beam, start_index, end_index
            )

    def _circuit_track_cells_python(
        self,
        omega_input: float,
        no_beam: bool,
        start_index: int,
        end_index: int,
    ) -> None:
        """
        Reference per-cell coarse-grid recursion (pure Python).

        The readable reference the numba kernel mirrors, and the exact fallback
        for degenerate coincident coarse points (zero step), which the kernel
        path defers here so the skip-and-warn behaviour is preserved.

        Parameters
        ----------
        omega_input
            Angular frequency of this segment.
        no_beam
            Whether the segment carries no beam.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.
        """
        for rf_centers_idx in range(start_index, end_index):
            if rf_centers_idx == 0:
                if self._last_rf_centers_entry is None:
                    # First centre ever tracked: there is no previous centre to
                    # step from, so use the spacing to the next centre as the
                    # step proxy. That next centre must live in *this* segment,
                    # though. With fine sectioning the first (reverse) segment
                    # can hold a single centre, in which case rf_centers[idx+1]
                    # belongs to the next segment -- which under acceleration
                    # runs at a different frequency -- so the cross-boundary
                    # diff is meaningless and can even go negative (tripping the
                    # ordering assertion below). Fall back to this segment's own
                    # coarse step (n * t_rf at omega_input) in that case.
                    if rf_centers_idx + 1 < end_index:
                        delta_t = (
                            self._rf_centers[rf_centers_idx + 1]
                            - self._rf_centers[rf_centers_idx]
                        )
                    else:
                        delta_t = (
                            self.n_rf_periods_per_coarse_grid
                            * 2
                            * np.pi
                            / omega_input
                        )
                else:
                    delta_t = (
                        self._rf_centers[0]
                        + self._residual_time_last_rf_centers_calculation
                    )
            elif rf_centers_idx == start_index:
                delta_t = (
                    self._rf_centers[rf_centers_idx]
                    + self._residual_time_last_rf_centers_calculation
                )
            else:
                delta_t = (
                    self._rf_centers[rf_centers_idx]
                    - self._rf_centers[rf_centers_idx - 1]
                )
            # delta_t can come out marginally negative (a few ULPs) when a
            # coarse-grid point lands almost exactly on a turn/segment
            # boundary -- e.g. for sub-stepping ratios (n < 1) that divide the
            # turn evenly, where the carry-over residual is numerically zero.
            # That floating-point noise is not a real ordering violation, so
            # clamp it to zero (handled as a coincident point below) rather
            # than tripping the hard assertion.
            rf_period = 2 * np.pi / omega_input
            if -1e-9 * rf_period < delta_t < 0:
                delta_t = 0.0
            assert delta_t >= 0, f"{delta_t}"
            if delta_t == 0:
                warnings.warn(
                    "double taking of rf_centers value, skipping", stacklevel=1
                )
                continue
            self.cavity_response(
                omega_input * delta_t,
                coarse_grid_index_to_update=rf_centers_idx,
                relative_detuning=self.delta_omega / omega_input,
                no_beam=no_beam,
            )

    def _coarse_step_sizes(
        self,
        omega_input: float,
        start_index: int,
        end_index: int,
    ) -> NumpyArray | None:
        """
        Vectorised per-cell coarse step sizes for a segment.

        Reproduces, bit-for-bit, the per-cell ``delta_t`` of
        :meth:`_circuit_track_cells_python` -- the first-cell special cases and
        the few-ULP negative clamp included.

        Parameters
        ----------
        omega_input
            Angular frequency of this segment.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.

        Returns
        -------
        delta_t
            Per-cell step sizes [s], or ``None`` when the segment contains a
            zero (coincident) step that the reference path skips-and-warns.
        """
        n_cells = end_index - start_index
        delta_t = np.empty(n_cells, dtype=np.float64)
        if n_cells > 1:
            # Bulk cells: consecutive rf_centers differences (== the reference
            # ``else`` branch), bit-identical to the scalar subtraction.
            delta_t[1:] = np.diff(self._rf_centers[start_index:end_index])
        if start_index == 0:
            if self._last_rf_centers_entry is None:
                if start_index + 1 < end_index:
                    delta_t[0] = self._rf_centers[1] - self._rf_centers[0]
                else:
                    delta_t[0] = (
                        self.n_rf_periods_per_coarse_grid
                        * 2
                        * np.pi
                        / omega_input
                    )
            else:
                delta_t[0] = (
                    self._rf_centers[0]
                    + self._residual_time_last_rf_centers_calculation
                )
        else:
            delta_t[0] = (
                self._rf_centers[start_index]
                + self._residual_time_last_rf_centers_calculation
            )
        rf_period = 2 * np.pi / omega_input
        tiny_negative = (delta_t > -1e-9 * rf_period) & (delta_t < 0)
        delta_t[tiny_negative] = 0.0
        # Any non-positive step is degenerate/invalid: a coincident (zero) step,
        # or a genuinely-negative one that violates ordering. Defer the whole
        # segment to the reference loop, which -- processing cells in order --
        # warns-and-skips a zero step and asserts on a negative one, so its
        # warnings and assertion message are reproduced exactly rather than
        # pre-empted by a vectorised assert here.
        if not (delta_t > 0).all():
            return None
        return delta_t

    def _circuit_track_cells_kernel(
        self,
        omega_input: float,
        no_beam: bool,
        start_index: int,
        end_index: int,
    ) -> None:
        """
        Compiled coarse-grid recursion over one segment.

        Precomputes on the host the per-cell step sizes and the solver-specific
        voltage multiplier / drive weight (both state-independent), marshals the
        PI controller state into a circular buffer, and runs the sequential
        recursion in a single :func:`~blond.physics.feedbacks.envelope_kernel.\
envelope_pi_scan` call. Degenerate segments (a zero-length coarse step from
        coincident points) fall back to :meth:`_circuit_track_cells_python`.

        Parameters
        ----------
        omega_input
            Angular frequency of this segment.
        no_beam
            Whether the segment carries no beam.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.
        """
        n_cells = end_index - start_index
        if n_cells <= 0:
            return

        delta_t = self._coarse_step_sizes(omega_input, start_index, end_index)
        if delta_t is None:
            # A coincident (zero) step needs the skip-and-warn reference path.
            self._circuit_track_cells_python(
                omega_input, no_beam, start_index, end_index
            )
            return

        omega_times_dt = omega_input * delta_t
        relative_detuning = self.delta_omega / omega_input
        voltage_multiplier, drive_weight = self._kernel_step_multipliers(
            omega_times_dt, relative_detuning
        )
        beam_current = self._kernel_beam_current(
            no_beam, start_index, end_index, n_cells
        )

        if start_index == 0:
            voltage_init = complex(self._last_val_ant_voltage)
            generator_current_init = complex(self._last_val_generator_current)
        else:
            voltage_init = self.antenna_voltage_coarse_grid[start_index - 1]
            generator_current_init = self.generator_current_coarse_grid[
                start_index - 1
            ]

        controller_active = self._controller_active and not no_beam
        (
            gain_proportional,
            gain_integral,
            generator_current_bias,
            max_output,
            pi_setpoint,
            delay_buffer,
            integral,
        ) = self._kernel_controller_params(controller_active)

        voltage_out = np.empty(n_cells, dtype=np.complex128)
        # Pre-fill with the current generator grid: the inactive (no-beam /
        # constant-current) path reads it as each cell's drive current, matching
        # cavity_response reading generator_current_coarse_grid[idx-1]; the
        # active path overwrites every cell with its PI output. astype copies,
        # so the kernel never mutates the grid before the write-back below.
        generator_current_out = self.generator_current_coarse_grid[
            start_index:end_index
        ].astype(np.complex128)

        delay_head, integral, saturation_possible = envelope_pi_scan(
            voltage_multiplier,
            drive_weight,
            omega_times_dt,
            beam_current,
            voltage_out,
            generator_current_out,
            voltage_init,
            generator_current_init,
            float(self.R_over_Q),
            controller_active,
            pi_setpoint,
            float(omega_input),
            gain_proportional,
            gain_integral,
            generator_current_bias,
            delay_buffer,
            0,
            integral,
            max_output,
        )

        if saturation_possible:
            # A cell reached the klystron limit, whose numpy-magnitude clamp the
            # kernel cannot reproduce bit-for-bit. Nothing has been committed
            # yet (grids not written, controller state untouched), so rerun the
            # segment on the exact reference path and discard the kernel result.
            self._circuit_track_cells_python(
                omega_input, no_beam, start_index, end_index
            )
            return

        self.antenna_voltage_coarse_grid[start_index:end_index] = voltage_out
        # Commit the generator grid. Active: the PI outputs. Inactive: the
        # unchanged pre-filled values, i.e. a no-op vs the reference (which
        # leaves the generator grid untouched on the constant-current/no-beam
        # path). Only the controller's own state is synced when it actually ran.
        self.generator_current_coarse_grid[start_index:end_index] = (
            generator_current_out
        )
        if controller_active:
            self._store_controller_state(delay_buffer, delay_head, integral)

        if not no_beam:
            self._check_beam_kicks(
                beam_current,
                omega_times_dt,
                voltage_init,
                voltage_out,
                skip_first=(start_index == 0),
            )

    def _kernel_step_multipliers(
        self,
        omega_times_dt: NumpyArray,
        relative_detuning: float,
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Per-cell voltage multiplier and drive weight for the kernel.

        Both depend only on the step size and detuning (not the recursion
        state), so they are precomputed here on the host: ``B = 1 + L`` /
        ``W = 1`` for forward Euler, ``B = e^L`` / ``W = (e^L - 1) / L`` for
        the exponential propagator, with ``L`` the per-cell growth exponent.

        Parameters
        ----------
        omega_times_dt
            Per-cell ``omega * dt`` (strictly positive; zero steps have already
            fallen back to the reference path).
        relative_detuning
            Detuning normalised to the segment frequency
            (``delta_omega / omega``).

        Returns
        -------
        voltage_multiplier
            Per-cell voltage multiplier ``B`` (complex128).
        drive_weight
            Per-cell drive weight ``W`` (complex128).
        """
        step_exponent = (
            -0.5 * omega_times_dt / self.Q_L
            + 1j * relative_detuning * omega_times_dt
        )
        if self._exponential_coarse_solver_enable:
            voltage_multiplier = np.exp(step_exponent)
            # omega_times_dt > 0, so step_exponent != 0 and (e^L - 1) / L is
            # well defined.
            drive_weight = np.expm1(step_exponent) / step_exponent
        else:
            voltage_multiplier = 1.0 + step_exponent
            drive_weight = np.ones(
                omega_times_dt.shape[0], dtype=np.complex128
            )
        return voltage_multiplier, drive_weight

    def _kernel_beam_current(
        self,
        no_beam: bool,
        start_index: int,
        end_index: int,
        n_cells: int,
    ) -> NumpyArray:
        """
        Per-cell beam current for a kernel segment.

        Mirrors ``cavity_response``: the carried ``rf_centers`` index 0 (when
        the segment starts there) always uses ``last_val_beam_current`` -- even
        for a ``no_beam`` segment, since the reference's idx==0 branch ignores
        ``no_beam`` -- and every later cell is zero for a no-beam segment or
        reads the forward beam-current grid otherwise.

        Parameters
        ----------
        no_beam
            Whether the segment carries no beam.
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.
        n_cells
            Number of cells in the segment.

        Returns
        -------
        beam_current
            Per-cell beam current (complex128, length ``n_cells``).
        """
        beam_current = np.zeros(n_cells, dtype=np.complex128)
        if start_index == 0:
            # Carried index 0 uses last_val_beam_current unconditionally --
            # cavity_response's idx==0 branch has no no_beam guard.
            beam_current[0] = self._last_val_beam_current
        if no_beam:
            return beam_current
        forward_offset = len(self._rf_centers) - self._rf_centers_lengths[-1]
        global_indices = np.arange(start_index, end_index)
        local_start = 1 if start_index == 0 else 0
        beam_current[local_start:] = self.beam_current_forward_coarse_grid[
            global_indices[local_start:] - forward_offset
        ]
        return beam_current

    def _kernel_controller_params(
        self, controller_active: bool
    ) -> tuple[float, float, complex, float, complex, NumpyArray, complex]:
        """
        PI controller parameters and marshalled state for the kernel.

        Extracts the controller gains, bias, klystron limit (``inf`` when
        unlimited), IQ setpoint, delay-line buffer and error integral. When the
        controller is inactive, returns neutral values so the kernel simply
        holds the generator current.

        Parameters
        ----------
        controller_active
            Whether the PI controller runs on this segment.

        Returns
        -------
        gain_proportional, gain_integral, generator_current_bias, max_output,\
 pi_setpoint, delay_buffer, integral
            The controller parameters and marshalled state passed to
            :func:`~blond.physics.feedbacks.envelope_kernel.envelope_pi_scan`.
        """
        if not controller_active:
            return (
                0.0,
                0.0,
                0.0 + 0.0j,
                np.inf,
                0.0 + 0.0j,
                np.zeros(1, dtype=np.complex128),
                0.0 + 0.0j,
            )
        controller = self._controller
        max_output = (
            np.inf
            if controller.max_output is None
            else float(controller.max_output)
        )
        return (
            float(controller.gain_proportional),
            float(controller.gain_integral),
            complex(controller.generator_current_bias),
            max_output,
            complex(self.pi_setpoint),
            np.array(controller._delay_line, dtype=np.complex128),
            complex(controller._integral),
        )

    def _store_controller_state(
        self,
        delay_buffer: NumpyArray,
        delay_head: int,
        integral: complex,
    ) -> None:
        """
        Write the kernel's PI state back into the controller.

        Rebuilds the controller's delay-line :class:`~collections.deque` from
        the circular buffer (oldest-first from ``delay_head``) and stores the
        error integral, so a subsequent segment or turn resumes exactly where
        the kernel left off.

        Parameters
        ----------
        delay_buffer
            The circular delay buffer after the kernel scan.
        delay_head
            The buffer head index after the scan.
        integral
            The committed error integral after the scan.
        """
        buffer_len = delay_buffer.shape[0]
        self._controller._delay_line = collections.deque(
            (
                delay_buffer[(delay_head + offset) % buffer_len]
                for offset in range(buffer_len)
            ),
            maxlen=buffer_len,
        )
        self._controller._integral = integral

    def _check_beam_kicks(
        self,
        beam_current: NumpyArray,
        omega_times_dt: NumpyArray,
        voltage_init: complex,
        voltage_out: NumpyArray,
        skip_first: bool,
    ) -> None:
        """
        Vectorised beam-kick tripwire for a forward kernel segment.

        Reproduces the per-cell :meth:`_check_beam_kick_magnitude` sweep of the
        reference path: it locates the offending cells with vectorised numpy
        and then delegates to :meth:`_check_beam_kick_magnitude` for the actual
        raise/warn, so the exception and warning messages (and the once-only
        warning flag) are identical. A merely-large kick that precedes any hard
        violation warns first, matching the reference ordering.

        Parameters
        ----------
        beam_current
            Per-cell beam current of the segment.
        omega_times_dt
            Per-cell ``omega * dt`` of the segment.
        voltage_init
            Antenna voltage preceding the first cell.
        voltage_out
            Per-cell antenna voltage just computed for the segment.
        skip_first
            Whether to skip the first cell (the carried ``rf_centers`` index 0,
            which the reference never checks).
        """
        # The exponential propagator integrates the piecewise-constant drive
        # (beam included) exactly, so a large per-step beam kick is not a
        # discretisation error -- the forward-Euler beam-kick tripwire does
        # not apply and must be skipped, mirroring _check_step_sizes.
        if self._exponential_coarse_solver_enable:
            return
        previous_voltage = np.empty_like(voltage_out)
        previous_voltage[0] = voltage_init
        previous_voltage[1:] = voltage_out[:-1]
        beam_kick = beam_current * 0.5 * self.R_over_Q * omega_times_dt
        previous_voltage_abs = np.abs(previous_voltage)
        valid = (beam_current != 0) & (previous_voltage_abs != 0)
        if skip_first:
            valid[0] = False
        if not valid.any():
            return
        relative_kick = np.zeros(beam_current.shape[0], dtype=np.float64)
        relative_kick[valid] = (
            np.abs(beam_kick[valid]) / previous_voltage_abs[valid]
        )
        # Same thresholds as the per-cell _check_beam_kick_magnitude.
        max_relative_kick = 0.1
        max_relative_kick_hard = 1.0
        hard = valid & (relative_kick > max_relative_kick_hard)
        soft = valid & (relative_kick > max_relative_kick)
        first_hard = int(np.argmax(hard)) if hard.any() else -1
        first_soft = int(np.argmax(soft)) if soft.any() else -1

        # Warn once if a merely-large kick precedes any hard violation, so the
        # observable warn-then-raise ordering matches the per-cell loop.
        if first_soft >= 0 and (first_hard < 0 or first_soft < first_hard):
            self._check_beam_kick_magnitude(
                beam_current[first_soft],
                omega_times_dt[first_soft],
                previous_voltage[first_soft],
            )
        if first_hard >= 0:
            self._check_beam_kick_magnitude(
                beam_current[first_hard],
                omega_times_dt[first_hard],
                previous_voltage[first_hard],
            )

    def _check_beam_kick_magnitude(
        self,
        beam_current: complex | float | int,
        omega_times_T_s: float | int,
        previous_voltage: complex | float | int,
    ) -> None:
        """
        Warn (once) if the beam-induced voltage kick is too large.

        Warns if the beam-induced voltage kick within a single coarse-grid
        step is not small compared to the antenna voltage it is added to.

        The beam-loading term ``-I_beam * 0.5 * R_over_Q * omega * dt`` is,
        like the cavity decay/detuning terms, a forward-Euler increment.
        It is only an accurate discretization of the underlying ODE if it
        represents a small relative change of the antenna voltage per step;
        a large beam current (or large step size) violates that assumption
        in the same way an excessive ``omega * dt / Q_L`` or
        ``delta_omega * dt`` would.

        Parameters
        ----------
        beam_current
            Beam current sample used for this step [A].
        omega_times_T_s
            Angular frequency times sampling time for this step.
        previous_voltage
            Antenna voltage of the previous coarse-grid step, which the
            kick is added to/subtracted from.
        """
        # Forward-Euler validity check only: the exponential propagator
        # integrates the piecewise-constant drive (beam included) exactly, so
        # a large per-step beam kick is not a discretisation error there. Skip
        # it for the exact solver, mirroring _check_step_sizes' early return.
        if self._exponential_coarse_solver_enable:
            return
        if beam_current == 0:
            return

        max_relative_kick = 0.1  # heuristic threshold for Euler validity

        # Beyond this, the single-step beam kick exceeds the antenna
        # voltage it is being subtracted from/added to: the discretized
        # update can flip the sign of the antenna voltage within one
        # coarse-grid step purely due to the beam, which the underlying
        # continuous cavity equation cannot do -- a divergent/unphysical
        # discretization rather than just an inaccurate one.
        max_relative_kick_hard = 1.0

        beam_kick = beam_current * 0.5 * self.R_over_Q * omega_times_T_s
        previous_voltage_abs = np.abs(previous_voltage)
        if previous_voltage_abs == 0:
            return

        relative_kick = np.abs(beam_kick) / previous_voltage_abs
        if relative_kick > max_relative_kick_hard:
            raise ValueError(
                f"{relative_kick=:.3g} > {max_relative_kick_hard}: the "
                "beam-induced voltage kick per coarse-grid step "
                "(beam_current * 0.5 * R_over_Q * omega * dt) exceeds the "
                "antenna voltage it acts on, i.e. the forward-Euler update "
                "in cavity_response() can flip the sign of the antenna "
                "voltage within a single step -- unphysical for the "
                "underlying cavity ODE. Decrease "
                "n_rf_periods_per_coarse_grid or check whether the beam "
                "current/intensity is physically reasonable for this "
                "cavity."
            )
        if self._beam_kick_warning_issued:
            return
        if relative_kick > max_relative_kick:
            self._beam_kick_warning_issued = True
            warnings.warn(
                f"{relative_kick=:.3g} is not << 1: the beam-induced "
                "voltage kick per coarse-grid step "
                "(beam_current * 0.5 * R_over_Q * omega * dt) is large "
                "compared to the antenna voltage. The forward-Euler update "
                "in cavity_response() may be inaccurate; consider "
                "decreasing n_rf_periods_per_coarse_grid or checking "
                "whether the beam current/intensity is physically "
                "reasonable for this cavity.",
                stacklevel=2,
            )

    def _advance_coarse_voltage(
        self,
        v_prev: complex,
        generator_current: complex,
        beam_current: complex,
        omega_times_T_s: float,
        relative_detuning: float,
    ) -> complex:
        r"""
        Advance the coarse-grid antenna voltage by one step.

        Integrates the cavity envelope ODE
        ``dV/dt = lambda V + (R/Q) omega (I_gen - I_beam/2)`` with
        ``lambda = -omega/(2 Q_L) + i delta_omega`` over one coarse step,
        using either the default forward-Euler discretisation or (when
        ``exponential_coarse_solver_enable`` is set) the exact exponential
        propagator for the piecewise-constant drive:

        .. math::
            V_{n+1} = e^{L} V_n
                + \mathrm{src}\,\frac{e^{L} - 1}{L},
            \quad L = -\frac{\omega\,\Delta t}{2 Q_L}
                + i\,\Delta\omega\,\Delta t,

        with the per-step drive ``src = (R/Q) omega dt (I_gen - I_beam/2)``
        (identical to the Euler source term). The exponential form is exact
        in both decay and detuning rotation and unconditionally stable, so
        it removes the forward-Euler step-size cap; as ``L -> 0`` it reduces
        to the Euler update. Default (Euler) behaviour is bit-unchanged.

        Parameters
        ----------
        v_prev
            Antenna voltage of the previous coarse sample [V].
        generator_current
            Generator current driving this step [A].
        beam_current
            Beam current of this step [A].
        omega_times_T_s
            Angular frequency times the step time (``omega * dt``).
        relative_detuning
            Detuning normalised to the step frequency
            (``delta_omega / omega``), so ``delta_omega * dt =
            relative_detuning * omega_times_T_s``.

        Returns
        -------
        complex
            The advanced antenna voltage [V].
        """
        drive = (
            self.R_over_Q
            * omega_times_T_s
            * (generator_current - 0.5 * beam_current)
        )
        # L = lambda * dt (dimensionless growth exponent for this step).
        step_exponent = (
            -0.5 * omega_times_T_s / self.Q_L
            + 1j * relative_detuning * omega_times_T_s
        )
        if not self._exponential_coarse_solver_enable:
            return v_prev * (1.0 + step_exponent) + drive
        # Exact exponential propagator. np.expm1 keeps the drive weight
        # (e^L - 1) / L accurate (-> 1) as L -> 0; guard the exact zero.
        growth = np.exp(step_exponent)
        if step_exponent == 0:
            drive_weight = 1.0
        else:
            drive_weight = np.expm1(step_exponent) / step_exponent
        return v_prev * growth + drive * drive_weight

    def cavity_response(
        self,
        omega_times_T_s: float,
        coarse_grid_index_to_update: int,
        relative_detuning: float,
        no_beam: bool = False,
    ):
        """
        Calculate antenna voltage on the coarse grid for a specific index.

        Parameters
        ----------
        omega_times_T_s
            Angular frequency times sampling time.
        coarse_grid_index_to_update
            Coarse grid index to update.
        relative_detuning
            Detuning normalized to the current RF frequency.
        no_beam
            If no beam is present, the beam current is set to 0.
        """
        if coarse_grid_index_to_update != 0:
            if no_beam:
                beam_current = 0
            else:
                forward_offset = (
                    len(self._rf_centers) - self._rf_centers_lengths[-1]
                )
                beam_current = self.beam_current_forward_coarse_grid[
                    coarse_grid_index_to_update - forward_offset
                ]
            self._check_beam_kick_magnitude(
                beam_current=beam_current,
                omega_times_T_s=omega_times_T_s,
                previous_voltage=self.antenna_voltage_coarse_grid[
                    coarse_grid_index_to_update - 1
                ],
            )
            self.antenna_voltage_coarse_grid[coarse_grid_index_to_update] = (
                self._advance_coarse_voltage(
                    v_prev=self.antenna_voltage_coarse_grid[
                        coarse_grid_index_to_update - 1
                    ],
                    generator_current=self.generator_current_coarse_grid[
                        coarse_grid_index_to_update - 1
                    ],
                    beam_current=beam_current,
                    omega_times_T_s=omega_times_T_s,
                    relative_detuning=relative_detuning,
                )
            )
        else:
            self.antenna_voltage_coarse_grid[coarse_grid_index_to_update] = (
                self._advance_coarse_voltage(
                    v_prev=self._last_val_ant_voltage,
                    generator_current=self._last_val_generator_current,
                    beam_current=self._last_val_beam_current,
                    omega_times_T_s=omega_times_T_s,
                    relative_detuning=relative_detuning,
                )
            )

        # With the PI control active, regulate the generator current of this
        # coarse-grid index from the antenna-voltage error just computed; it
        # then drives the next step. Inactive by default (constant current).
        # Only on the real forward pass (not the no_beam reverse
        # reconstruction segments): the reverse cells carry a per-segment
        # frame phase (corrected only on the last sample), so stepping the
        # controller there would integrate frame-rotated errors and
        # double-advance its delay line / integrator. Single-section rings
        # have no reverse segments, so this is a no-op there.
        if self._controller_active and not no_beam:
            self._update_generator_current(
                omega_times_T_s=omega_times_T_s,
                coarse_grid_index_to_update=coarse_grid_index_to_update,
            )

    def update_feedback_variables(self) -> None:
        """No-op: this feedback has no per-turn variables to refresh."""
        pass

    def reset_arrays(self):
        """Reset coarse grid arrays to match rf_centers length and save last values."""
        if self.antenna_voltage_coarse_grid is None:
            self._last_val_ant_voltage = self._init_voltage
        else:
            self._last_val_ant_voltage = self.antenna_voltage_coarse_grid[-1]
        self.antenna_voltage_coarse_grid = np.zeros(
            len(self._rf_centers), dtype=np.complex128
        )
        if self.generator_current_coarse_grid is None:
            self._last_val_generator_current = self._generator_current_bias
        else:
            self._last_val_generator_current = (
                self.generator_current_coarse_grid[-1]
            )

        self.generator_current_coarse_grid = (
            np.ones(len(self._rf_centers), dtype=np.complex128)
            * self._generator_current_bias
        )

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Track the feedback for one turn.

        Parameters
        ----------
        beam
            Beam to be tracked.

        Raises
        ------
        NotImplementedError
            When two counter-rotating beams pass this station simultaneously
            (the station sits at a meeting azimuth of the two beams).
        """
        # Simultaneous counter-rotating passage guard. When the station sits
        # at a meeting azimuth of the two beams (e.g. the single mid-ring
        # station of a one-section layout), both beams arrive at the same
        # reference time and the per-passage grid machinery would silently
        # serialize the two arrivals one full projection window apart -- the
        # envelope then runs at twice the physical rate and the summed
        # loading is wrong (measured ~47 % L2 on the first turn). Interleaved
        # (offset-time) passages, e.g. any even section count with stations
        # away from the meeting points, are handled correctly and pass this
        # guard.
        if (
            self._last_track_is_counter_rotating is not None
            and beam.is_counter_rotating
            != self._last_track_is_counter_rotating
            and self._last_track_arrival_time is not None
            and self._last_forward_cell_width is not None
            and abs(beam.reference.time - self._last_track_arrival_time)
            < 0.5 * self._last_forward_cell_width
        ):
            raise NotImplementedError(
                "Two counter-rotating beams pass this RF station "
                "simultaneously (station at a meeting azimuth of the two "
                "beams). The cavity feedback cannot yet integrate two "
                "coincident beam currents; place the station away from the "
                "beams' meeting points (e.g. an even number of sections "
                "with the half-drift / station / half-drift layout), or "
                "model the beam loading of this station with the "
                "MultiPassResonatorSolver wakefield "
                "(allow_delta_t_zero=True) instead."
            )
        self._last_track_arrival_time = beam.reference.time
        self._last_track_is_counter_rotating = beam.is_counter_rotating

        # Live tail of the RF-frequency-offset phase slip: the station's
        # kick clock (delta_phi_rf) is accumulated only at the END of each
        # station track (the blond2 convention the LHC comparisons pin), so
        # during this passage it lags the true integral
        # ``int delta_omega_rf dt`` by the slip since its last tick. The
        # station clock plus this gap is the exact, continuous slip at the
        # current passage; the demodulation subtracts it and
        # ``phase_correction`` adds it back at the readout, anchoring the
        # envelope frame to the actual RF carrier on both sides (see
        # calculate_rf_beam_current_partial). Exactly 0.0 without an
        # offset.
        station_clock_last = (
            self._parent_rf_station._last_reference_time_phase_slip
        )
        self._carrier_slip_gap = (
            0.0
            if station_clock_last is None
            else self.delta_omega_rf
            * (beam.reference.time - station_clock_last)
        )

        if len(self._rf_centers) != 0:
            self._last_rf_centers_entry = self._rf_centers[-1]

        self._clear_segments()

        if self._tracked_forward_until_element is not None:  # noqa: SIM102
            if (
                self._tracked_forward_until_element
                is not self._parent_rf_station
            ):  # otherwise, the full turn was already tracked
                self.calculate_rf_centers_for_reverse_direction(beam=beam)
        elif self._parent_rf_station._turn_counter.value == 0:
            # at first call, this always needs to be tracked, since the values from the start of the simulation until now are not retrieved yet.
            self.calculate_rf_centers_for_reverse_direction(beam=beam)

        len_rev = len(self._rf_centers)

        remaining_delta_t_from_reverse_tracking = (
            self._residual_time_last_rf_centers_calculation
        )

        self.calculate_rf_centers_for_forward_direction(beam=beam)

        # The flat rf_centers / rf_centers_lengths arrays are derived from
        # _segments; assert they stayed consistent after this turn's
        # generation. This is a per-turn integrity check with no effect on the
        # result, so it runs only under ``debug`` (it walks the whole grid
        # every turn otherwise).
        if self._debug:
            self._validate_grid()

        # Coincidence tolerance for the simultaneous-passage guard above:
        # one coarse-cell width, taken from the last two grid centers.
        min_centers_for_cell_width = 2
        if len(self._rf_centers) >= min_centers_for_cell_width:
            self._last_forward_cell_width = float(
                self._rf_centers[-1] - self._rf_centers[-2]
            )

        self.reset_arrays()

        # Only walk the reverse segments when this turn actually generated
        # some (len_rev > 0). For a single section the reverse omega list
        # from turn 0 is never refreshed, so without the len_rev gate this
        # loop re-ran the ENTIRE forward grid every turn at the frozen
        # turn-0 frequency (no_beam) before the demodulation and the real
        # forward pass. The envelope overwrite was recomputed identically by
        # the real pass, but under a ramp the spurious pass corrupted the
        # sub-stepped demodulation frame by -(turn+1) * 2 pi S per turn and
        # stepped an attached controller once per turn on garbage errors.
        if len_rev > 0 and self._reverse_tracking_omega_list is not None:
            for omega_index, omega_track in enumerate(
                self._reverse_tracking_omega_list
            ):
                start_index = np.sum(
                    self._rf_centers_lengths[:omega_index], dtype=int
                )
                end_index = np.sum(
                    self._rf_centers_lengths[: omega_index + 1], dtype=int
                )

                self.circuit_track(
                    omega_input=omega_track,
                    start_index=start_index,
                    end_index=end_index,
                    no_beam=True,
                )

        len_frwrd = len(self._rf_centers) - len_rev

        # Multi-section grid-vs-carrier registration phase. A multi-section
        # passage builds its coarse grid piecewise: each reverse segment k
        # spans T_seg,k on its own (past-station) design frequency omega_k,
        # then the forward segment runs at omega_0. Over the passage the grid
        # therefore accumulates RF phase sum_k omega_k * T_seg,k, whereas the
        # beam-current demodulation and the readout both reference the single
        # forward carrier omega_0 (i.e. omega_0 * T_total). The two differ by
        #
        #     Psi = sum_k (omega_k - omega_0) * T_seg,k ,
        #
        # a pure bookkeeping mismatch between the piecewise grid clock and the
        # single readout carrier. A single section builds the whole passage
        # from one segment at omega_0, so Psi is identically zero there -- and
        # that is exactly why single-section runs need no correction at all.
        #
        # This is SEPARATE from the cavity's resonance detuning: circuit_track
        # passes relative_detuning = delta_omega / omega_input on every
        # segment, so the physical precession from delta_omega != 0 is already
        # applied by the recursion and must not be duplicated here (the
        # detuned multi-turn comparison confirms no double-counting).
        #
        # Psi is carried as an explicit *carrier* phase -- accumulated into
        # ``_grid_carrier_phase`` and folded into ``_carrier_slip_gap``, which
        # the demodulation subtracts (carrier_phase_offset) and the readout
        # adds back (phase_correction) -- exactly the idiom the RF-frequency
        # offset already uses, and the same one the design-clock invariant
        # prescribes: frequency mismatches enter as phases, never as grid
        # geometry. A deposit made at turn m is then read out at turn N with
        # the relative phase Phi_N - Phi_m, which is what the carried wake
        # needs to match the retuning convolution.
        #
        # It must NOT be applied as a rotation of the antenna-voltage state.
        # Doing that (the former behaviour) also rotated the generator-driven
        # field, which carries no registration error at all -- it is
        # re-injected on the current grid every coarse cell. The constant
        # drive then pulled the rotating state back toward the real axis and
        # the driven |V_ant| drifted ~3 % over 5 turns on the fast ramp
        # (~0.6 %/turn, diverging), while a single section held its steady
        # state to ~2e-12. See TestDrivenSteadyStateFastRamp.
        if (
            self._n_rf_stations_in_ring > 1
            and len_rev > 0
            and self._reverse_tracking_omega_list is not None
        ):
            self._grid_carrier_phase += float(
                np.sum(
                    (
                        np.asarray(self._reverse_tracking_omega_list)
                        - self._forward_tracking_omega_rf
                    )
                    * np.asarray(self._reverse_tracking_time_array)
                )
            )
        # Exactly +0.0 for a single section and for an unaccelerated ring, so
        # both stay bit-identical.
        self._carrier_slip_gap += self._grid_carrier_phase

        if self._debug:
            self.relative_voltage_correction = np.ones_like(
                self.profile.hist_x
            )
            self.phase_correction = np.zeros_like(self.profile.hist_x)
            return

        # default behavior
        self.calculate_rf_beam_current_partial(
            beam=beam,
            n_points=len_frwrd,
            remaining_delta_t_from_reverse_tracking=remaining_delta_t_from_reverse_tracking,
        )

        self.circuit_track(
            omega_input=self._forward_tracking_omega_rf,
            no_beam=False,
            start_index=len(self._rf_centers) - len_frwrd,
            end_index=len(self._rf_centers),
        )  # for all rf_centers

        # Convert to amplitude and phase
        self.relative_voltage_correction, alpha_sum = cartesian_to_polar(
            IQ_vector=self.antenna_voltage_fine_grid,
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.relative_voltage_correction /= (
            self.get_voltage_from_parent_rf_station()
        )
        # The station applies its (end-of-track-lagged) kick clock via
        # phi_rf; adding the live slip gap here completes the readout to
        # the exact accumulated actual-RF phase at this passage -- the
        # same total the demodulation subtracted (see
        # calculate_rf_beam_current_partial). Exactly +0.0 without an
        # RF-frequency offset.
        self.phase_correction = (
            alpha_sum
            - np.mean(np.angle(self.voltage_setpoint))
            + self._carrier_slip_gap
        )

        # dummy values

    def cavity_response_fine(
        self,
        initial_voltage_fine_grid: float,
        initial_voltage_gradient_fine_grid: float,
        initial_generator_current_fine_grid: float,
        samples_per_rf_fine_grid: float,
        relative_detuning: float,
    ):
        r"""
        ACS cavity response model in matrix form on the fine-grid.

        Parameters
        ----------
        initial_voltage_fine_grid : float
            Initial condition of the voltage on the fine grid.
        initial_voltage_gradient_fine_grid : float
            Initial condition of the voltage gradient on the fine grid.
        initial_generator_current_fine_grid : float
            Initial condition of the generator current on the fine grid.
        samples_per_rf_fine_grid
            Sample points per period on the fine grid.
        relative_detuning
            Cavity detuning relative to the center frequency.
        """
        # Enforce the controller's actuator (klystron) limit on the fine grid
        # too, so the response matrix never sees a current above the limit.
        # The coarse values are already clamped; this guards the interpolated
        # initial condition and any externally set fine-grid current.
        if self._controller is not None:
            self.generator_current_fine_grid = self._controller.limit(
                self.generator_current_fine_grid
            )
            initial_generator_current_fine_grid = self._controller.limit(
                initial_generator_current_fine_grid
            )

        cavity_response_solver = (
            cavity_response_sparse_matrix_second_order
            if self._second_order_fine_grid_solver_enable
            else cavity_response_sparse_matrix
        )
        self.antenna_voltage_fine_grid = cavity_response_solver(
            I_beam=self.beam_current_fine_grid,
            I_gen=self.generator_current_fine_grid,
            V_ant_init=initial_voltage_fine_grid,
            I_gen_init=initial_generator_current_fine_grid,
            samples_per_rf=samples_per_rf_fine_grid,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            relative_detuning=relative_detuning,
        )

        self.antenna_voltage_fine_grid *= self.n_cavities

    def calculate_rf_beam_current_partial(
        self,
        beam: BeamBaseClass,
        n_points: int,
        remaining_delta_t_from_reverse_tracking: float,
    ) -> None:
        r"""
        Calculate the IQ beam current for the coarse and fine grid.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        n_points
            Number of points in the resulting coarse grid.
        remaining_delta_t_from_reverse_tracking
            Remaining time from the last rf_centers calculation, causes phase shift in beam current calculation.
        """
        if self.profile.active:
            self.profile.track(beam=beam)

        # Beam current from profile
        sampling_time_frwrd = (
            self.n_rf_periods_per_coarse_grid
            * 2
            * np.pi
            / self._forward_tracking_omega_rf
        )
        self._last_val_beam_current = (
            self.beam_current_forward_coarse_grid[-1]
            if self.beam_current_forward_coarse_grid is not None
            else 0
        )
        # The demodulated current must be rotated into the frame of the
        # coarse-grid envelope recursion. Where that phase lives depends on
        # the grid convention of _generate_rf_centers:
        #
        # * n >= 1 (grid re-seeded at the design bucket phase every turn):
        #   the residual measures the grid against the design buckets and
        #   therefore already contains the *accumulated* acceleration frame
        #   slip (mod t_rf) plus the half-period bucket-centre offset -- the
        #   former residual-only demodulation term, validated by the n = 1
        #   acceleration tests. (An RF-frequency offset never enters the
        #   residual: the grid geometry is design-clock only.)
        # * n < 1 (sub-stepped grid, tiling continuously across turns): the
        #   demod frame is the gap from the previous turn's last centre to
        #   the first forward centre, which by the tiling construction is
        #   exactly one previous-frequency step: first-centre offset plus
        #   the carried residual (complementary by construction in
        #   _generate_rf_centers, so the sum is immune to the float-bistable
        #   residual landing flip and, being a pure time, to any mod-2*pi
        #   wrap). Constant frame turn over turn to O((n/h) * 2*pi*S) under
        #   a ramp with frame slip S; for n = 0.5 it evaluates to half an RF
        #   period (a pi rotation), the value validated by the static
        #   sub-stepped convolution comparison.
        if self.n_rf_periods_per_coarse_grid < 1:
            dT_demodulation = (
                self._rf_centers[len(self._rf_centers) - n_points]
                + remaining_delta_t_from_reverse_tracking
            )
        else:
            dT_demodulation = remaining_delta_t_from_reverse_tracking
        (
            self.beam_current_fine_grid,
            self.beam_current_forward_coarse_grid,
        ) = rf_beam_current_partial(
            beam=beam,
            profile=self.profile,
            # The demodulation carrier is the *design* RF frequency; the grid
            # geometry stays on the design clock too. The RF-frequency offset
            # enters only as the constant carrier_phase_offset below, not as
            # a within-window carrier shift (the residual intra-window
            # mismatch delta_omega_rf * hist_x is bunch-local and negligible;
            # see the class docstring).
            omega_c=self._forward_tracking_omega_rf,
            T_rev=self._forward_tracking_time,
            sampling_time=sampling_time_frwrd,
            n_points=n_points,
            dT=dT_demodulation,
            # Anchor the demodulation to the accumulated actual RF phase: the
            # slip ``int delta_omega_rf dt`` at this passage, i.e. the
            # station's kick clock (delta_phi_rf) plus the live gap since its
            # last end-of-track tick. The readout applies the same total
            # (station clock via phi_rf, gap via phase_correction), so the
            # inter-turn slip cancels for every deposit, however long it is
            # carried. Exactly 0.0 without an RF-frequency offset
            # (bit-identical demodulation).
            carrier_phase_offset=-(self.delta_phi_rf + self._carrier_slip_gap),
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / sampling_time_frwrd
        )
