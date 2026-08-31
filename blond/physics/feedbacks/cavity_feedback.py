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
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import AltersReference
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.cavities import (
    MultiHarmonicRFStation,
    RFStationBaseClass,
    SingleHarmonicRFStation,
    _coerce_harmonic_index,
)
from blond.physics.feedbacks.base import LocalFeedback
from blond.physics.feedbacks.beam_current import rf_beam_current
from blond.physics.feedbacks.cavity_solvers import (
    ForwardEulerValidityGuard,
    cavity_response_sparse_matrix,
    cavity_response_sparse_matrix_second_order,
    coarse_step_exponent,
    euler_voltage_multiplier,
    exponential_drive_weight,
    exponential_voltage_multiplier,
    pretrack_fill_voltage,
)
from blond.physics.feedbacks.envelope_kernel import (
    envelope_pi_scan,
    inactive_controller_scan_state,
)
from blond.physics.feedbacks.generator_regulation import (
    GeneratorRegulationMixin,
)
from blond.physics.feedbacks.iq import cartesian_to_polar
from blond.physics.feedbacks.rf_center_grid import RFCenterGridMixin
from blond.physics.feedbacks.rf_center_segment import (
    PerTurnGridSpan,
    RFCenterSegment,
)
from blond.physics.profiles import StaticProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.feedbacks.generator_current_controller import (
        GeneratorCurrentController,
    )


class IQCavityFeedbackBase(LocalFeedback):
    """
    Base class to design cavity feedbacks.

    Abstract IQ-envelope cavity feedback: it owns the beam profile, the
    coarse/fine grid arrays and the RF-parameter accessors onto the
    parent RF station. The muon-collider
    :class:`IQCavityFeedbackTimingClass` is its concrete subclass. The
    vocabulary is defined in the "Concepts and notation" section of
    :ref:`mucol_cavity_feedback_overview`.

    Parameters
    ----------
    profile
        Beam profile the feedback acts on.
    n_cavities
        Number of cavities the feedback controls. May be fractional: an
        effective-voltage scale (the summed fine-grid antenna voltage is
        the per-cavity voltage multiplied by ``n_cavities``) rather than
        a physical cavity count.
    n_rf_periods_per_coarse_grid
        Number of periods for the coarse grid.
    harmonic_index
        Index of the RF harmonic that should be controlled by the feedback.
    name
        Name of the object.

    Attributes
    ----------
    n_cavities
        Number of cavities the feedback is working on (may be fractional,
        see above).
    harmonic_index
        The harmonic index the cavity feedback is working on.
    n_rf_periods_per_coarse_grid
        Width of one coarse-grid step in RF periods; sets the coarse sampling
        time and thereby the number of coarse samples per turn.
    """

    def __init__(
        self,
        profile: StaticProfile,
        n_cavities: int | float,
        n_rf_periods_per_coarse_grid: int | float,
        harmonic_index: int,
        name: str | None = None,
    ):
        assert isinstance(profile, StaticProfile), (
            "IQ cavity feedbacks require static profiles"
        )
        super().__init__(
            profile=profile,
            name=name,
        )

        # Number of cavities the feedback is working on. Deliberately not
        # coerced to int: a fractional value is an effective-voltage scale.
        assert n_cavities > 0, f"{n_cavities=}, but must be bigger 0."
        self.n_cavities = n_cavities

        # The harmonic index the cavity feedback is working on. Strict
        # coercion: int / np.integer / integral float pass silently, a
        # fractional value is a hard error (a harmonic index is a list
        # slot, not a physical quantity to be rounded).
        self.harmonic_index = _coerce_harmonic_index(harmonic_index)

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

        # --- The eight IQ state arrays -------------------------------
        # All eight are complex IQ envelopes (demodulated at the design RF
        # carrier), all are host (numpy) arrays, and all are ``None``
        # until the first passage fills them. They live on two different
        # time grids, with two different index origins and two different
        # voltage scalings; the docstrings below are the contract each
        # reader must honour. Two of the conventions deliberately DIVERGE
        # from their siblings and are called out where they occur:
        # ``antenna_voltage_fine_grid`` is the only array scaled by
        # ``n_cavities``, and ``beam_current_forward_coarse_grid`` is the
        # only coarse array that is not whole-turn indexed.
        self.beam_current_forward_coarse_grid: NumpyArray | None = None
        """Beam current on the coarse grid, in [A].

        GRID: coarse, but only over this passage's FORWARD segment --
        ``n_forward_centers`` entries, one per forward coarse centre
        (``calculate_rf_beam_current_partial`` passes that count as
        ``n_points`` to ``rf_beam_current``).

        INDEX ORIGIN: FORWARD-SEGMENT-LOCAL, unlike the two whole-turn
        coarse grids below. Entry ``0`` is the first coarse centre of the
        forward segment, not of the turn, so a reader holding a whole-turn
        ``rf_centers`` index must subtract the forward offset
        ``len(rf_centers) - rf_centers_lengths[-1]`` -- which is exactly
        what ``cavity_response`` and ``_kernel_beam_current`` do. The
        backfill (already-elapsed) span at the head of the turn has no
        entry here at all, because it is replayed with ``no_beam=True``.

        SCALING: total beam current, NOT per cavity. It is never divided
        by ``n_cavities``: every cavity of the station is passed by the
        whole beam, so this is the current each single cavity sees, and it
        enters the per-cavity envelope step of ``_advance_coarse_voltage``
        directly.

        UNITS: amperes -- the demodulated beam charge [C] of
        ``rf_beam_current`` divided by the forward coarse sampling time.
        The sign follows the direction-signed charge convention of
        ``rf_beam_current`` (a counter-rotating beam gives the same gap
        current as a co-rotating one).
        """
        self.beam_current_fine_grid: NumpyArray | None = None
        """Beam current on the fine grid, in [A].

        GRID: fine -- the beam profile's own histogram grid, one entry per
        profile bin (``profile.n_bins``, sampled at ``profile.hist_x``).

        INDEX ORIGIN: profile-bin local, i.e. bunch-local time inside the
        profile window ``[profile.cut_left, profile.cut_right]``, reset
        every turn. It is not indexed against ``rf_centers`` at all.

        SCALING: total beam current, NOT per cavity (see the coarse
        sibling above).

        UNITS: amperes -- the demodulated beam charge [C] of
        ``rf_beam_current`` divided by ``profile.hist_step``.
        """
        self.antenna_voltage_coarse_grid: NumpyArray | None = None
        """Antenna voltage on the coarse grid, in [V].

        GRID: coarse, spanning the WHOLE passage -- ``len(rf_centers)``
        entries, i.e. this passage's backfill (already-elapsed) segments
        followed by its forward segment. ``reset_arrays`` sizes it from
        ``rf_centers`` every turn.

        INDEX ORIGIN: whole-turn -- entry ``i`` belongs to
        ``rf_centers[i]``, aligned one-to-one with the grid. (Beware when
        comparing against ``beam_current_forward_coarse_grid``, which is
        forward-segment-local.)

        SCALING: PER CAVITY -- the voltage of one single cavity of the
        station. The station total is this times ``n_cavities``; only the
        fine-grid antenna voltage below carries that factor already.

        UNITS: volts, as a complex IQ envelope in the antenna-voltage
        frame (amplitude ``abs``, phase ``angle``); it is not the
        instantaneous gap voltage.

        FRAME / COMPOSITION: this is the DEMODULATION-FRAME SUM of the two
        source-split components below,
        ``antenna_voltage_beam_coarse_grid +
        antenna_voltage_gen_coarse_grid * generator frame rotation``
        (see ``IQCavityFeedbackTimingClass._update_frame_rotations``). The
        components are the propagated state; this sum is (re)composed from
        them with the CURRENT passage's rotation. For an undriven feedback
        (zero generator current for the whole run) it equals the beam
        component bit-for-bit.
        """
        self.antenna_voltage_gen_coarse_grid: NumpyArray | None = None
        """Generator-sourced antenna voltage on the coarse grid, in [V].

        The generator-driven component of the (linear) envelope ODE:
        same grid, index origin and per-cavity scaling as
        ``antenna_voltage_coarse_grid``, propagated by the same coarse
        recursion but sourced by the generator current alone.

        FRAME: natively anchored to the piecewise DESIGN clock -- the
        generator current is injected as a constant per segment at each
        segment's own design frequency, which *are* samples of the design
        program, so this component carries neither the kick-clock slip
        nor the multi-section registration phase. It is rotated into the
        demodulation frame only when the sum above is composed. Stays
        identically zero while the generator current is zero and no
        initial/pre-fill voltage was given.
        """
        self.antenna_voltage_beam_coarse_grid: NumpyArray | None = None
        """Beam-sourced antenna voltage on the coarse grid, in [V].

        The beam-induced component of the (linear) envelope ODE: same
        grid, index origin and per-cavity scaling as
        ``antenna_voltage_coarse_grid``, propagated by the same coarse
        recursion but sourced by ``-I_beam / 2`` alone.

        FRAME: the demodulation frame -- deposits enter through the
        demodulated beam current (whose ``carrier_phase_offset``
        subtracted the accumulated actual-RF slip and the registration
        phase) and the readout adds the identical total back, closing
        the chain for every carried deposit exactly as before the split.
        For an undriven feedback this component IS the former single
        state, bit-for-bit.
        """
        self.antenna_voltage_fine_grid: NumpyArray | None = None
        """Antenna voltage on the fine grid, in [V], times ``n_cavities``.

        GRID: fine -- the profile grid (``profile.n_bins`` entries at
        ``profile.hist_x``), integrated by ``cavity_response_fine`` from
        the first forward coarse cell's initial condition.

        INDEX ORIGIN: profile-bin local (bunch-local time), like the fine
        beam current above.

        SCALING: TOTAL STATION -- and this is the one scaling divergence
        among the arrays: ``cavity_response_fine`` ends with
        ``antenna_voltage_fine_grid *= n_cavities``, while both coarse
        grids stay per cavity. That is what makes the readout
        ``relative_voltage_correction = abs(V_ant_fine) / station voltage``
        come out around 1, since the parent station's ``voltage`` is the
        total station voltage.

        UNITS: volts, complex IQ envelope (same frame as the coarse
        antenna voltage).
        """
        self.generator_current_coarse_grid: NumpyArray | None = None
        """Generator current on the coarse grid, in [A].

        GRID: coarse, spanning the WHOLE passage -- ``len(rf_centers)``
        entries, sized and seeded by ``reset_arrays`` (with the
        feedforward ``generator_current_bias``, except over the leading
        backfill cells, which hold the last commanded value).

        INDEX ORIGIN: whole-turn, aligned one-to-one with ``rf_centers``,
        exactly like ``antenna_voltage_coarse_grid``. Entry ``i`` is the
        current that DRIVES the step to cell ``i + 1``
        (``cavity_response`` reads index ``i - 1`` when updating ``i``).

        SCALING: PER CAVITY -- the current fed to one single cavity, the
        same convention as the coarse antenna voltage (the two are related
        by the per-cavity ``R_over_Q``). With a controller attached it is
        the controller's per-cavity output.

        UNITS: amperes, complex IQ envelope.
        """
        self.generator_current_fine_grid: NumpyArray | None = None
        """Generator current on the fine grid, in [A].

        GRID: fine -- the profile grid (``profile.n_bins`` entries at
        ``profile.hist_x``), obtained in ``circuit_track`` by interpolating
        the FORWARD segment of ``generator_current_coarse_grid`` onto
        ``profile.hist_x``.

        INDEX ORIGIN: profile-bin local (bunch-local time), like the other
        two fine grids.

        SCALING: PER CAVITY -- it is an interpolation of the per-cavity
        coarse grid and is never multiplied by ``n_cavities`` (the factor
        is applied to the resulting fine antenna voltage instead).

        UNITS: amperes, complex IQ envelope.
        """

        # Number of RF stations in the ring, filled in on_run_simulation once
        # the ring is known. The default of one station is the conservative
        # choice for consumers that size buffers from it (see
        # n_rf_stations_in_ring): fewer stations means a wider grid margin.
        self._n_rf_stations_in_ring: int = 1

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
    def circuit_track(
        self,
        omega_input: float,
        no_beam: bool = False,
        start_index: int = 0,
        end_index: int = -1,
    ) -> None:
        r"""
        Advance the feedback circuit over a coarse-grid segment.

        Parameters
        ----------
        omega_input
            Frequency in the tracked segment.
        no_beam
            Beam dependant parts of the feedback can be skipped if this is True.
        start_index
            Index of the coarse grid at which to start computing the response.
        end_index
            Index of the coarse grid until which to compute the response.

        Notes
        -----
        This is meant to be implemented in the child class by the user.
        """
        pass

    @property
    def n_rf_stations_in_ring(self) -> int:
        """
        Number of RF stations in the ring this feedback belongs to.

        Counted against ``RFStationBaseClass``, so single- and multi-harmonic
        stations count alike. Consumers that size a per-turn buffer need it
        because one turn of coarse grid can overshoot by up to one section
        (see
        :class:`~blond.handle_results.observables.IQCavityFeedbackObservation`).

        Returns
        -------
        n_rf_stations_in_ring
            Number of RF stations; one until the simulation is initialised.
        """
        return self._n_rf_stations_in_ring

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
        Coarse step evaluated on the actual RF frequency [s].

        ``n_rf_periods_per_coarse_grid`` periods of ``omega_rf``, the actual
        (offset) RF frequency. Note this is *not* the step the coarse grid is
        built with: the grid is generated on the design clock,
        ``2 * pi / omega_rf_design`` (see ``rf_center_grid``), so with a
        non-zero ``delta_omega_rf`` the two differ by the relative offset
        ``delta_omega_rf / omega_rf``. Do not use this value to build grid
        geometry -- take the step from the design frequency instead.

        Its only consumer is ``_check_step_sizes``, where the companion
        factor ``omega_rf * sampling_time_coarse`` cancels to exactly
        ``2 * pi * n_rf_periods_per_coarse_grid``, so the forward-Euler decay
        bound is unaffected by the clock this property uses.

        Returns
        -------
        sampling_time_coarse
            Coarse step on the actual RF frequency [s].
        """
        return self.n_rf_periods_per_coarse_grid * 2 * np.pi / self.omega_rf

    @property
    def station_voltage_coarse_grid(self) -> NumpyArray:
        """
        Parent rf station voltage replicated over the coarse grid [V].

        The *total* station voltage (all cavities of this station), one value
        per coarse sample, at phase 0 by construction. This is the frame the
        readout ``phase_correction`` is referenced to.

        This is not the controller setpoint. The PI regulates to
        ``pi_setpoint``, which is the explicit per-cavity ``voltage_setpoint``
        given at construction, or -- when that is None -- this station voltage
        divided by ``n_cavities``.

        Returns
        -------
        station_voltage_coarse_grid
            Station voltage on the coarse grid [V].
        """
        return (
            np.ones_like(self.antenna_voltage_coarse_grid)
            * self.get_voltage_from_parent_rf_station()
        )


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
    step the feedback forms the antenna-voltage error in the *kick frame*
    -- ``V_set - V_sum[n] * exp(+i (gap + Psi))``, the envelope the station
    actually applies -- and lets the controller convert it into the
    generator current (see ``_update_generator_current``). All control
    tuning (gains, loop delay, klystron limit) lives on the controller.

    Where this sits in the turn: the parent RF station first applies its
    scheduled parameters for this passage, then calls this feedback's
    ``_track``, which rebuilds the coarse grid, demodulates the beam
    current, advances the envelope and finally writes
    ``relative_voltage_correction`` and ``phase_correction``; only after
    that does the station advance the beam reference
    (``track_reference``) and build the interpolated kick out of those two
    arrays (``calc_gap_voltage_with_feedbacks``). So the
    antenna voltage computed during a passage shapes THAT SAME passage's
    kick, not the next one's -- the loop is closed within the turn. The
    one quantity deliberately left a step behind is the station's kick
    clock ``delta_phi_rf``, accumulated at the *end* of the station track,
    which is precisely why the demodulation has to add the live tail
    ``_carrier_slip_gap`` on top of it (see the RF-frequency offset note
    below).

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
        Save inspection-only diagnostic parameters during runtime (the
        element slices and reference time/energy snapshots written by
        :class:`~blond.physics.feedbacks.rf_center_grid.RFCenterGridMixin`).
        Pure observation: it does not change the physics and, since the
        flag split described in Notes, it no longer disables the
        correction either. Default is False.
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
        Explicit **per-cavity** voltage setpoint in the IQ frame [V] used to
        form the error the controller acts on. Reachable as ``pi_setpoint``;
        if None, ``pi_setpoint`` derives it from the parent rf station as
        station voltage / ``n_cavities``. Distinct from the read-only
        ``station_voltage_coarse_grid`` property, which is the *total*
        station voltage over the coarse grid. Must be real and positive
        (phase 0): the station's phase correction is referenced to the
        station voltage at phase 0, so a rotated setpoint would be regulated
        but not applied -- a non-real value raises ``ValueError``. Rotate
        ``phi_rf`` on the station instead.
    n_pretrack
        Feedforward cavity fill budget in turns. If given, the initial antenna
        voltage is seeded (in ``on_run_simulation``) from the constant-current
        fill of the cavity instead of the scalar ``initial_voltage``; see
        :func:`~blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`. The
        fill uses the constant ``generator_current_bias`` only -- the
        controller, if any, acts on the tracked turns after injection, and it
        is evaluated on the *design* RF clock, the clock the coarse recursion
        it seeds is driven at. Default None (start from ``initial_voltage``).
    injection_voltage
        Target ``|V_ant|`` [V] at injection. When set (requires ``n_pretrack``)
        the seed is the fill transient at the moment ``|V_ant|`` first reaches
        this value, i.e. the beam is injected part-way through the fill.
        Default None (seed from the fill after ``n_pretrack`` turns).
    validate_grid_each_turn
        Re-check every turn that the flat ``rf_centers`` arrays still
        agree with the ``_segments`` they are derived from, and that the
        forward segment's boundary residual equals its demodulation
        frame. A pure integrity check with no effect on the result; it
        walks the whole grid every turn, which is why it is opt-in.
        Default is False.
    grid_only_no_correction
        Build the coarse grid and replay the elapsed backfill span, then
        END THE TURN THERE: the beam current is never demodulated, the
        forward segment is never tracked, and **the feedback applies NO
        correction to the parent RF station**. It writes the neutral
        readout instead -- unit relative voltage, zero phase -- so the
        station kicks as if no feedback were attached. Only for
        inspecting the grid geometry in isolation; it is not a physical
        mode. Default is False.
    harmonic_index
        Index into the parent station's harmonic list that this feedback
        regulates: every RF parameter (``omega_rf``, ``phi_rf``, the
        station voltage, ...) and the coarse-grid design frequency are
        read at this harmonic. This is only the default used while the
        feedback is unattached: ``attach_cavity_feedback`` (and the
        station constructor, which routes through it) overrides it with
        the slot the feedback is placed at in ``cavity_feedback_list``
        -- the slot is authoritative. A ``cavity_feedback_list`` mutated
        directly after the attach is still caught at run start (see
        ``_validate_multi_harmonic_slot``); a single-harmonic station
        only has harmonic 0. Must be integral -- ``int``, ``np.integer``
        or integral ``float``; a fractional value is rejected. Default
        is 0.

    Notes
    -----
    **The coarse grid (** ``_rf_centers`` **).** Everything this class
    computes is indexed by this one array: the coarse antenna voltage, the
    coarse generator current and the forward beam current are all sampled
    on it, and the fine (profile) grid is seeded from its first forward
    entry. It has two properties that a reader will not guess from the
    array itself, and they are independent of one another.

    *Segment-local times, hence NOT globally monotonic.* One passage's
    grid is built as an ordered list of ``RFCenterSegment`` records -- one
    per backfill frequency segment, plus the forward segment --
    and ``_rebuild_grid_arrays`` simply concatenates their ``centers``.
    Each segment's centres start near zero *in that segment's own frame*
    (see ``_generate_rf_centers``), so the flat array rises inside a
    segment and drops back to about ``t_rf / 2`` at every segment
    boundary. ``_rf_centers[k]`` is therefore NOT an absolute time:
    placing it globally would need the durations of all preceding
    segments. That is why ``_preceding_segment_residual`` exists, and why
    the coarse step into the first cell of segment ``j`` is
    ``residual_{j-1} + _rf_centers[start_j]`` instead of a difference of
    two neighbouring entries. Differencing ACROSS a segment boundary is
    meaningless (it comes out negative).

    *Phase-consistent, but not uniformly spaced in time.* Every centre
    sits at the same RF phase -- the falling-edge zero of
    ``sin(omega t)``, half an RF period into the bucket -- laid out on the
    DESIGN clock (``calc_omega_rf_design``). Station phases never move
    that seed: ``phi_rf_design`` and the accumulated ``delta_omega_rf``
    kick-clock slip enter only as demodulation/readout phases, never as
    grid geometry. The step, however, is
    ``n_rf_periods_per_coarse_grid * t_rf`` with the *design* ``t_rf`` OF
    THAT SEGMENT, and that period changes from segment to segment and
    from turn to turn under acceleration. Measured on a two-section
    accelerating ring: ``np.diff`` of a segment reproduces its own
    ``n * t_rf`` to ~1e-21 s, while the segment period itself shrinks by
    ~1e-15 s from one segment to the next. So the spacing is exact within
    a segment and different in the next one -- do not read a single
    constant ``dt`` off the flat array.

    The sub-stepping mode (``n_rf_periods_per_coarse_grid < 1``, see
    below) keeps the first property and drops the second's seed: there a
    segment's centres continue the previous segment's tiling -- one full
    *previous* step after its last centre, i.e. a first local centre of
    ``step_previous - residual_previous`` -- rather than restarting at the
    bucket's falling edge, so they are continuation samples and not bucket
    centres. Measured, that first local centre is only near ``t_rf / 2``
    while the carried residual is still ~0 (the very first turn) and sits
    at ~0 from the second turn on. The grid stays segment-local either
    way, and the local clock still restarts at every segment.

    **The three diagnostic flags.** ``debug``,
    ``validate_grid_each_turn`` and ``grid_only_no_correction`` were once
    a single ``debug`` flag that did all three things at once, so asking
    for diagnostics silently switched the physics off. They are now
    independent: ``debug`` only records, ``validate_grid_each_turn`` only
    checks, and ``grid_only_no_correction`` -- and nothing else -- stops
    the turn before the correction is computed. The former
    ``debug=True`` behaviour is all three set together. With all three at
    their default (``False``) the tracked result is bit-for-bit what
    ``debug=False`` produced before the split.

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
    (see ``forward_segment_omega_design``); the offset enters only as an
    explicit phase. Concretely, the beam current is demodulated at the design
    carrier ``forward_segment_omega_design`` and rotated by the accumulated
    slip ``int delta_omega_rf dt`` -- the parent station's kick clock
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

    # Compile the per-cell coarse-envelope recursion to a numba host kernel
    # (see :mod:`~blond.physics.feedbacks.envelope_kernel`). The pure-Python
    # path is kept as the byte-identical reference and the fallback for
    # degenerate (coincident) coarse steps and klystron-limit saturation. Set
    # ``False`` on an instance to force the reference path.
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
        validate_grid_each_turn: bool = False,
        grid_only_no_correction: bool = False,
        harmonic_index: int = 0,
    ):
        super().__init__(
            profile=profile,
            n_cavities=n_cavities,
            harmonic_index=harmonic_index,
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
        """Flat coarse-grid centre times [s] of the current passage.

        The concatenation of the per-segment ``centers`` of ``_segments``,
        rebuilt by ``_rebuild_grid_arrays``. Every other quantity of this
        class is indexed by it, which is why its two counter-intuitive
        properties -- the entries are segment-LOCAL times (so the array is
        not globally monotonic) and the step is the design ``t_rf`` of the
        segment it belongs to (so the spacing is not one constant ``dt``)
        -- are spelled out in "The coarse grid" under Notes in the class
        docstring. Read that before indexing or differencing this array.
        """
        self._rf_centers_lengths = np.zeros(0, dtype=int)
        # Unfilled tail [s] between the last coarse centre generated
        # BEFORE the current passage and that passage; the
        # demodulation frame of calculate_rf_beam_current_partial. The
        # 0.0 here is a placeholder: the design RF period is not known
        # yet, so on_run_simulation overwrites it with the
        # tiling-consistent first-passage value (see
        # _seed_initial_demodulation_frame). Without that seed a
        # station that is the ring's FIRST reference-altering element
        # generates no backfill on turn 0 and demodulates that turn pi
        # out of phase -- the beam-induced voltage then comes out with
        # the wrong sign.
        self._residual_time_last_rf_centers_calculation = 0.0
        # Residual [s] the PREVIOUS turn's last segment ended on. The first
        # segment of a turn steps across the turn boundary from it; the live
        # scalar above cannot serve, because by the time the grid is walked
        # it has been overwritten by THIS turn's last-generated segment (see
        # _preceding_segment_residual).
        self._residual_time_carried_into_turn: float | None = None

        self._ring_circumference: float | None = None

        self._reference_altering_elements: (
            tuple[AltersReference, ...] | None
        ) = None
        self._reference_altering_elements_reverse: (
            tuple[AltersReference, ...] | None
        ) = None
        self._own_index_in_reference_list: int | None = None
        self._own_index_in_reference_list_reverse: int | None = None

        self._forward_segment_omega_design: float | None = None
        self._forward_tracking_time: float | None = None
        self._tracked_forward_until_element: AltersReference | None = None
        self._last_segment_omega_design: float | None = None
        # The residual tail expressed in RF PERIODS, i.e. a fraction, not a
        # count: ``rf_center_grid`` assigns
        # ``_residual_time_last_rf_centers_calculation / t_rf`` to it. It was
        # annotated ``int``, which the assignment never honours (only the
        # ``int(...)`` truncation at its n != 1 read site does).
        self._residual_taps_last_rf_centers_calculation: float = 0.0

        self._backfill_time_array: NumpyArray | None = None
        self._backfill_segment_omega_design_list: NumpyArray | None = None

        self._reference_state_until_tracked: ReferenceCoordinates | None = None
        self._reference_turn_offset: int = 0
        self._last_tracked_turn_frwrd: int = 0
        self._last_tracked_beam_state_frwrd: bool | None = None

        self._init_passage_tracking_state()

        self._phase_offset_frwrd_next: float = 0.0
        self._phase_offset_frwrd: float = 0.0

        self._init_turn_boundary_carries()

        self._init_voltage = initial_voltage

        # Three independent diagnostic switches (see the class Notes).
        # ``_debug`` records the inspection-only snapshots of
        # RFCenterGridMixin, ``_validate_grid_each_turn`` runs the
        # per-turn grid integrity check, and ``_grid_only_no_correction``
        # -- alone -- short-circuits _track before any correction is
        # computed. All three default to False, which is the tracked
        # (bit-unchanged) path.
        self._debug = debug
        self._validate_grid_each_turn = validate_grid_each_turn
        self._grid_only_no_correction = grid_only_no_correction

        self._second_order_fine_grid_solver_enable = (
            second_order_fine_grid_solver_enable
        )
        self._exponential_coarse_solver_enable = (
            exponential_coarse_solver_enable
        )

        self._generator_current_bias = generator_current_bias

        # Forward-Euler validity tripwires (per-step decay, detuning phase and
        # beam kick). Disabled in one place for the exact exponential
        # propagator, which is not subject to any of them.
        self._euler_guard = ForwardEulerValidityGuard(
            enabled=not exponential_coarse_solver_enable
        )

        # --- Optional generator-current controller ---
        # When ``controller`` is None the generator current stays at the
        # constant value (pure constant-current drive). Otherwise the
        # controller converts the antenna-voltage error into the generator
        # current; see _update_generator_current. All control tuning lives on
        # the controller, not on this feedback.
        self._controller = controller
        # Setpoint policy (the real-and-positive rule and its rationale)
        # lives with ``pi_setpoint`` on GeneratorRegulationMixin.
        self._validate_voltage_setpoint(voltage_setpoint)
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

    def _init_turn_boundary_carries(self) -> None:
        """
        Initialise the turn-boundary carries of the coarse recursion.

        The two antenna-voltage components are the propagated state (see
        :meth:`reset_arrays` / :meth:`cavity_response`); the un-suffixed
        value is the carried demodulation-frame SUM, kept for diagnostics
        and the coincident-first-cell duplication. ``_generator_active``
        says whether the generator-sourced component carries any signal
        at all this turn (bias, controller, carried current or carried
        voltage); refreshed by :meth:`reset_arrays`. While False, the
        component update and every composition multiply are skipped,
        keeping an undriven feedback bit-identical to the former
        single-state recursion. True is the safe default for direct
        (test) driving.
        """
        self._last_val_ant_voltage: complex = 0.0
        self._last_val_ant_voltage_gen: complex = 0.0
        self._last_val_ant_voltage_beam: complex = 0.0
        self._last_val_beam_current: float = 0.0
        self._last_val_generator_current: float = 0.0
        self._last_rf_centers_entry: float | None = None
        self._generator_active: bool = True

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
        # Live tail of the kick-clock slip at this passage (the slip since
        # the station clock's last end-of-track tick); one of the two
        # constituents folded into ``_carrier_slip_gap``.
        self._kick_clock_slip_gap: float = 0.0
        self._carrier_slip_gap: float = 0.0
        # Running total of the multi-section grid-vs-carrier registration
        # phase ``sum_k (omega_k - omega_0) T_seg,k`` (see ``_track``);
        # the other constituent of ``_carrier_slip_gap``.
        # Stays exactly 0.0 for a single section and without acceleration.
        self._grid_carrier_phase: float = 0.0
        # Per-passage frame rotations (see ``_update_frame_rotations``);
        # exactly unity until a passage computes them, which is also the
        # neutral value for direct (test) driving of the cell loops.
        self._generator_frame_rotation: complex = 1.0 + 0.0j
        self._kick_frame_rotation: complex = 1.0 + 0.0j
        self._pi_error_frame_rotation: complex = 1.0 + 0.0j

    def _seed_initial_demodulation_frame(self) -> None:
        r"""
        Seed the demodulation frame of the very first passage.

        Notes
        -----
        ``_residual_time_last_rf_centers_calculation`` is the unfilled
        tail between the last coarse centre generated before a passage
        and that passage; :meth:`calculate_rf_beam_current_partial`
        consumes it as the demodulation frame ``dT``. The fundamental
        theorem of beam loading -- a bunch must LOSE energy to its own
        wake -- holds only when the demodulation phase
        ``omega_c * dT`` comes out to ``pi`` (mod ``2 pi``); half an
        RF period off, and the bunch is accelerated by its own wake.

        Every later passage gets that tail from the segment
        generation. The very first one does not whenever the parent
        station is the ring's first reference-altering element: there
        is no elapsed span to reconstruct,
        ``calculate_rf_centers_for_backfill`` returns without
        generating anything, and the scalar would still hold its
        ``__init__`` placeholder -- so turn 0 alone would be
        demodulated exactly ``pi`` out of phase, and the wrongly
        signed deposit then decays only over ``2 Q_L / omega``, i.e.
        many turns.

        The seeded value is the backward continuation of the segment's
        own tiling. :meth:`_generate_rf_centers` seeds every segment at
        the falling-edge zero ``t_rf / 2`` and steps by ``n * t_rf``,
        so the virtual centre one full step before the first lies at
        ``t_rf / 2 - n * t_rf`` and the tail from it to the passage is
        ``n * t_rf - t_rf / 2``, giving ``omega_c * r_0 = 2 pi n - pi``,
        i.e. ``pi`` (mod ``2 pi``) for integer ``n``. Turn 0 is then
        demodulated in the same frame as every other turn.

        Seeded rather than guarded against: a ring is a loop, so which
        element the element list starts at is a bookkeeping choice and
        no physics may depend on it.

        Only the time is seeded, never
        ``_residual_taps_last_rf_centers_calculation``: the taps carry
        where the NEXT segment is seeded, and the first forward segment
        must still start at the design bucket phase, which ``taps == 0``
        encodes. The grid geometry -- and with it every ``delta_t`` the
        coarse recursion and its numba twin consume -- therefore stays
        bit-identical.

        ORDERING: must run before the first :meth:`_track`, i.e. before
        ``_close_previous_turn_grid`` snapshots
        ``_residual_time_carried_into_turn`` off this scalar.
        """
        if self._last_segment_omega_design is not None:
            # A segment already exists (a second run_simulation call on
            # a feedback that has tracked): the live scalar then holds
            # the real carried tail and must not be clobbered.
            return
        t_rf_design = 2.0 * np.pi / self.omega_rf_design
        self._residual_time_last_rf_centers_calculation = (
            self.n_rf_periods_per_coarse_grid * t_rf_design - t_rf_design / 2.0
        )

    @requires(["RFStationBaseClass"])
    def _check_step_sizes(self) -> None:
        """
        Hand this cavity's parameters to the forward-Euler step-size guard.

        See Also
        --------
        blond.physics.feedbacks.cavity_solvers.ForwardEulerValidityGuard.check_step_sizes : The thresholds, the messages and why the step must stay small.
        """
        self._euler_guard.check_step_sizes(
            omega_rf=self.omega_rf,
            sampling_time=self.sampling_time_coarse,
            Q_L=self.Q_L,
            delta_omega=self.delta_omega,
        )

    def _validate_multi_harmonic_slot(self) -> None:
        """
        Enforce slot/index agreement on a multi-harmonic parent station.

        ``MultiHarmonicRFStation.calc_gap_voltage_with_feedbacks``
        applies each feedback's ``phase_correction`` /
        ``relative_voltage_correction`` at the feedback's LIST slot
        (``enumerate(cavity_feedback_list)``), while the feedback
        computes them from the RF parameters at its OWN
        ``harmonic_index``. If the two disagree, corrections computed
        from harmonic A are silently applied to harmonic B -- no crash,
        wrong physics. Hence this run-start check: locate SELF in the
        parent's ``cavity_feedback_list`` by identity and require the
        slot to equal ``harmonic_index``.

        Run-start, not construction: the parent station is attached
        AFTER this feedback is built (``attach_cavity_feedback`` calls
        ``set_parent_rf_station``, typically from the station's own
        ``__init__`` with the feedback as an argument), so ``__init__``
        cannot see it. ``on_run_simulation`` is the first hook that both
        knows the parent and still precedes every grid build -- the same
        reason ``_check_step_sizes`` is called from there.

        Raises
        ------
        ValueError
            If this feedback is missing from the parent's
            ``cavity_feedback_list``, or occupies a slot different from
            its ``harmonic_index``.
        """
        if not isinstance(self._parent_rf_station, MultiHarmonicRFStation):
            return
        slots = [
            index
            for index, feedback in enumerate(
                self._parent_rf_station.cavity_feedback_list
            )
            if feedback is self
        ]
        if not slots:
            raise ValueError(
                f"{type(self).__name__} has a MultiHarmonicRFStation "
                "parent, but is not in that station's "
                "cavity_feedback_list, so the station would never apply "
                "its corrections. Pass the feedback to the station "
                "(cavity_feedback=..., "
                f"harmonic_index={self.harmonic_index}) when building "
                "it, instead of only setting the parent station."
            )
        if len(slots) > 1:
            raise ValueError(
                f"{type(self).__name__} occupies several slots "
                f"({slots}) of the parent MultiHarmonicRFStation's "
                "cavity_feedback_list. One feedback instance regulates "
                "one harmonic; build a separate feedback per harmonic."
            )
        slot = slots[0]
        if slot != self.harmonic_index:
            raise ValueError(
                f"{type(self).__name__} regulates the RF parameters of "
                f"harmonic_index={self.harmonic_index}, but occupies "
                f"slot {slot} of the parent MultiHarmonicRFStation's "
                "cavity_feedback_list, where "
                "calc_gap_voltage_with_feedbacks would silently apply "
                f"its corrections to harmonic {slot}. Construct the "
                f"feedback with harmonic_index={slot}, or place it at "
                f"slot {self.harmonic_index} when building the station."
            )

    @requires(["RFStationBaseClass", "BeamBaseClass"])
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

        Raises
        ------
        ValueError
            If the parent RF station is a
            :class:`~blond.physics.cavities.MultiHarmonicRFStation` and
            this feedback's ``harmonic_index`` disagrees with its slot in
            the parent's ``cavity_feedback_list``; see
            ``_validate_multi_harmonic_slot``.
        """
        self._validate_multi_harmonic_slot()

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

        # ... and so can the first passage's demodulation frame, which
        # the segment generation cannot supply when this station is the
        # ring's first reference-altering element (see the method).
        self._seed_initial_demodulation_frame()

        # Feedforward cavity pre-fill: seed the initial antenna voltage from
        # the constant-current fill (optionally injection-matched), now that
        # omega_rf / t_rev are available. The PI controller, if attached, only
        # acts on the tracked turns after injection, so the fill stays a pure
        # feedforward (constant generator_current_bias) transient.
        if self.n_pretrack is not None:
            # DESIGN CLOCK, not omega_rf: the seed initialises the coarse
            # recursion, which is driven at _forward_segment_omega_design, i.e.
            # calc_omega_rf_design. Its no-beam fixed point is therefore
            # V* = -(R/Q) omega_design I_gen / lambda(omega_design), and
            # evaluating the fill at the actual (offset) frequency would miss
            # it by O(delta_omega_rf / omega) -- an injection transient the
            # PI would then have to burn off. omega_rf_design is the run-start
            # value of that forward-tracking frequency (which only exists once
            # tracking has started), and t_rev below reads the same clock, so
            # the whole call is clock-consistent.
            self._init_voltage = pretrack_fill_voltage(
                r_over_q=self.R_over_Q,
                q_l=self.Q_L,
                omega=self.omega_rf_design,
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

        Two steps: advance the coarse cells of the segment
        (``_circuit_track_cells``), and -- when the segment carries beam --
        resolve the resulting envelope onto the fine (profile) grid
        (``_resolve_fine_grid_voltage``), which is what the station readout
        is built from. A no-beam segment (a replayed backfill span) stops
        after the coarse cells: it has no beam current to resolve and its
        fine grid is never read.

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
            self._resolve_fine_grid_voltage(omega_input=omega_input)

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
        # The optional controller update recovers the per-step sampling time
        # from ``omega_times_dt / omega_input``; expose omega_input for it.
        self._omega_input_for_pi = omega_input
        # The compiled scan runs the control law inside the loop, so it needs
        # the controller to supply a compiled form of itself. A controller
        # that does not (and any custom implementation of the interface) is
        # driven cell-by-cell on the reference path instead. A span the
        # controller sits out (a no-beam backfill segment) never consults it,
        # so it can still take the compiled path.
        controller_runs = self._controller_active and not no_beam
        if self.use_numba_envelope_kernel and (
            not controller_runs or self._controller.supports_envelope_scan
        ):
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
        path defers here so the duplicate-and-warn handling is applied.

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
        # The step into this segment's first cell crosses a segment (or turn)
        # boundary, so it is the local time of that cell plus the PRECEDING
        # segment's unfilled tail -- a per-segment quantity, not the live
        # host scalar (see _preceding_segment_residual).
        preceding_residual = self._preceding_segment_residual(start_index)
        for rf_centers_idx in range(start_index, end_index):
            if rf_centers_idx == 0:
                if self._last_rf_centers_entry is None:
                    # First centre ever tracked: there is no previous centre to
                    # step from, so use the spacing to the next centre as the
                    # step proxy. That next centre must live in *this* segment,
                    # though. With fine sectioning the first (backfill)
                    # segment
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
                    delta_t = self._rf_centers[0] + preceding_residual
            elif rf_centers_idx == start_index:
                delta_t = self._rf_centers[rf_centers_idx] + preceding_residual
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
                # A coincident coarse point carries ZERO elapsed time, so the
                # state at this cell is exactly the previous one's:
                # V(t + 0) = V(t). Duplicate it (and the generator current
                # with it) instead of leaving the zeros prefill -- otherwise
                # the next cell would advance from v_prev = 0, destroying the
                # coherent cavity voltage and refilling it only over
                # tau = 2 Q_L / omega (hundreds of turns at Q_L ~ 1e6).
                # Duplication also keeps the two downstream readers of the
                # grid honest: reset_arrays carries the LAST cell into the
                # next turn, and the fine-grid solver takes its initial
                # condition from the FIRST forward cell.
                warnings.warn(
                    "double taking of rf_centers value, duplicating the "
                    "previous cell",
                    stacklevel=1,
                )
                if rf_centers_idx == 0:
                    # No predecessor in this grid: the state carried across
                    # the turn boundary is the previous cell.
                    self.antenna_voltage_gen_coarse_grid[0] = (
                        self._last_val_ant_voltage_gen
                    )
                    self.antenna_voltage_beam_coarse_grid[0] = (
                        self._last_val_ant_voltage_beam
                    )
                    self.generator_current_coarse_grid[0] = (
                        self._last_val_generator_current
                    )
                else:
                    self.antenna_voltage_gen_coarse_grid[rf_centers_idx] = (
                        self.antenna_voltage_gen_coarse_grid[
                            rf_centers_idx - 1
                        ]
                    )
                    self.antenna_voltage_beam_coarse_grid[rf_centers_idx] = (
                        self.antenna_voltage_beam_coarse_grid[
                            rf_centers_idx - 1
                        ]
                    )
                    self.generator_current_coarse_grid[rf_centers_idx] = (
                        self.generator_current_coarse_grid[rf_centers_idx - 1]
                    )
                # The demodulation-frame sum duplicates with its parts
                # (composed with THIS passage's rotation -- the sum is
                # derived from the component state, never propagated).
                self.antenna_voltage_coarse_grid[rf_centers_idx] = (
                    self._compose_coarse_sum(rf_centers_idx)
                )
                # The controller is deliberately NOT stepped: no time has
                # elapsed, so there is no new sample to regulate on.
                continue
            self.cavity_response(
                omega_input * delta_t,
                coarse_grid_index_to_update=rf_centers_idx,
                relative_detuning=self.delta_omega / omega_input,
                no_beam=no_beam,
            )

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
            # A coincident (zero) step needs the reference path, which is
            # the only one that duplicates the previous cell into it.
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
            voltage_gen_init = complex(self._last_val_ant_voltage_gen)
            voltage_beam_init = complex(self._last_val_ant_voltage_beam)
            # The carried demodulation-frame sum; only the beam-kick guard
            # reads it, and only on a segment that does NOT start at the
            # carried cell (skip_first) -- kept for the guard's signature.
            voltage_init = complex(self._last_val_ant_voltage)
            generator_current_init = complex(self._last_val_generator_current)
        else:
            voltage_gen_init = self.antenna_voltage_gen_coarse_grid[
                start_index - 1
            ]
            voltage_beam_init = self.antenna_voltage_beam_coarse_grid[
                start_index - 1
            ]
            voltage_init = self.antenna_voltage_coarse_grid[start_index - 1]
            generator_current_init = self.generator_current_coarse_grid[
                start_index - 1
            ]

        controller_active = self._controller_active and not no_beam
        # The controller owns its compiled law and marshals its own tuning and
        # state; this class passes the result straight through without
        # inspecting it. On a span the controller sits out, the neutral state
        # keeps the generator current constant.
        if controller_active:
            envelope_scan = self._controller.envelope_scan_kernel()
            controller_state = self._controller.envelope_scan_state()
            voltage_setpoint = complex(self.pi_setpoint)
        else:
            envelope_scan = envelope_pi_scan
            controller_state = inactive_controller_scan_state()
            # No regulation on this span, so the error is never formed and the
            # setpoint stays unevaluated (it may need the parent RF station).
            voltage_setpoint = 0.0 + 0.0j

        voltage_gen_out = np.empty(n_cells, dtype=np.complex128)
        voltage_beam_out = np.empty(n_cells, dtype=np.complex128)
        voltage_out = np.empty(n_cells, dtype=np.complex128)
        # Pre-fill with the current generator grid: the inactive (no-beam /
        # constant-current) path reads it as each cell's drive current, matching
        # cavity_response reading generator_current_coarse_grid[idx-1]; the
        # active path overwrites every cell with its PI output. astype copies,
        # so the kernel never mutates the grid before the write-back below.
        generator_current_out = self.generator_current_coarse_grid[
            start_index:end_index
        ].astype(np.complex128)

        delay_buffer, delay_head, integral, saturation_possible = (
            envelope_scan(
                voltage_multiplier,
                drive_weight,
                omega_times_dt,
                beam_current,
                voltage_gen_out,
                voltage_beam_out,
                voltage_out,
                generator_current_out,
                voltage_gen_init,
                voltage_beam_init,
                generator_current_init,
                float(self.R_over_Q),
                bool(self._generator_active),
                complex(self._generator_frame_rotation),
                complex(self._kick_frame_rotation),
                complex(self._pi_error_frame_rotation),
                controller_active,
                voltage_setpoint,
                float(omega_input),
                *controller_state,
            )
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

        self.antenna_voltage_beam_coarse_grid[start_index:end_index] = (
            voltage_beam_out
        )
        if self._generator_active:
            # The kernel writes the generator component only while it is
            # active; otherwise the grid keeps its zeros prefill, exactly
            # like the reference path, which skips the component update.
            self.antenna_voltage_gen_coarse_grid[start_index:end_index] = (
                voltage_gen_out
            )
        self.antenna_voltage_coarse_grid[start_index:end_index] = voltage_out
        # Commit the generator grid. Active: the PI outputs. Inactive: the
        # unchanged pre-filled values, i.e. a no-op vs the reference (which
        # leaves the generator grid untouched on the constant-current/no-beam
        # path). Only the controller's own state is synced when it actually ran.
        self.generator_current_coarse_grid[start_index:end_index] = (
            generator_current_out
        )
        if controller_active:
            self._controller.absorb_envelope_scan_state(
                (delay_buffer, delay_head, integral)
            )

        if not no_beam:
            self._check_beam_kicks(
                beam_current,
                omega_times_dt,
                voltage_init,
                voltage_out,
                skip_first=(start_index == 0),
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
            zero (coincident) step, which only the reference path handles.
        """
        n_cells = end_index - start_index
        delta_t = np.empty(n_cells, dtype=np.float64)
        if n_cells > 1:
            # Bulk cells: consecutive rf_centers differences (== the reference
            # ``else`` branch), bit-identical to the scalar subtraction.
            delta_t[1:] = np.diff(self._rf_centers[start_index:end_index])
        # Same per-segment boundary residual the reference loop uses; the two
        # paths MUST take it from the same source or the kernel-vs-python
        # byte-identity pin breaks.
        preceding_residual = self._preceding_segment_residual(start_index)
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
                delta_t[0] = self._rf_centers[0] + preceding_residual
        else:
            delta_t[0] = self._rf_centers[start_index] + preceding_residual
        rf_period = 2 * np.pi / omega_input
        tiny_negative = (delta_t > -1e-9 * rf_period) & (delta_t < 0)
        delta_t[tiny_negative] = 0.0
        # Any non-positive step is degenerate/invalid: a coincident (zero) step,
        # or a genuinely-negative one that violates ordering. Defer the whole
        # segment to the reference loop, which -- processing cells in order --
        # warns and duplicates the previous cell on a zero step and asserts on
        # a negative one, so its warnings and assertion message are reproduced
        # exactly rather than pre-empted by a vectorised assert here.
        if not (delta_t > 0).all():
            return None
        return delta_t

    def _advance_coarse_voltage(
        self,
        v_prev: complex,
        generator_current: complex,
        beam_current: complex,
        omega_times_dt: float,
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

        The step exponent and the propagator weights come from
        :mod:`~blond.physics.feedbacks.cavity_solvers`, beside the
        ``ForwardEulerValidityGuard`` that caps them, so this per-cell path
        and the vectorised :meth:`_kernel_step_multipliers` spell the
        recursion once.

        Parameters
        ----------
        v_prev
            Antenna voltage of the previous coarse sample [V].
        generator_current
            Generator current driving this step [A].
        beam_current
            Beam current of this step [A].
        omega_times_dt
            RF phase advanced in this step [rad], i.e. ``omega * dt``.
        relative_detuning
            Detuning normalised to the step frequency
            (``delta_omega / omega``), so ``delta_omega * dt =
            relative_detuning * omega_times_dt``.

        Returns
        -------
        complex
            The advanced antenna voltage [V].
        """
        drive = (
            self.R_over_Q
            * omega_times_dt
            * (generator_current - 0.5 * beam_current)
        )
        # L = lambda * dt (dimensionless growth exponent for this step).
        step_exponent = coarse_step_exponent(
            omega_times_dt, self.Q_L, relative_detuning
        )
        if not self._exponential_coarse_solver_enable:
            return v_prev * euler_voltage_multiplier(step_exponent) + drive
        # Exact exponential propagator; the drive weight (e^L - 1) / L stays
        # accurate (-> 1) as L -> 0 and is guarded at the exact zero, which
        # this scalar path -- unlike the vectorised one -- can be handed.
        growth = exponential_voltage_multiplier(step_exponent)
        drive_weight = exponential_drive_weight(step_exponent)
        return v_prev * growth + drive * drive_weight

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
        The arithmetic itself is the shared one of
        :mod:`~blond.physics.feedbacks.cavity_solvers`
        (:func:`~blond.physics.feedbacks.cavity_solvers.coarse_step_exponent`
        and the propagator weights), so this vectorised path and the per-cell
        :meth:`_advance_coarse_voltage` cannot drift apart.

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
        step_exponent = coarse_step_exponent(
            omega_times_dt, self.Q_L, relative_detuning
        )
        if self._exponential_coarse_solver_enable:
            voltage_multiplier = exponential_voltage_multiplier(step_exponent)
            # omega_times_dt > 0, so step_exponent != 0 and (e^L - 1) / L is
            # well defined -- the weight's zero guard is never reached here.
            drive_weight = exponential_drive_weight(step_exponent)
        else:
            voltage_multiplier = euler_voltage_multiplier(step_exponent)
            drive_weight = np.ones(
                omega_times_dt.shape[0], dtype=np.complex128
            )
        return voltage_multiplier, drive_weight

    def _compose_coarse_sum(self, coarse_grid_index: int) -> complex:
        """
        Compose the demodulation-frame sum at one coarse-grid index.

        ``V_beam + V_gen * generator frame rotation``: the beam component
        already lives in the demodulation frame, the design-anchored
        generator component is rotated into it with this passage's
        rotation (see :meth:`_update_frame_rotations`). While the
        generator component is inactive the sum IS the beam component --
        assigned, not added, so an undriven feedback stays bit-identical
        to the former single-state recursion.

        Parameters
        ----------
        coarse_grid_index
            Coarse-grid index to compose; both component arrays must
            already hold this cell.

        Returns
        -------
        composed_sum
            The demodulation-frame antenna voltage at that cell [V].
        """
        voltage_beam = self.antenna_voltage_beam_coarse_grid[coarse_grid_index]
        if not self._generator_active:
            return voltage_beam
        return voltage_beam + (
            self.antenna_voltage_gen_coarse_grid[coarse_grid_index]
            * self._generator_frame_rotation
        )

    def cavity_response(
        self,
        omega_times_dt: float,
        coarse_grid_index_to_update: int,
        relative_detuning: float,
        no_beam: bool = False,
    ):
        """
        Calculate antenna voltage on the coarse grid for a specific index.

        Advances the two source-split components (the envelope ODE is
        linear, so running the same propagator once per source is exact
        superposition): the beam-sourced component with the generator
        current pinned to zero, the generator-sourced component with the
        beam current pinned to zero -- then composes the
        demodulation-frame sum via ``_compose_coarse_sum``.

        Parameters
        ----------
        omega_times_dt
            RF phase advanced in this step [rad], i.e. ``omega * dt``.
        coarse_grid_index_to_update
            Coarse grid index to update.
        relative_detuning
            Detuning normalized to the current RF frequency.
        no_beam
            If no beam is present, the beam current is set to 0.
        """
        index = coarse_grid_index_to_update
        if index != 0:
            if no_beam:
                beam_current = 0
            else:
                forward_offset = (
                    len(self._rf_centers) - self._rf_centers_lengths[-1]
                )
                beam_current = self.beam_current_forward_coarse_grid[
                    index - forward_offset
                ]
            self._check_beam_kick_magnitude(
                beam_current=beam_current,
                omega_times_dt=omega_times_dt,
                previous_voltage=self.antenna_voltage_coarse_grid[index - 1],
            )
            voltage_gen_prev = self.antenna_voltage_gen_coarse_grid[index - 1]
            voltage_beam_prev = self.antenna_voltage_beam_coarse_grid[
                index - 1
            ]
            generator_current = self.generator_current_coarse_grid[index - 1]
        else:
            voltage_gen_prev = self._last_val_ant_voltage_gen
            voltage_beam_prev = self._last_val_ant_voltage_beam
            generator_current = self._last_val_generator_current
            beam_current = self._last_val_beam_current
        # Beam-sourced component: the former recursion with the generator
        # current pinned to (0 + 0j) -- bit-identical to the old single
        # state for an undriven feedback (whose generator grid is zero).
        self.antenna_voltage_beam_coarse_grid[index] = (
            self._advance_coarse_voltage(
                v_prev=voltage_beam_prev,
                generator_current=(0.0 + 0.0j),
                beam_current=beam_current,
                omega_times_dt=omega_times_dt,
                relative_detuning=relative_detuning,
            )
        )
        if self._generator_active:
            # Generator-sourced component: same propagator, beam current
            # pinned to (0 + 0j).
            self.antenna_voltage_gen_coarse_grid[index] = (
                self._advance_coarse_voltage(
                    v_prev=voltage_gen_prev,
                    generator_current=generator_current,
                    beam_current=(0.0 + 0.0j),
                    omega_times_dt=omega_times_dt,
                    relative_detuning=relative_detuning,
                )
            )
        self.antenna_voltage_coarse_grid[index] = self._compose_coarse_sum(
            index
        )

        # With the PI control active, regulate the generator current of this
        # coarse-grid index from the antenna-voltage error just computed; it
        # then drives the next step. Inactive by default (constant current).
        # Only on the real forward pass (not the no_beam backfill
        # reconstruction segments): the backfill cells carry a per-segment
        # frame phase (corrected only on the last sample), so stepping the
        # controller there would integrate frame-rotated errors and
        # double-advance its delay line / integrator. Single-section rings
        # have no backfill segments, so this is a no-op there.
        if self._controller_active and not no_beam:
            self._update_generator_current(
                omega_times_dt=omega_times_dt,
                coarse_grid_index_to_update=coarse_grid_index_to_update,
            )

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

    def _check_beam_kick_magnitude(
        self,
        beam_current: complex | float | int,
        omega_times_dt: float | int,
        previous_voltage: complex | float | int,
    ) -> None:
        """
        Hand one cell's beam kick to the forward-Euler validity guard.

        Parameters
        ----------
        beam_current
            Beam current sample used for this step [A].
        omega_times_dt
            RF phase advanced in this step [rad], i.e. ``omega * dt``.
        previous_voltage
            Antenna voltage of the previous coarse-grid step, which the
            kick is added to/subtracted from.

        See Also
        --------
        blond.physics.feedbacks.cavity_solvers.ForwardEulerValidityGuard.check_beam_kick_magnitude : The thresholds, the messages and what an excessive kick means.
        """
        self._euler_guard.check_beam_kick_magnitude(
            beam_current,
            omega_times_dt,
            previous_voltage,
            self.R_over_Q,
        )

    def _check_beam_kicks(
        self,
        beam_current: NumpyArray,
        omega_times_dt: NumpyArray,
        voltage_init: complex,
        voltage_out: NumpyArray,
        skip_first: bool,
    ) -> None:
        """
        Hand a whole kernel segment's beam kicks to the guard at once.

        Parameters
        ----------
        beam_current
            Per-cell beam current of the segment.
        omega_times_dt
            Per-cell RF phase advanced in one step [rad], i.e.
            ``omega * dt``.
        voltage_init
            Antenna voltage preceding the first cell.
        voltage_out
            Per-cell antenna voltage just computed for the segment.
        skip_first
            Whether to skip the first cell (the carried ``rf_centers`` index 0,
            which the reference never checks).

        See Also
        --------
        blond.physics.feedbacks.cavity_solvers.ForwardEulerValidityGuard.check_beam_kicks : The vectorised sweep, and why it reproduces the per-cell ordering exactly.
        """
        self._euler_guard.check_beam_kicks(
            beam_current,
            omega_times_dt,
            voltage_init,
            voltage_out,
            skip_first,
            self.R_over_Q,
        )

    def reset_arrays(self, n_backfill_cells: int = 0) -> None:
        """
        Reset the coarse grids for a new turn, carrying the last values over.

        The antenna voltage is carried as its two source-split components
        (``_last_val_ant_voltage_gen`` / ``_last_val_ant_voltage_beam``,
        the propagated state) plus the composed demodulation-frame sum
        (``_last_val_ant_voltage``, diagnostics and the coincident
        first-cell duplication). On the very first turn the initial (or
        pre-fill) voltage seeds the generator component: it is a
        generator-established, design-anchored field.

        The generator grid is seeded with the feedforward bias, except over
        the leading ``n_backfill_cells`` no-beam backfill-reconstruction
        cells,
        which are seeded with the last commanded generator current (a
        zero-order hold). Those cells replay an interval that has already
        elapsed and during which the loop issued no new command: the
        generator kept running at whatever it was last told, it did not snap
        back to the feedforward value. :meth:`cavity_response` already drives
        the *first* backfill cell from ``_last_val_generator_current``; this
        extends the same held value over the rest of the span. Without a
        controller the held value *is* the bias, so the constant-current path
        is bit-unchanged.

        Parameters
        ----------
        n_backfill_cells
            Number of leading coarse cells belonging to this turn's no-beam
            backfill segments. 0 (the default) leaves the whole grid at the
            bias, which is what a grid without backfill segments gets.
        """
        if self.antenna_voltage_coarse_grid is None:
            # First turn: the initial (or pre-fill) voltage is a
            # generator-established field, so it seeds the design-anchored
            # generator component; the beam component starts empty.
            self._last_val_ant_voltage = self._init_voltage
            self._last_val_ant_voltage_gen = self._init_voltage
            self._last_val_ant_voltage_beam = 0.0 + 0.0j
        else:
            self._last_val_ant_voltage = self.antenna_voltage_coarse_grid[-1]
            self._last_val_ant_voltage_gen = (
                self.antenna_voltage_gen_coarse_grid[-1]
            )
            self._last_val_ant_voltage_beam = (
                self.antenna_voltage_beam_coarse_grid[-1]
            )
        self.antenna_voltage_coarse_grid = np.zeros(
            len(self._rf_centers), dtype=np.complex128
        )
        self.antenna_voltage_gen_coarse_grid = np.zeros(
            len(self._rf_centers), dtype=np.complex128
        )
        self.antenna_voltage_beam_coarse_grid = np.zeros(
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
        if n_backfill_cells > 0:
            self.generator_current_coarse_grid[:n_backfill_cells] = (
                self._last_val_generator_current
            )
        # Whether the generator-sourced component carries any signal this
        # turn. Everything that can source it is checked: an attached
        # controller, the feedforward bias seeding the grid, the held
        # (zero-order-hold) command over the backfill cells and the
        # carried component voltage. While False, the component update
        # and every composition multiply are skipped -- the sum is then
        # assigned from the beam component, keeping an undriven feedback
        # bit-identical to the former single-state recursion.
        self._generator_active = (
            self._controller is not None
            or self._generator_current_bias != 0
            or self._last_val_generator_current != 0
            or self._last_val_ant_voltage_gen != 0
        )

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Track the feedback for one turn.

        This method is the per-turn call-order declaration of this element
        (the idiom of
        :meth:`~blond.physics.cavities.SingleHarmonicRFStation._track`): it
        does no work itself, it only names the phases in order. Where a
        phase depends on a value another phase produced, that value is
        *returned* and *passed*, not left on ``self`` -- so the argument
        lists below are the dependency graph, and the ordering cannot be
        broken by reshuffling the calls.

        The two constraints that cannot be expressed that way
        (:meth:`reset_arrays` sizing the coarse state before any
        :meth:`circuit_track`, and ``_carrier_slip_gap`` being complete
        before :meth:`calculate_rf_beam_current_partial` reads it off the
        instance) are stated in the docstrings of
        :meth:`_replay_backfill_span` and :meth:`_track_forward_span`, and
        the first of the two is additionally asserted.

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
        self._guard_simultaneous_passage(beam=beam)
        self._kick_clock_slip_gap = self._carrier_slip_gap_at_passage(
            beam=beam
        )

        span = self._rebuild_per_turn_grid(beam=beam)
        self._carrier_slip_gap = (
            self._kick_clock_slip_gap
            + self._accumulate_registration_phase(
                n_backfill_centers=span.n_backfill_centers
            )
        )
        self._update_frame_rotations()

        self._replay_backfill_span(n_backfill_centers=span.n_backfill_centers)

        if self._grid_only_no_correction:
            self._write_no_correction_readout()
            return

        self._track_forward_span(beam=beam, span=span)
        self._write_station_readout(carrier_slip_gap=self._carrier_slip_gap)

    def _guard_simultaneous_passage(self, beam: BeamBaseClass) -> None:
        """
        Reject a coincident counter-rotating passage; record this one.

        Parameters
        ----------
        beam
            Beam passing this station now.

        Raises
        ------
        NotImplementedError
            When two counter-rotating beams pass this station simultaneously
            (the station sits at a meeting azimuth of the two beams).

        Notes
        -----
        ORDERING: the two ``_last_track_*`` writes at the end are the record
        the NEXT passage compares itself against, so they must follow the
        comparison -- they are the tail of this very method for that reason.
        Called first in :meth:`_track` so that a rejected passage cannot
        leave a half-rebuilt grid behind.
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
                "with the half-drift / station / half-drift layout). The "
                "MultiPassResonatorSolver wakefield with "
                "allow_delta_t_zero=True runs such a station, but its "
                "coincident kicks are wrong (0.5 and 1.5 times the "
                "correct mutual term, depending on track order), so it is "
                "not a substitute -- it warns about exactly this."
            )
        self._last_track_arrival_time = beam.reference.time
        self._last_track_is_counter_rotating = beam.is_counter_rotating

    def _carrier_slip_gap_at_passage(self, beam: BeamBaseClass) -> float:
        """
        Live tail of the RF-frequency-offset phase slip at this passage.

        Parameters
        ----------
        beam
            Beam passing this station now.

        Returns
        -------
        carrier_slip_gap
            ``delta_omega_rf * (t_passage - station kick-clock tick)`` [rad];
            exactly ``0.0`` without an RF-frequency offset.

        Notes
        -----
        ORDERING: the gap is *returned*, not assigned, so that the caller's
        ``self._kick_clock_slip_gap = ...`` makes visible that it is RESET
        at every passage rather than accumulated. ``_carrier_slip_gap`` is
        then formed as this gap plus the multi-section registration phase
        (see :meth:`_accumulate_registration_phase`); the two constituents
        stay separately available because the generator-component frame
        rotation needs the kick-clock part with the station clock on top
        (see :meth:`_update_frame_rotations`).
        """
        # Live tail of the RF-frequency-offset phase slip: the station's
        # kick clock (delta_phi_rf) is accumulated only at the END of each
        # station track (a blond2-era convention this code builds on), so
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
        return (
            0.0
            if station_clock_last is None
            else self.delta_omega_rf
            * (beam.reference.time - station_clock_last)
        )

    def _rebuild_per_turn_grid(self, beam: BeamBaseClass) -> PerTurnGridSpan:
        """
        Rebuild this passage's coarse grid and size the coarse state.

        Parameters
        ----------
        beam
            Beam passing this station now; supplies the reference clock the
            grid is generated against.

        Returns
        -------
        span
            The per-turn span: the backfill / forward centre counts and the
            residual snapshot taken before the forward generation.

        Notes
        -----
        ORDERING: :meth:`reset_arrays` is the last statement before the
        return, so it can neither precede the grid generation it takes its
        size from (it also re-snapshots the previous turn's last antenna
        voltage / generator current) nor follow any :meth:`circuit_track`.
        """
        self._close_previous_turn_grid()

        self._generate_backfill_segments_if_due(beam=beam)

        n_backfill_centers = len(self._rf_centers)

        # ORDERING: snapshot the residual HERE, between the two generations.
        # The forward generation below overwrites the instance scalar, and
        # the demodulation needs the backfill-span value -- see
        # PerTurnGridSpan.residual_from_backfill_span.
        residual_from_backfill_span = (
            self._residual_time_last_rf_centers_calculation
        )

        self.calculate_rf_centers_for_forward_direction(beam=beam)

        # The flat rf_centers / rf_centers_lengths arrays are derived from
        # _segments; assert they stayed consistent after this turn's
        # generation. This is a per-turn integrity check with no effect on the
        # result, so it runs only under ``validate_grid_each_turn`` (it walks
        # the whole grid every turn otherwise).
        if self._validate_grid_each_turn:
            self._validate_grid()
            # The demodulation frame of the forward segment and the coarse
            # step into its first cell are the same physical quantity -- the
            # tail of the segment preceding the forward one. They used to be
            # derived independently (snapshot vs live scalar) and silently
            # disagreed; tie them together so they cannot drift apart again.
            assert (
                self._preceding_segment_residual(n_backfill_centers)
                == residual_from_backfill_span
            ), (
                "forward-segment boundary residual "
                f"{self._preceding_segment_residual(n_backfill_centers)} != "
                f"demodulation frame {residual_from_backfill_span}"
            )

        # Coincidence tolerance for the simultaneous-passage guard above:
        # one coarse-cell width, taken from the last two grid centers.
        # CORRECTNESS RELIES on the >=2-centres-per-segment invariant
        # enforced in RFCenterSegment.__post_init__: rf_centers are
        # segment-LOCAL times, so only that invariant guarantees both
        # entries lie inside the forward segment -- a single-centre forward
        # segment would make this difference cross the segment boundary,
        # go negative and silently disarm the guard. Do not relax the
        # invariant without revisiting this computation.
        min_centers_for_cell_width = 2
        if len(self._rf_centers) >= min_centers_for_cell_width:
            self._last_forward_cell_width = float(
                self._rf_centers[-1] - self._rf_centers[-2]
            )

        self.reset_arrays(n_backfill_cells=n_backfill_centers)

        return PerTurnGridSpan(
            n_backfill_centers=n_backfill_centers,
            n_forward_centers=len(self._rf_centers) - n_backfill_centers,
            residual_from_backfill_span=residual_from_backfill_span,
        )

    def _replay_backfill_span(self, n_backfill_centers: int) -> None:
        """
        Re-run this passage's elapsed backfill segments with no beam.

        Parameters
        ----------
        n_backfill_centers
            ``PerTurnGridSpan.n_backfill_centers`` of this passage; ``0``
            makes the replay a no-op.

        Notes
        -----
        PRECONDITION: :meth:`reset_arrays` must have sized the coarse state
        to the freshly generated grid -- the very first thing a
        :meth:`circuit_track` does is index those arrays. That is what
        :meth:`_rebuild_per_turn_grid` guarantees by calling
        :meth:`reset_arrays` last, and what the ``assert`` below re-checks
        per turn (stripped by ``python -O``, the repo's validation idiom).

        The walk iterates the :attr:`_segments` records themselves, taking
        each segment's own frequency (``RFCenterSegment.omega``) and its own
        length. The grid is rebuilt from scratch every passage
        (``_close_previous_turn_grid`` clears it), the backfill generation
        appends exactly one segment per entry of
        ``_backfill_time_array`` -- whose companion
        ``_backfill_segment_omega_design_list`` is masked with it under the
        single mask of ``_unify_same_frequency_time_points_backfill`` -- and the
        forward generation then appends exactly one more. So the backfill
        segments are ``_segments[:-1]``, and their frequencies and lengths
        are the ones the flat parallel arrays used to be sliced for.
        """
        assert (
            self.antenna_voltage_coarse_grid is not None
            and self.generator_current_coarse_grid is not None
            and len(self.antenna_voltage_coarse_grid) == len(self._rf_centers)
            and len(self.generator_current_coarse_grid)
            == len(self._rf_centers)
        ), "reset_arrays() must size the coarse state before circuit_track"

        # Only walk the backfill segments when this turn actually generated
        # centres for them (n_backfill_centers > 0). Historically this loop
        # ran off a *stale* backfill omega list: for a single section the
        # list from turn 0 is never refreshed, so without the gate the loop
        # re-ran the ENTIRE forward grid every turn at the frozen turn-0
        # frequency (no_beam) before the demodulation and the real forward
        # pass. The envelope overwrite was recomputed identically by the
        # real pass, but under a ramp the spurious pass corrupted the
        # sub-stepped demodulation frame by -(turn+1) * 2 pi S per turn and
        # stepped an attached controller once per turn on garbage errors.
        # Walking the segment list cannot go stale that way (it is rebuilt
        # every passage), and with the >=2-centres-per-segment invariant
        # (RFCenterSegment.__post_init__) n_backfill_centers == 0 means
        # there are no backfill segments at all, so the gate merely skips
        # an empty loop.
        if n_backfill_centers > 0:
            start_index = 0
            for segment in self._segments[:-1]:
                end_index = start_index + len(segment)

                self.circuit_track(
                    omega_input=segment.omega,
                    start_index=start_index,
                    end_index=end_index,
                    no_beam=True,
                )
                start_index = end_index

    def _accumulate_registration_phase(self, n_backfill_centers: int) -> float:
        """
        Accumulate the multi-section grid-vs-carrier registration phase.

        Parameters
        ----------
        n_backfill_centers
            ``PerTurnGridSpan.n_backfill_centers`` of this passage; a passage
            without backfill segments contributes nothing.

        Returns
        -------
        grid_carrier_phase
            The RUNNING TOTAL ``sum_k (omega_k - omega_0) T_seg,k`` [rad]
            after this passage's contribution -- not the increment. Exactly
            ``+0.0`` for a single section and for an unaccelerated ring.

        Notes
        -----
        ORDERING: must run after :meth:`_rebuild_per_turn_grid` (it reads
        the backfill segment list that call generated) and before
        :meth:`_update_frame_rotations` -- the per-passage frame rotations
        fold the returned total in -- and hence before every
        :meth:`circuit_track` of the passage, whose sum composition uses
        those rotations; :meth:`_track_forward_span`'s demodulation then
        subtracts the identical total via ``_carrier_slip_gap``. Like the
        backfill replay it reads the segment records themselves --
        ``RFCenterSegment.omega`` is omega_k and
        ``RFCenterSegment.duration`` is T_seg,k.
        """
        # Multi-section grid-vs-carrier registration phase. A multi-section
        # passage builds its coarse grid piecewise: each backfill segment k
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
        #
        # The n_backfill_centers > 0 gate RELIES on the >=2-centres-per-
        # segment invariant enforced in RFCenterSegment.__post_init__:
        # only that invariant makes n_backfill_centers == 0 equivalent to
        # "no backfill segments at all" (where skipping is correct) -- an
        # all-empty backfill span would otherwise permanently drop its Psi
        # from the running total. Do not relax the invariant without
        # revisiting this gate.
        if self._n_rf_stations_in_ring > 1 and n_backfill_centers > 0:
            # Same records the replay walks: the backfill segments are
            # ``_segments[:-1]``, each carrying its own omega_k and the
            # T_seg,k it was generated over (see _replay_backfill_span).
            backfill_segments = self._segments[:-1]
            self._grid_carrier_phase += float(
                np.sum(
                    (
                        np.array(
                            [segment.omega for segment in backfill_segments]
                        )
                        - self._forward_segment_omega_design
                    )
                    * np.array(
                        [segment.duration for segment in backfill_segments]
                    )
                )
            )
        # Exactly +0.0 for a single section and for an unaccelerated ring, so
        # both stay bit-identical.
        return self._grid_carrier_phase

    def _update_frame_rotations(self) -> None:
        r"""
        Compute this passage's component frame rotations.

        The coarse state is source-split (the envelope ODE is linear, so
        superposition is exact): the BEAM component lives in the
        demodulation frame, the GENERATOR component is natively anchored
        to the piecewise design clock (its current is injected as a
        constant per segment at each segment's own design frequency --
        samples of the design program). Composing the demodulation-frame
        sum therefore rotates the generator component by

        .. math::
            e^{-i(\Delta\phi_\mathsf{rf} + \mathrm{gap} + \Psi)}

        (station kick clock + live kick-clock gap + registration phase
        ``Psi``): the readout later adds ``gap + Psi`` back and the
        station adds ``delta_phi_rf`` through ``phi_rf``, so the
        generator component nets to its design-clock phase -- under an
        RF-frequency offset it appears at MINUS the kick-clock slip
        relative to the actual RF, the physical walk-off of a
        design-locked drive (see :meth:`_write_station_readout`).

        The kick-frame rotation ``exp(+i (gap + Psi))`` rotates the
        demodulation-frame sum into the frame of the applied kick; the PI
        error is formed there, so the loop regulates the voltage the
        station actually applies.

        The PI-error rotation ``exp(+i delta_phi_rf)`` then takes that
        error into the ACTUATOR frame. The controller returns a generator
        current, which drives the design-anchored generator component, so
        ``d(V_kick) / d(I_gen)`` carries the composition's
        ``exp(-i delta_phi_rf)``; rotating the error back cancels it, and
        the open-loop gain stays real instead of turning with the station
        clock. Note the ``gap`` and ``Psi`` halves cancel between the two
        rotations, which is why this third one uses ``delta_phi_rf``
        alone.

        The first two are exactly ``1 + 0j`` without an RF-frequency
        offset and without multi-section acceleration; the third is
        exactly ``1 + 0j`` whenever ``delta_phi_rf`` is zero, independently
        of ``gap`` and ``Psi`` (the zero short-circuits keep the unrotated
        path free of ``exp`` sign dust).

        Notes
        -----
        ORDERING: needs ``delta_phi_rf`` (per-passage station clock) and
        the completed ``_carrier_slip_gap`` of this passage; must precede
        every :meth:`circuit_track` of the passage, whose per-cell sum
        composition and PI error read the rotations off the instance.
        """
        total_generator_slip = self.delta_phi_rf + self._carrier_slip_gap
        self._generator_frame_rotation = (
            1.0 + 0.0j
            if total_generator_slip == 0.0
            else complex(np.exp(-1j * total_generator_slip))
        )
        self._kick_frame_rotation = (
            1.0 + 0.0j
            if self._carrier_slip_gap == 0.0
            else complex(np.exp(1j * self._carrier_slip_gap))
        )
        # Actuator frame of the PI error. The error is read out in the
        # KICK frame, but the controller's output is a generator
        # current, which drives the DESIGN-anchored generator
        # component: the composition multiplies it by
        # ``_generator_frame_rotation``, so ``d(V_kick) / d(I_gen)``
        # carries ``exp(-i delta_phi_rf)``. Handing the kick-frame
        # error straight to the controller would therefore rotate the
        # open-loop gain by that factor, which grows without bound
        # while an RF-frequency offset is applied (the proportional
        # path's sign inverts past |delta_phi_rf| = pi/2). Rotating
        # the error back cancels it exactly. Unity, so bit-identical,
        # whenever no RF-frequency offset ever acted.
        self._pi_error_frame_rotation = (
            1.0 + 0.0j
            if self.delta_phi_rf == 0.0
            else complex(np.exp(1j * self.delta_phi_rf))
        )

    def _write_no_correction_readout(self) -> None:
        """
        Write the neutral readout: unit gain, zero phase.

        Notes
        -----
        This is the readout of the ``grid_only_no_correction`` mode, and
        it means the feedback applies NO correction: unit relative
        voltage and zero phase make the parent station's
        ``calc_gap_voltage_with_feedbacks`` reduce to
        ``voltage * sin(omega_rf * ts + phi_rf)``, i.e. the unperturbed
        RF wave. The mode stops the turn here: the caller returns from
        :meth:`_track` right after this call, so neither the beam-current
        demodulation nor the forward pass runs.
        """
        self.relative_voltage_correction = np.ones_like(self.profile.hist_x)
        self.phase_correction = np.zeros_like(self.profile.hist_x)

    def _track_forward_span(
        self, beam: BeamBaseClass, span: PerTurnGridSpan
    ) -> None:
        """
        Demodulate the beam current and advance the forward segment.

        Parameters
        ----------
        beam
            Beam passing this station now.
        span
            The span :meth:`_rebuild_per_turn_grid` returned for this
            passage. Its ``residual_from_backfill_span`` is the demodulation
            frame; re-reading
            ``_residual_time_last_rf_centers_calculation`` off the instance
            instead would yield the forward-overwritten value and silently
            shift that frame.

        Notes
        -----
        PRECONDITION: ``self._carrier_slip_gap`` must already include
        ``_grid_carrier_phase`` --
        :meth:`calculate_rf_beam_current_partial` reads the attribute
        directly (``carrier_phase_offset = -(delta_phi_rf +
        _carrier_slip_gap)``) and :meth:`_write_station_readout` adds the
        identical total back, or the demodulation/readout chain no longer
        closes.
        """
        # default behavior
        self.calculate_rf_beam_current_partial(
            beam=beam,
            n_points=span.n_forward_centers,
            remaining_delta_t_from_backfill=span.residual_from_backfill_span,
        )

        self.circuit_track(
            omega_input=self._forward_segment_omega_design,
            no_beam=False,
            start_index=len(self._rf_centers) - span.n_forward_centers,
            end_index=len(self._rf_centers),
        )  # for all rf_centers

    def _write_station_readout(self, carrier_slip_gap: float) -> None:
        r"""
        Write ``relative_voltage_correction`` and ``phase_correction``.

        Parameters
        ----------
        carrier_slip_gap
            The accumulated actual-RF phase [rad] the demodulation of this
            passage subtracted, i.e. ``self._carrier_slip_gap``. Passed in
            rather than re-read so the readout provably adds back the very
            same total (see :meth:`_track_forward_span`).

        Notes
        -----
        ORDERING: must run after :meth:`_track_forward_span`, which fills
        the fine-grid antenna voltage this readout converts.

        **Readout composition (per-component anchoring).** The fine-grid
        envelope this readout converts is the demodulation-frame sum

        .. math::
            V = V_\mathrm{beam}
                + V_\mathrm{gen}\,
                  e^{-i(\Delta\phi_\mathsf{rf} + g + \Psi)},

        with ``g`` the live kick-clock gap, ``Psi`` the multi-section
        registration phase (``carrier_slip_gap = g + Psi``) and
        ``delta_phi_rf`` the station kick clock. The station applies
        ``sin(omega_rf ts + phi_rf_design + delta_phi_rf +
        phase_correction)`` with ``phase_correction = angle(V) +
        carrier_slip_gap``, so each component nets, relative to the
        design carrier:

        - beam component: ``angle(V_beam) + delta_phi_rf + g + Psi`` --
          exactly the total its demodulation subtracted
          (``carrier_phase_offset = -(delta_phi_rf + g + Psi)``, see
          :meth:`calculate_rf_beam_current_partial`); the chain closes
          for every carried deposit, byte-for-byte as before the split;
        - generator component: ``angle(V_gen) + 0`` -- design-locked, as
          the klystron drive follows the design frequency. Relative to
          the ACTUAL RF (which leads the design carrier by the kick-clock
          slip ``delta_phi_rf + g``) the driven field therefore appears at
          MINUS that slip: the physical walk-off of a design-locked drive
          under an RF-frequency offset. Without an offset and without
          multi-section acceleration every phase above is zero and a
          driven, beam-free cavity on its setpoint reads out
          ``phase_correction == 0`` -- the feedback is a no-op.
        """
        # Convert to amplitude and phase
        self.relative_voltage_correction, alpha_sum = cartesian_to_polar(
            IQ_vector=self.antenna_voltage_fine_grid,
        )

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        # Guard the zero: with no programmed voltage at this harmonic there
        # is nothing to correct RELATIVE to, and the division would make the
        # correction inf/NaN. calc_gap_voltage_with_feedbacks multiplies the
        # same zero back in, so the harmonic's contribution should simply be
        # zero -- a correction factor of 0 reproduces that exactly, whereas
        # NaN poisons the whole summed gap voltage and every particle kick
        # taken from it.
        parent_voltage = self.get_voltage_from_parent_rf_station()
        if parent_voltage == 0.0:
            self.relative_voltage_correction = np.zeros_like(
                self.relative_voltage_correction
            )
        else:
            self.relative_voltage_correction /= parent_voltage
        # The station applies its (end-of-track-lagged) kick clock via
        # phi_rf; adding the live slip gap here completes the readout to
        # the exact accumulated actual-RF phase at this passage -- the
        # same total the demodulation subtracted (see
        # calculate_rf_beam_current_partial). Exactly +0.0 without an
        # RF-frequency offset.
        self.phase_correction = (
            alpha_sum
            - np.mean(np.angle(self.station_voltage_coarse_grid))
            + carrier_slip_gap
        )

    def _check_fine_grid_initial_condition_is_causal(
        self, init_beam_time: float
    ) -> None:
        """
        Reject a fine-grid window that starts before its own seed.

        The fine solve is seeded with the coarse envelope at the first
        forward coarse centre and then integrates the beam current over
        ``[profile.cut_left, profile.cut_right]``. Both times are
        segment-local (origin ``beam.reference.time`` at this passage --
        the same frame the ``np.interp`` of the generator current onto
        ``profile.hist_x`` relies on), so the seed is causal only when the
        centre it comes from is not later than the start of the window:

        ``0 < rf_centers_forward[0] <= profile.cut_left < t_first_charge``

        Checked EVERY turn, not once at setup: the first forward centre
        depends on the design frequency and on the residual carried from the
        previous passage, both of which move turn to turn under acceleration
        and sub-stepping, and ``cut_left`` is itself settable.

        Gated on ``beam_current_fine_grid`` -- the current the fine solve
        actually consumes, filled by
        :meth:`calculate_rf_beam_current_partial` immediately before
        ``circuit_track`` in :meth:`_track_forward_span`. A charge-free
        window has nothing to be causal about, and grid-geometry tests
        legitimately drive throwaway empty profiles through this path. It is
        the right gate rather than ``profile.hist``, which the direct-drive
        tests never slice even when they hand the fine grid a beam current.

        This guard sits BESIDE ``forbid_charge_in_first_coarse_cell`` (which
        keeps the seeding cell itself charge-free); neither subsumes the
        other.

        Parameters
        ----------
        init_beam_time
            ``profile.cut_left``, the left edge of the fine grid [s].

        Raises
        ------
        ValueError
            If the window carries charge and starts before the first
            forward coarse centre.
        """
        beam_current = self.beam_current_fine_grid
        if beam_current is None or not np.any(beam_current):
            return

        first_forward_center = float(
            self._rf_centers[-self._rf_centers_lengths[-1] :][0]
        )
        if init_beam_time >= first_forward_center:
            return

        raise ValueError(
            f"Acausal fine-grid initial condition: profile.cut_left "
            f"({init_beam_time:.6g} s) lies before the first forward coarse "
            f"centre ({first_forward_center:.6g} s), which is the coarse "
            "sample the fine solve is seeded from, while the window carries "
            "charge. The seed would then postdate the start of the interval "
            "it initialises and the beam current would be integrated twice. "
            "Move the profile window right, to "
            "cut_left >= max(t_rf / 2, sampling_time_coarse)."
        )

    def _resolve_fine_grid_voltage(self, omega_input: float) -> None:
        """
        Resolve this passage's forward segment onto the fine (profile) grid.

        The second half of :meth:`circuit_track`, run only when the segment
        carries beam. The coarse recursion has just filled the forward
        segment; this takes its FIRST forward cell as the initial condition,
        interpolates the forward generator current onto ``profile.hist_x``
        and hands both to :meth:`cavity_response_fine`, which integrates the
        fine-grid antenna voltage the station readout is built from.

        Writes ``generator_current_fine_grid`` (the interpolation) and, via
        :meth:`cavity_response_fine`, ``antenna_voltage_fine_grid``.

        Parameters
        ----------
        omega_input
            Frequency of the segment just tracked [rad/s]; sets both the
            fine-grid step phase ``omega * profile.hist_step`` and the
            normalisation of the cavity detuning.

        Raises
        ------
        ValueError
            If the fine-grid window starts before the coarse centre it is
            seeded from; see
            ``_check_fine_grid_initial_condition_is_causal``.
        """
        init_beam_time = self.profile.cut_left
        assert init_beam_time > 0, (
            f"{init_beam_time=} has to be > 0, shift profile."
        )

        self._check_fine_grid_initial_condition_is_causal(init_beam_time)

        # The last _rf_centers_lengths entry is the forward segment, so
        # the slice below is this passage's forward grid.
        # Index [0] -- the FIRST forward coarse centre -- is the correct
        # initial condition, and must stay. Do NOT "improve" this into an
        # interpolation of the coarse envelope onto ``init_beam_time``:
        # coarse cell 0 is charge-free by construction
        # (``forbid_charge_in_first_coarse_cell`` in
        # calculate_rf_beam_current_partial), but cell 1 typically already
        # holds a large part of the bunch and therefore its beam-induced
        # voltage step. Interpolating from cell 0 toward cell 1 pulls that
        # beam loading BACKWARDS in time, into an initial condition that
        # predates the charge which produced it -- and the fine solve then
        # integrates the very same current a second time. (Measured: the
        # interpolated variant injects up to ~10% of the beam kick early
        # and breaks the independent-model comparisons against
        # MultiPassResonatorSolver.) The guard above enforces the other
        # half of the invariant, that this centre is not itself later than
        # the start of the fine window.
        antenna_voltage_init = self.antenna_voltage_coarse_grid[
            -self._rf_centers_lengths[-1] :
        ][0]

        generator_current_init = self.generator_current_coarse_grid[
            -self._rf_centers_lengths[-1] :
        ][0]

        omega_times_dt_fine_grid = omega_input * self.profile.hist_step
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
            initial_voltage_fine_grid=antenna_voltage_init,
            initial_generator_current_fine_grid=generator_current_init,
            omega_times_dt_fine_grid=omega_times_dt_fine_grid,
            relative_detuning=relative_detuning,
        )

    def cavity_response_fine(
        self,
        initial_voltage_fine_grid: float,  # TODO: these should all also be complex
        initial_generator_current_fine_grid: float,
        omega_times_dt_fine_grid: float,
        relative_detuning: float,
    ):
        r"""
        ACS cavity response model in matrix form on the fine-grid.

        Parameters
        ----------
        initial_voltage_fine_grid : float
            Initial condition of the voltage on the fine grid.
        initial_generator_current_fine_grid : float
            Initial condition of the generator current on the fine grid.
        omega_times_dt_fine_grid
            RF phase advanced in one fine-grid step [rad], i.e.
            ``omega * profile.hist_step``.
        relative_detuning
            Cavity detuning relative to the center frequency.
        """
        # Clamp to the actuator limit BEFORE the solve; see
        # ``_limit_fine_grid_generator_current`` on
        # GeneratorRegulationMixin.
        initial_generator_current_fine_grid = (
            self._limit_fine_grid_generator_current(
                initial_generator_current_fine_grid
            )
        )

        # The fine solve runs in the DEMODULATION frame: its seed (the
        # first forward coarse cell of the composed sum) carries the
        # generator component rotated by the generator frame rotation,
        # and its beam current was demodulated in that frame. The raw
        # (design-frame) generator current is rotated the same way into
        # LOCAL inputs -- the solve is linear, so this reproduces the
        # superposition of the two per-component fine solutions exactly.
        # The public ``generator_current_fine_grid`` stays the raw
        # (klystron-limited) design-frame current. Skipped while the
        # generator component is inactive, keeping an undriven feedback
        # bit-identical.
        generator_current_fine_grid = self.generator_current_fine_grid
        if self._generator_active:
            generator_current_fine_grid = (
                generator_current_fine_grid * self._generator_frame_rotation
            )
            initial_generator_current_fine_grid = (
                initial_generator_current_fine_grid
                * self._generator_frame_rotation
            )

        cavity_response_solver = (
            cavity_response_sparse_matrix_second_order
            if self._second_order_fine_grid_solver_enable
            else cavity_response_sparse_matrix
        )
        self.antenna_voltage_fine_grid = cavity_response_solver(
            I_beam=self.beam_current_fine_grid,
            I_gen=generator_current_fine_grid,
            V_ant_init=initial_voltage_fine_grid,
            I_gen_init=initial_generator_current_fine_grid,
            omega_times_dt=omega_times_dt_fine_grid,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            relative_detuning=relative_detuning,
        )

        self.antenna_voltage_fine_grid *= self.n_cavities

    def _assert_demodulation_frame_aligned(self, dT: float) -> None:
        r"""
        Reject a demodulation frame that would invert the beam loading.

        Parameters
        ----------
        dT
            The demodulation frame handed to :func:`rf_beam_current` for
            the forward coarse grid.

        Raises
        ------
        ValueError
            If ``omega_c * dT`` is not an odd multiple of ``pi`` while the
            demodulation can actually reach the beam.

        Notes
        -----
        Working the full phase chain through (the ``-e`` charge gauge, the
        ``-i omega_c t`` mixing, the ``+pi/2`` axis alignment, the solver's
        ``-I_beam`` sign, and the station kick ``sin(omega_rf t + phi_rf +
        phase_correction)``), the energy a bunch gives its own wake reduces
        to

        .. math:: \Delta E \propto (R/Q)\,\omega\,q\,\cos(\omega_c\,dT)

        because ``carrier_phase_offset = -(phi_rf + _carrier_slip_gap)``
        cancels the station phase and the readout phase identically.
        Neither ``phi_rf_design`` nor ``delta_omega_rf`` survives -- the
        grid geometry is design-clock only -- so ``omega_c * dT`` is the
        ONLY free phase left in the sign of beam loading. The fundamental
        theorem (a bunch must LOSE energy to its own wake) is therefore
        exactly ``cos(omega_c * dT) < 0``, and the frame is aligned only at
        ``omega_c * dT == pi`` (mod ``2 pi``) -- the value
        :meth:`_seed_initial_demodulation_frame` already seeds turn 0 to.

        Half an RF period off and the induced voltage is sign-inverted: the
        bunch is ACCELERATED by its own wake, and the wrongly signed deposit
        then decays only over ``2 Q_L / omega``, i.e. over many turns. This
        is reachable from ordinary inputs -- a segment that does not span a
        whole number of RF periods leaves ``residual = t_rf / 2 + frac *
        t_rf`` -- and no comparison in the suite covers it, hence the check.

        Gated on the demodulation being observable at all: with
        ``R_over_Q == 0`` the beam current cannot produce any antenna
        voltage, and with an empty profile histogram the demodulated charge
        is identically zero. Both are common in pure grid-geometry fixtures,
        whose off-``pi`` frames are inert and must not be rejected.

        The tolerance is ``1e-3 pi``: the worst float noise measured over
        the well-formed feedback suite is ``3.5e-6 pi`` (~290x margin),
        while the smallest reachable real defect is ``0.2 pi``
        (``n_rf_periods_per_coarse_grid = 0.6``), ~200x above it. The
        outcome is unchanged for any tolerance in ``[1e-5, 1e-2] pi``.
        """
        omega_c = self._forward_segment_omega_design
        if omega_c is None:
            return

        theta = omega_c * dT
        # Signed distance from the nearest ODD multiple of pi.
        deviation = (theta % (2 * np.pi)) - np.pi
        if abs(deviation) <= 1e-3 * np.pi:
            return

        # Only complain when the frame can actually reach the beam.
        if not self.R_over_Q:
            return
        if self.profile.hist_y is None:
            return
        if not np.any(copy_to_cpu(self.profile.hist_y)):
            return

        raise ValueError(
            "The beam-current demodulation frame is not aligned with the "
            "RF bucket. The fundamental theorem of beam loading requires "
            "omega_c * dT == pi (mod 2 pi), but "
            f"omega_c * dT = {theta / np.pi:.9f} pi, off by "
            f"{deviation / np.pi:.9f} pi. With "
            f"cos(omega_c * dT) = {np.cos(theta):+.6f} the beam-induced "
            "voltage is rotated by that angle -- and for an offset beyond "
            "0.5 pi it is sign-inverted, so the bunch would be ACCELERATED "
            "by its own wake instead of losing energy to it. Usual causes: "
            "the harmonic is not a whole number of RF periods per segment "
            "(not divisible by the number of reference-altering elements); "
            "n_rf_periods_per_coarse_grid < 1 with n != 0.5 (the "
            "sub-stepped grid tiles at omega_c * dT = 2 pi n, which is an "
            "odd multiple of pi only for n = 0.5); or a per-turn "
            "design-frequency change so large that the residual carried "
            "from the previous segment is stale. "
            f"[section_index={self.section_index}, "
            f"n_rf_periods_per_coarse_grid="
            f"{self.n_rf_periods_per_coarse_grid}, dT={dT!r}, "
            f"omega_c={omega_c!r}]"
        )

    def calculate_rf_beam_current_partial(
        self,
        beam: BeamBaseClass,
        n_points: int,
        remaining_delta_t_from_backfill: float,
    ) -> None:
        r"""
        Calculate the IQ beam current for the coarse and fine grid.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        n_points
            Number of points in the resulting coarse grid.
        remaining_delta_t_from_backfill
            Remaining time from the last rf_centers calculation, causes phase shift in beam current calculation.
        """
        if self.profile.active:
            self.profile.track(beam=beam)

        # Beam current from profile
        sampling_time_frwrd = (
            self.n_rf_periods_per_coarse_grid
            * 2
            * np.pi
            / self._forward_segment_omega_design
        )
        # Carried into the NEXT turn's cell 0. Note the asymmetry with
        # the generator current: the forward span consumes this grid
        # INCLUSIVE of its last element, so the sample carried here has
        # already been used by this turn's final step and is re-consumed
        # at the turn boundary -- the beam term does not have the
        # generator's ``i-1`` offset. It is inert in every shipped
        # configuration (measured exactly 0.0 over 495 demodulations
        # across the multi-section, sub-stepped, accelerating, multibunch
        # and counter-rotating suites): the forward coarse grid spans the
        # whole inter-station drift while the profile window is a few RF
        # periods at its start, so the last cell is never written. A
        # section-filling bunch train would bias the steady-state loading
        # by ~1/n_points, i.e. 4e-5 to 8e-5.
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
                + remaining_delta_t_from_backfill
            )
        else:
            dT_demodulation = remaining_delta_t_from_backfill

        # omega_c * dT is the only phase left in the beam-loading sign
        # after carrier_phase_offset cancels the station and readout
        # phases; it must be an odd multiple of pi or the bunch gains
        # energy from its own wake. Checked here, at the coarse-grid
        # call site, NOT inside rf_beam_current: the fine-grid-only
        # reference calls anchor on the profile's own hist_x and pass
        # dT = 0.0, for which 0 is the correct frame.
        self._assert_demodulation_frame_aligned(dT_demodulation)

        (
            self.beam_current_fine_grid,
            self.beam_current_forward_coarse_grid,
        ) = rf_beam_current(
            beam=beam,
            profile=self.profile,
            # The demodulation carrier is the *design* RF frequency; the grid
            # geometry stays on the design clock too. The RF-frequency offset
            # enters only as the constant carrier_phase_offset below, not as
            # a within-window carrier shift (the residual intra-window
            # mismatch delta_omega_rf * hist_x is bunch-local and negligible;
            # see the class docstring).
            #
            # KNOWN APPROXIMATION -- stale-residual frequency lag.
            # ``dT_demodulation`` is the tail left by the PRECEDING segment
            # (``_preceding_segment_residual``), but it is consumed here
            # against THIS segment's carrier. Under a ramp the two design
            # frequencies differ, so the demodulation frame
            # ``omega_c * dT`` is short/long by ``(omega_fwd - omega_prod) *
            # dT``. Since ``dT ~ t_rf / 2 = pi / omega``, that error in units
            # of pi is just the FRACTIONAL per-segment frequency change,
            #     frame lag [pi]  ~  (omega_fwd - omega_prod) / omega .
            # ``rf_center_grid.py`` makes the same distinction explicitly for
            # the grid geometry, where it uses ``_last_segment_omega_design``
            # rather than the current one; the demodulation does not.
            #
            # Left as-is deliberately: measured over the shipped programmes
            # the lag is 7.9e-8 pi on RCS1 -- the FASTEST ramp at ~23 %/turn
            # -- and 9.4e-10 pi on RCS2, against the 1e-3 pi tolerance of
            # ``_assert_demodulation_frame_aligned``. That is ~1.3e4 of
            # margin, i.e. the per-segment frequency step would have to grow
            # by four orders of magnitude (to ~0.1 % per segment) before the
            # frame is even flagged, let alone physically wrong.
            #
            # A MORE VIOLENT RAMP WOULD CHANGE THAT, and the failure is loud
            # rather than silent: the guard raises once the lag reaches
            # 1e-3 pi. If that happens, the fix is cheap and local --
            # ``RFCenterSegment`` already stores ``omega`` beside
            # ``residual``, so ``_preceding_segment_residual`` can return the
            # pair and the frame can be built from the producing carrier.
            # Do not simply rescale ``dT``: whether the residual should carry
            # the accumulated slip is a separate question that the comment
            # above answers in the negative, and no test pins either way.
            omega_c=self._forward_segment_omega_design,
            sampling_time=sampling_time_frwrd,
            n_points=n_points,
            dT=dT_demodulation,
            # Anchor the demodulation to the phase the BEAM actually
            # sees: minus the total that the station and the readout
            # add back on top of ``angle(V_ant)``. That total is the
            # station's RF phase ``phi_rf = phi_rf_design +
            # delta_phi_rf`` (applied by the kick, cavities.py) plus
            # the live kick-clock gap and the registration phase, both
            # carried in ``_carrier_slip_gap`` (applied via
            # ``phase_correction``), so the inter-turn slip cancels for
            # every deposit however long it is carried.
            #
            # Subtracting exactly that total is what makes a bunch LOSE
            # energy to its own wake: with the grid seeded half an RF
            # period into the bucket (dT = t_rf / 2, omega * dT = pi)
            # the fundamental theorem of beam loading needs
            # ``omega * dT + carrier_phase_offset + total == pi``,
            # which holds only when the DESIGN RF phase is subtracted
            # too. Omitting ``phi_rf_design`` rotates the beam-induced
            # voltage by ``-phi_rf_design``, and at
            # ``phi_rf_design = pi`` -- the ordinary above-transition
            # idiom -- it inverts the beam loading outright: the bunch
            # is accelerated by its own wake.
            #
            # The generator component deliberately does NOT carry
            # ``phi_rf_design``: the klystron drive is locked to the
            # design RF wave, which the station itself supplies through
            # ``phi_rf``, so ``_generator_frame_rotation`` stays as it
            # is (see :meth:`_update_frame_rotations`). Exactly -0.0,
            # hence a bit-identical demodulation, for the shipped
            # ``phi_rf_design = 0`` runs without an RF-frequency
            # offset.
            carrier_phase_offset=-(self.phi_rf + self._carrier_slip_gap),
            # The fine-grid initial antenna voltage is taken from the first
            # coarse cell (see circuit_track), so that cell must stay
            # charge-free or its beam kick would be double-counted.
            forbid_charge_in_first_coarse_cell=True,
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / sampling_time_frwrd
        )
