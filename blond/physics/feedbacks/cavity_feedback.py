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
    cavity_response_sparse_matrix,
    cavity_response_sparse_matrix_second_order,
    coarse_step_exponent,
    exponential_drive_weight,
    exponential_voltage_multiplier,
    pretrack_fill_voltage,
    propagate_beam_free_voltage,
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
        # points per RF period, i.e. a finer sampling of the generator
        # command and of the coarse beam current. The coarse step itself is
        # exact for any step length (see _advance_coarse_voltage), so this
        # is not a stability device. It is a deliberate configuration and
        # is therefore accepted without warning.
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
        them cell by cell: with the CURRENT passage's rotation over the
        forward span, and over the backfill span with the rotation of the
        phase accumulated up to each cell. With nothing driving the
        generator -- no controller, zero ``generator_current_bias``, and
        neither a carried generator current nor a carried
        generator-sourced voltage (an initial or pre-fill voltage seeds
        that one) -- the generator component is identically zero and this
        sum equals the beam component bit-for-bit, the composition adding
        an exact zero.
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
        With nothing driving the generator (see above) this component
        IS the former single state, bit-for-bit.
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
    def phi_rf_loop(self) -> float:
        """
        Per-station phase-loop offset of the parent cavity.

        The offset a
        :class:`~blond.physics.feedbacks.station_phase_loop.StationPhaseLoop`
        writes into the station's actual RF phase. A phase STEP, not a
        frequency slip: it enters the station clock the frame rotations
        use (``delta_phi_rf + phi_rf_loop``), and a change of it between
        two passages counter-rotates the carried beam-sourced envelope
        (``_absorb_phase_loop_step``). Exactly ``0.0`` without such a
        loop.

        Returns
        -------
        phi_rf_loop
            The parent station's current offset [rad].
        """
        return float(self._parent_rf_station.phi_rf_loop)

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

        No tracking code reads it. Its one consumer, the forward-Euler
        step-size check, was removed with the Euler coarse step on
        2026-09-11; it remains a public, user-facing estimate of the coarse
        cell width, e.g. in the remedy of the fine-grid causality error.

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
    the exact exponential propagator of the cavity-envelope ODE for a source
    held constant over each step; see ``cavity_response`` and, for the
    derivation, ``_advance_coarse_voltage``.

    By default (no ``controller``) the generator current is a constant value
    (``generator_current_bias``). Passing a
    :class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController`
    instead turns it into a regulated generator current: each coarse-grid
    step the feedback forms the antenna-voltage error in the *kick frame*
    -- ``V_set - V_sum[n] * exp(+i (gap + phi_acc[n]))``, the envelope the
    station actually applies, ``phi_acc[n]`` being the grid-vs-carrier
    phase accumulated up to cell ``n`` (the forward segment's value over
    the whole forward span) -- and lets the controller convert it into the
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
    second_order_fine_grid_solver_enable
        If True, integrate the fine-grid cavity response with the second-order
        (trapezoidal / Crank-Nicolson) solver instead of the default
        first-order forward-Euler one. The second-order solver is much more
        accurate at coarse profile binning (its error scales as the bin size
        squared rather than linearly). Default is False.
    controller
        Optional generator-current controller (a
        :class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController`)
        that converts the antenna-voltage error into the generator current.
        If None, the generator current stays at the constant value
        ``generator_current_bias``.
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
    controller_update_interval
        Coarse cells between controller updates [1]. The coarse grid is the
        *cavity model's* step -- one RF period per cell by default, so
        1.3 GHz on an RCS -- and no LLRF samples that fast. This decouples
        the two rates: the controller is evaluated every
        ``controller_update_interval``-th cell and its command is held
        (zero order) over the cells in between, exactly as a digital loop
        holds its DAC between samples, while the cavity recursion keeps
        stepping every cell. Default 1, which regulates on every cell as
        before. Note that the controller's own ``n_delay`` then counts
        *controller* samples, not coarse cells, so a physical loop delay
        must be discretised on ``controller_update_interval * coarse step``.
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

    **Diagnostics.** The switches that record grid snapshots
    (``debug``), re-check the grid every passage
    (``validate_grid_each_turn``) or end the passage after the grid
    without a correction (``grid_only_no_correction``) are not part of
    this class. Only tests consume them, so they live on the test variant
    ``blond.testing.cavity_feedback.DiagnosticIQCavityFeedbackTimingClass``,
    which tracks bit-for-bit like this class with all three off.

    **Sub-stepping (** ``n_rf_periods_per_coarse_grid`` **< 1).** A
    fractional ``n`` places several coarse samples per RF period,
    ``dt = n_rf_periods_per_coarse_grid * t_rf``. The coarse step is exact
    for any step length -- it integrates the decay and the detuning rotation
    in closed form for a source held over the step (see
    ``_advance_coarse_voltage``) -- so sub-stepping is not needed for
    stability at low ``Q_L`` or large detuning. What it changes is the
    sampling: the generator command is held, and the controller stepped,
    over shorter cells, and the beam current is binned onto them. In this
    mode the coarse grid no longer re-aligns to an RF bucket each turn; the
    centres tile continuously across the turn boundary (see
    ``_generate_rf_centers``). That tiling makes the demodulation frame one
    previous coarse step, ``omega * dT = 2 pi n``, an odd multiple of ``pi``
    only at ``n = 0.5``, so ``0.5`` is the only sub-step
    ``_assert_demodulation_frame_aligned`` accepts once the beam loading is
    observable.

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
    # path is kept as the readable reference and the fallback for degenerate
    # (coincident) coarse steps; klystron-limit saturation is handled inside
    # the kernel. Set ``False`` on an instance to force the reference path.
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
        second_order_fine_grid_solver_enable: bool = False,
        controller: GeneratorCurrentController | None = None,
        voltage_setpoint: complex | None = None,
        controller_update_interval: int = 1,
        n_pretrack: int | None = None,
        injection_voltage: float | None = None,
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
        # yet, so the first _close_previous_turn_grid replaces it with
        # the tiling-consistent first-passage value (see its FIRST
        # PASSAGE note). Without that continuation a station that is the
        # ring's FIRST reference-altering element generates no backfill
        # on turn 0 and demodulates that turn pi out of phase -- the
        # beam-induced voltage then comes out with the wrong sign.
        self._residual_time_last_rf_centers_calculation = 0.0
        # Residual [s] the PREVIOUS turn's last segment ended on. The first
        # segment of a turn steps across the turn boundary from it; the live
        # scalar above cannot serve, because by the time the grid is walked
        # it has been overwritten by THIS turn's last-generated segment (see
        # _preceding_segment_residual).
        self._residual_time_carried_into_turn: float | None = None
        # The forward segment the PREVIOUS passage ended on. Its ``omega``
        # is the carrier the envelope carried into this passage was
        # demodulated against, and its ``accumulated_phase`` is what this
        # passage's backfill segments continue from (see
        # RFCenterGridMixin._backfill_accumulated_phases). ``None`` before
        # this station's first passage.
        self._forward_segment_carried_into_turn: RFCenterSegment | None = None

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
        self._last_tracked_turn_frwrd: int = 0
        self._last_tracked_beam_state_frwrd: bool | None = None

        self._init_passage_tracking_state()

        self._init_turn_boundary_carries()

        self._init_voltage = initial_voltage

        self._second_order_fine_grid_solver_enable = (
            second_order_fine_grid_solver_enable
        )

        self._generator_current_bias = generator_current_bias

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
        # Sampling rate of the loop, in cavity-model steps. The phase is the
        # free-running clock: it counts coarse cells across spans and turns,
        # so a segment boundary or a passage does not re-phase the LLRF.
        if (
            int(controller_update_interval) != controller_update_interval
            or controller_update_interval < 1
        ):
            raise ValueError(
                "controller_update_interval must be a positive whole "
                f"number of coarse cells, got "
                f"{controller_update_interval!r}"
            )
        self._controller_update_interval = int(controller_update_interval)
        self._controller_update_phase = 0

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

    @property
    def rf_centers(self) -> NumpyArray:
        """
        Flat coarse-grid centre times of the current passage, in [s].

        The stated read surface of ``_rf_centers``: the concatenation of
        the per-segment ``centers`` of ``_segments``. The whole-turn
        coarse quantities of this class are indexed by it; the one
        coarse array that is not is
        ``beam_current_forward_coarse_grid``, which is
        forward-segment-local and has to be reached through
        :attr:`forward_offset` (see the INDEX ORIGIN paragraph of that
        attribute's docstring). Its own two counter-intuitive
        properties -- the entries are segment-LOCAL times (the array is
        not globally monotonic) and the step is the design ``t_rf`` of
        the segment an entry belongs to (the spacing is not one constant
        ``dt``) -- must be read in "The coarse grid" under Notes in the
        class docstring before indexing or differencing it.

        Read-only on purpose: ``_segments`` is the source of truth of the
        grid and ``_rebuild_grid_arrays`` derives this array from it.
        Assigning here would write the flat array alone and desync it
        from the segment list the recorded grid is reconstructed from.

        Returns
        -------
        rf_centers
            Coarse-grid centre times of the current passage, in [s].
        """
        return self._rf_centers

    @property
    def rf_centers_lengths(self) -> NumpyArray:
        """
        Number of coarse cells each grid segment contributed.

        The stated read surface of ``_rf_centers_lengths``: entry ``j``
        is the length of segment ``j`` of :attr:`rf_centers`, in grid
        order, so the last entry is the forward (real passage) segment
        and the entries before it are the backfill reconstruction.

        Read-only for the same reason as :attr:`rf_centers`: both flat
        arrays are derived from ``_segments`` by ``_rebuild_grid_arrays``,
        and writing one here would desync it from that source of truth.

        Returns
        -------
        rf_centers_lengths
            Cell count per segment of the current passage's coarse grid.
        """
        return self._rf_centers_lengths

    @property
    def forward_offset(self) -> np.integer:
        """
        Index origin conversion, whole-turn to forward-segment index.

        ``len(rf_centers) - rf_centers_lengths[-1]``, i.e. the number of
        backfill cells preceding the forward segment. A whole-turn coarse
        index minus this offset is the matching index into the
        forward-segment-local ``beam_current_forward_coarse_grid`` (see
        the INDEX ORIGIN paragraph of that attribute's docstring), which
        is the one coarse array that is not whole-turn indexed.

        Read-only: it is derived from the two flat arrays, which are
        themselves derived from ``_segments``.

        Returns
        -------
        forward_offset
            Number of coarse cells before the forward segment, as a
            ``numpy.integer`` and NOT a Python ``int``:
            :attr:`rf_centers_lengths` is an integer array, so
            subtracting its last entry yields a NumPy scalar. That is
            deliberate -- the value stays bit-for-bit the expression its
            callers used to open-code. It indexes and slices exactly
            like an ``int``; a caller that needs a true Python ``int``
            (a JSON payload, a type-checked signature) should cast it,
            the way ``observables.py`` does with ``int(...)``.

        Raises
        ------
        IndexError
            If the coarse grid has not been built yet:
            ``_rf_centers_lengths`` stays empty until the first passage
            fills it, and there is then no last segment to subtract.
            The other coarse readers of this class fail the same way, so
            read this only once the feedback has tracked a passage.
        """
        return len(self._rf_centers) - self._rf_centers_lengths[-1]

    @property
    def controller_update_interval(self) -> int:
        """
        Coarse cells between controller updates [1].

        Returns
        -------
        interval
            1 regulates on every cavity-model step; ``x`` samples the loop
            ``x`` times slower and holds the command in between.
        """
        return self._controller_update_interval

    def _init_turn_boundary_carries(self) -> None:
        """
        Initialise the turn-boundary carries of the coarse recursion.

        The two antenna-voltage components are the propagated state (see
        :meth:`reset_arrays` / :meth:`cavity_response`); the un-suffixed
        value is the carried demodulation-frame SUM, kept for diagnostics
        and the coincident-first-cell duplication.
        """
        self._last_val_ant_voltage: complex = 0.0
        self._last_val_ant_voltage_gen: complex = 0.0
        self._last_val_ant_voltage_beam: complex = 0.0
        self._last_val_generator_current: float = 0.0
        self._last_rf_centers_entry: float | None = None

    def _init_passage_tracking_state(self) -> None:
        """
        Initialise the per-passage bookkeeping attributes.

        Groups the state written once per ``_track`` call: the
        simultaneous counter-rotating passage detection (the arrival time
        and direction of the previous ``_track`` call, plus the
        coarse-cell width of its forward grid as the coincidence
        tolerance), the live tail of the RF-frequency-offset phase
        slip (the slip accumulated since the station kick clock's last
        end-of-track tick; ``0.0`` without an offset) and the carrier slip
        gap it is folded into, and the frame rotations derived from them.
        The accumulated grid-vs-carrier phase is not state of the feedback:
        each coarse-grid segment record stores it
        (``RFCenterSegment.accumulated_phase``).
        """
        self._last_track_arrival_time: float | None = None
        self._last_track_is_counter_rotating: bool | None = None
        self._last_forward_cell_width: float | None = None
        # Live tail of the kick-clock slip at this passage (the slip since
        # the station clock's last end-of-track tick); one of the two
        # constituents folded into ``_carrier_slip_gap``, the other being
        # the forward segment's accumulated phase.
        self._kick_clock_slip_gap: float = 0.0
        self._carrier_slip_gap: float = 0.0
        # Per-passage frame rotations (see ``_update_frame_rotations``);
        # exactly unity until a passage computes them, which is also the
        # neutral value for direct (test) driving of the cell loops.
        self._generator_frame_rotation: complex = 1.0 + 0.0j
        self._kick_frame_rotation: complex = 1.0 + 0.0j
        self._pi_error_frame_rotation: complex = 1.0 + 0.0j
        # The parent station's per-station phase-loop offset the carried
        # beam-sourced envelope was last demodulated and read out in. A
        # change since then is a STEP of the RF reference, which the next
        # passage absorbs (see ``_absorb_phase_loop_step``); the backfill
        # span of that passage still belongs to the interval before the
        # step and keeps this value.
        self._phi_rf_loop_seen: float = 0.0
        # The generator and kick rotations of the BACKFILL cells, one per
        # backfill centre, each with the phase accumulated up to its cell
        # (see ``_update_frame_rotations``). Empty until a passage computes
        # them: every cell beyond them -- the forward span, and any grid
        # driven directly -- takes the per-passage scalars above.
        self._backfill_generator_frame_rotations: NumpyArray = np.zeros(
            0, dtype=np.complex128
        )
        self._backfill_kick_frame_rotations: NumpyArray = np.zeros(
            0, dtype=np.complex128
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
        reason the first passage's demodulation frame is seeded there.

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
        # driven cell-by-cell on the reference path instead. The controller
        # runs on every tracked span, backfill segments included; only a
        # feedback with no controller attached takes the compiled path
        # unconditionally.
        controller_runs = self._controller_active
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
        # Advance the free-running update clock by the cells this span
        # consumed, so the next span (or turn) continues it instead of
        # restarting. Coincident cells count: they carry no time, but the
        # clock is a cell counter and the jitter is one cell of a sub-step.
        self._controller_update_phase = (
            self._controller_update_phase + max(end_index - start_index, 0)
        ) % self._controller_update_interval

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
                # condition from the cell BEFORE the first forward one.
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
                # (composed with THIS cell's rotation -- the sum is derived
                # from the component state, never propagated).
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
                update_controller=(
                    (
                        self._controller_update_phase
                        + rf_centers_idx
                        - start_index
                    )
                    % self._controller_update_interval
                    == 0
                ),
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

        Precomputes on the host the per-cell step sizes, the exact
        propagator's voltage multiplier / drive weight and the frame rotations
        (all state-independent), marshals the
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
            generator_current_init = complex(self._last_val_generator_current)
        else:
            voltage_gen_init = self.antenna_voltage_gen_coarse_grid[
                start_index - 1
            ]
            voltage_beam_init = self.antenna_voltage_beam_coarse_grid[
                start_index - 1
            ]
            generator_current_init = self.generator_current_coarse_grid[
                start_index - 1
            ]

        controller_active = self._controller_active
        # The controller owns its compiled law and marshals its own tuning and
        # state; this class passes the result straight through without
        # inspecting it. It runs on every tracked span, the no-beam backfill
        # segments included (see ``cavity_response``); only when no
        # controller is attached does the neutral state keep the generator
        # current constant.
        if controller_active:
            envelope_scan = self._controller.envelope_scan_kernel()
            controller_state = self._controller.envelope_scan_state()
            voltage_setpoint = complex(self.pi_setpoint)
        else:
            envelope_scan = envelope_pi_scan
            controller_state = inactive_controller_scan_state()
            # No controller attached, so the error is never formed and the
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
        # Per cell, like the multipliers: a backfill cell carries the phase
        # accumulated up to it, a forward cell the passage's rotation.
        generator_frame_rotations, kick_frame_rotations = (
            self._frame_rotations_of_cells(start_index, end_index)
        )

        delay_buffer, delay_head, integral = envelope_scan(
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
            generator_frame_rotations,
            kick_frame_rotations,
            complex(self._pi_error_frame_rotation),
            controller_active,
            self._controller_update_interval,
            self._controller_update_phase,
            voltage_setpoint,
            float(omega_input),
            *controller_state,
        )

        self.antenna_voltage_beam_coarse_grid[start_index:end_index] = (
            voltage_beam_out
        )
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

    def _step_into_first_cell(
        self,
        omega_input: float,
        start_index: int,
        end_index: int,
    ) -> float:
        """
        Length of the coarse step that ends at a segment's first centre.

        That step crosses a segment (or turn) boundary, so it is the first
        cell's own local time plus the PRECEDING segment's unfilled tail
        (see :meth:`_preceding_segment_residual`). The very first centre
        ever tracked has no predecessor to step from, so the spacing to
        the next centre of the same segment stands in for it -- or this
        segment's own coarse step when it holds a single centre, since the
        next centre would then belong to a segment at another frequency.

        Parameters
        ----------
        omega_input
            Angular frequency of this segment [rad/s].
        start_index
            First ``rf_centers`` index of the segment.
        end_index
            One past the last ``rf_centers`` index of the segment.

        Returns
        -------
        delta_t
            Length of the step ending at ``rf_centers[start_index]`` [s].
        """
        if start_index == 0 and self._last_rf_centers_entry is None:
            if start_index + 1 < end_index:
                return float(self._rf_centers[1] - self._rf_centers[0])
            return float(
                self.n_rf_periods_per_coarse_grid * 2 * np.pi / omega_input
            )
        return float(
            self._rf_centers[start_index]
            + self._preceding_segment_residual(start_index)
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
        delta_t[0] = self._step_into_first_cell(
            omega_input, start_index, end_index
        )
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

        Applies the exact exponential propagator of the cavity-envelope ODE
        for a source held constant over the step,
        ``V_next = e^L * v_prev + drive * (e^L - 1) / L``, with the per-step
        drive ``drive = (R/Q) omega dt (I_gen - I_beam/2)``. The Notes below
        derive it and explain why the forward-Euler step this class took
        until 2026-09-11 is only its first-order truncation.

        The step exponent and the propagator weights come from
        :mod:`~blond.physics.feedbacks.cavity_solvers`, so this per-cell
        path and the vectorised :meth:`_kernel_step_multipliers` spell the
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

        Notes
        -----
        **Envelope ODE.** Per cavity, the IQ envelope of the antenna voltage
        obeys

        .. math::
            \frac{\mathrm{d}V}{\mathrm{d}t} = \lambda V + s, \qquad
            \lambda = -\frac{\omega}{2 Q_L} + i\,\Delta\omega, \qquad
            s = \frac{R}{Q}\,\omega
                \left(I_\mathrm{gen} - \frac{I_\mathrm{beam}}{2}\right).

        The coarse model holds ``s`` constant over one step: ``I_gen`` is the
        zero-order-held command of the previous cell (the controller output,
        or the bias, written there one step earlier) and ``I_beam`` is the
        beam current binned onto this cell.

        **Exact step.** For constant ``s`` the ODE integrates in closed form
        over a step ``dt``:

        .. math::
            V_{n+1} = e^{L} V_n
                + \int_0^{\Delta t} e^{\lambda (\Delta t - \tau)}\,
                  s\,\mathrm{d}\tau
              = e^{L} V_n + s\,\Delta t\,\frac{e^{L} - 1}{L},
            \qquad L = \lambda\,\Delta t.

        Nothing is expanded or truncated: for a piecewise-constant source this
        is the solution of the ODE at the end of the step, for any ``dt``,
        ``Q_L`` and ``delta_omega``. The code spells it
        ``v_prev * B + drive * W`` with ``B = e^L``, ``W = (e^L - 1) / L``
        and ``drive = s dt``.

        **Forward Euler is its first-order truncation.** Keeping only
        ``e^L ~ 1 + L`` and ``(e^L - 1) / L ~ 1`` gives

        .. math::
            V_{n+1} = (1 + L)\,V_n + s\,\Delta t,

        the update of BLonD 2's ``LHCCavityLoop.cavity_response``; with
        ``samples = omega dt`` and ``detuning = delta_omega / omega`` it reads
        ``V[n] = V[n-1] (1 - samples / (2 Q_L) + i detuning samples) +
        (R/Q) samples (I_gen[n-1] - I_beam[n-1] / 2)``. This class inherited
        that form as its default coarse step; the forward-Euler step, the
        switch to the exact one and the Euler validity guard were removed on
        2026-09-11.

        **Why Euler is only an approximation.**

        * Its local error is ``e^L - (1 + L) = L^2 / 2 + O(L^3)`` per step,
          so it is only first order globally.
        * Pure detuning: ``|1 + i delta_omega dt| =
          sqrt(1 + (delta_omega dt)^2) > 1``, so the Euler envelope grows
          every step where the exact one, ``|exp(i delta_omega dt)| = 1``,
          only rotates.
        * Decay: the Euler factor ``1 - d`` with the per-step decay
          ``d = omega dt / (2 Q_L)`` changes sign once ``d > 1``, and the
          recursion diverges once ``|1 + L| > 1`` -- with detuning already
          at a tiny ``d`` once ``(delta_omega dt)^2 > d (2 - d)``. The exact
          ``|e^L| = exp(-d)`` is at most 1 for every step.
        * Deposit weight: Euler weights the held source by ``1`` instead of
          ``(e^L - 1) / L = 1 + L / 2 + O(L^2)``.

        **Size and cost.** For the shipped muon-collider parameters (one RF
        period per coarse cell, ``Q_L ~ 1.3e6``, detuning of order kHz)
        ``|L|`` is a few ``1e-6`` per step. On the multi-turn convolution
        harness of the unit tests (one section static, and four sections
        accelerating) the two steps gave beam-induced voltages differing by
        7.1e-7 and 8.5e-7 relative on the second and third turn (measured
        2026-09-11). The exact step costs the same: ``B`` and ``W`` depend
        only on the step length and the cavity parameters, so the kernel
        path precomputes them per cell (:meth:`_kernel_step_multipliers`)
        and the recursion is the same multiply-and-add either way.
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
        # The drive weight (e^L - 1) / L stays accurate (-> 1) as L -> 0 and
        # is guarded at the exact zero, which this scalar path -- unlike the
        # vectorised one -- can be handed.
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
        state), so they are precomputed here on the host: ``B = e^L`` and
        ``W = (e^L - 1) / L`` of the exact exponential propagator, with ``L``
        the per-cell growth exponent (derivation, and the forward-Euler
        ``B = 1 + L``, ``W = 1`` it replaced: Notes of
        :meth:`_advance_coarse_voltage`). The arithmetic itself is the shared
        one of
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
        voltage_multiplier = exponential_voltage_multiplier(step_exponent)
        # omega_times_dt > 0, so step_exponent != 0 and (e^L - 1) / L is
        # well defined -- the weight's zero guard is never reached here.
        drive_weight = exponential_drive_weight(step_exponent)
        return voltage_multiplier, drive_weight

    def _compose_coarse_sum(self, coarse_grid_index: int) -> complex:
        """
        Compose the demodulation-frame sum at one coarse-grid index.

        ``V_beam + V_gen * generator frame rotation``: the beam component
        already lives in the demodulation frame, the design-anchored
        generator component is rotated into it with this cell's rotation
        (see :meth:`_frame_rotations_of_cell`). With nothing driving the
        generator that component is identically zero, so the sum is then
        the beam component bit-for-bit without a special case.

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
        generator_frame_rotation, _ = self._frame_rotations_of_cell(
            coarse_grid_index
        )
        return voltage_beam + (
            self.antenna_voltage_gen_coarse_grid[coarse_grid_index]
            * generator_frame_rotation
        )

    def _frame_rotations_of_cell(
        self, coarse_grid_index: int
    ) -> tuple[complex, complex]:
        """
        Generator and kick frame rotation of one coarse cell.

        A backfill cell takes the rotations of the phase accumulated up to
        it; every other cell -- the forward span, and every cell of a grid
        driven directly rather than through :meth:`_track` -- takes the
        per-passage scalars (see :meth:`_update_frame_rotations`).

        Parameters
        ----------
        coarse_grid_index
            Whole-turn coarse-grid index of the cell.

        Returns
        -------
        generator_frame_rotation
            Rotation the design-anchored generator component of this cell
            is composed with.
        kick_frame_rotation
            Rotation taking this cell's demodulation-frame sum into the kick
            frame, in which the PI error is formed.
        """
        if coarse_grid_index < len(self._backfill_generator_frame_rotations):
            return (
                self._backfill_generator_frame_rotations[coarse_grid_index],
                self._backfill_kick_frame_rotations[coarse_grid_index],
            )
        return self._generator_frame_rotation, self._kick_frame_rotation

    def _frame_rotations_of_cells(
        self, start_index: int, end_index: int
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Generator and kick frame rotations of a span of coarse cells.

        The vectorised twin of :meth:`_frame_rotations_of_cell`, for the
        compiled scan and the whole-grid readouts; the two must agree cell
        by cell, or the kernel-vs-reference byte identity breaks.

        Parameters
        ----------
        start_index
            First whole-turn coarse-grid index of the span.
        end_index
            One past the last index of the span.

        Returns
        -------
        generator_frame_rotations
            Per-cell generator frame rotation (complex128, length
            ``end_index - start_index``).
        kick_frame_rotations
            Per-cell kick frame rotation (complex128, same length).
        """
        n_cells = end_index - start_index
        generator_frame_rotations = np.full(
            n_cells, self._generator_frame_rotation, dtype=np.complex128
        )
        kick_frame_rotations = np.full(
            n_cells, self._kick_frame_rotation, dtype=np.complex128
        )
        backfill_end = min(
            end_index, len(self._backfill_generator_frame_rotations)
        )
        if start_index < backfill_end:
            n_backfill_cells = backfill_end - start_index
            generator_frame_rotations[:n_backfill_cells] = (
                self._backfill_generator_frame_rotations[
                    start_index:backfill_end
                ]
            )
            kick_frame_rotations[:n_backfill_cells] = (
                self._backfill_kick_frame_rotations[start_index:backfill_end]
            )
        return generator_frame_rotations, kick_frame_rotations

    def cavity_response(
        self,
        omega_times_dt: float,
        coarse_grid_index_to_update: int,
        relative_detuning: float,
        no_beam: bool = False,
        update_controller: bool = True,
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
        update_controller
            Whether this cell is a controller sample. False holds the
            previous command over the cell (zero order); see
            ``controller_update_interval``. The cell-loop passes the free-
            running clock's verdict; a direct caller regulating every cell
            leaves it True.
        """
        index = coarse_grid_index_to_update
        # A cell's beam current drives the step that ENDS at its own
        # centre, so it is read from THIS passage's grid at every index --
        # index 0 included, where that step crosses the passage boundary.
        # Nothing is carried from the previous passage: its last cell has
        # already driven its own step, and re-using it here counted the
        # same charge twice. The boundary step has no cell of its own,
        # which is why the demodulation refuses charge in the last cell
        # (``forbid_charge_in_last_coarse_cell``).
        if no_beam:
            beam_current = 0
        else:
            beam_current = self.beam_current_forward_coarse_grid[
                index - self.forward_offset
            ]
        if index != 0:
            voltage_gen_prev = self.antenna_voltage_gen_coarse_grid[index - 1]
            voltage_beam_prev = self.antenna_voltage_beam_coarse_grid[
                index - 1
            ]
            generator_current = self.generator_current_coarse_grid[index - 1]
        else:
            voltage_gen_prev = self._last_val_ant_voltage_gen
            voltage_beam_prev = self._last_val_ant_voltage_beam
            generator_current = self._last_val_generator_current
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
        # Stepped on EVERY tracked cell, the no_beam backfill reconstruction
        # segments included: a real LLRF regulates continuously, and a loop
        # confined to the forward passage would be open-loop for
        # (N - 1) / N of every turn on an N-section ring, merely holding the
        # current the forward pass last commanded. The error on a backfill
        # cell is formed in the frame of THAT cell: its rotations carry the
        # phase accumulated up to the cell (``_update_frame_rotations``
        # computes them per backfill cell before the replay), not the
        # passage's final phase, which would rotate the beam-induced part of
        # the carried voltage by the phase still to accumulate over the rest
        # of the span and make the regulated voltage jump by one passage's
        # increment where the previous forward span hands over. All of them
        # are exactly unity without an RF-frequency offset and without
        # multi-section acceleration.
        if self._controller_active:
            if update_controller:
                self._update_generator_current(
                    omega_times_dt=omega_times_dt,
                    coarse_grid_index_to_update=coarse_grid_index_to_update,
                )
            else:
                # Between samples the loop holds its last command, which is
                # the one that drove this very step.
                self.generator_current_coarse_grid[index] = generator_current

    def _kernel_beam_current(
        self,
        no_beam: bool,
        start_index: int,
        end_index: int,
        n_cells: int,
    ) -> NumpyArray:
        """
        Per-cell beam current for a kernel segment.

        Mirrors ``cavity_response``: zero for a no-beam segment, and this
        passage's own forward beam-current grid otherwise -- index 0
        included, since nothing is carried across the passage boundary.

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
        if no_beam:
            return np.zeros(n_cells, dtype=np.complex128)
        forward_start = start_index - self.forward_offset
        return self.beam_current_forward_coarse_grid[
            forward_start : forward_start + n_cells
        ].astype(np.complex128)

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
        which are seeded with the last commanded generator current. That is
        the initial condition of the span the controller then regulates
        over: those cells replay an interval that began with the generator
        running at whatever it was last told, not snapped back to the
        feedforward value, and the loop steps on every one of them
        (:meth:`cavity_response`), overwriting the seed cell by cell.
        :meth:`cavity_response` drives the *first* backfill cell from
        ``_last_val_generator_current``; this seeds the rest of the span
        consistently. Without a controller nothing overwrites the seed and
        the held value *is* the bias, so the constant-current path is
        bit-unchanged.

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

        One phase acts on state instead of producing a value:
        :meth:`_absorb_phase_loop_step`, between the backfill replay and
        the forward span, counter-rotates the carried beam-sourced
        envelope when the station's per-station phase-loop offset
        (``phi_rf_loop``) changed since the previous passage. The
        backfill span belongs to the interval before that step, the
        forward span to the one after, which fixes its place.

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
        # The forward segment stores the grid-vs-carrier phase accumulated
        # up to this passage (see RFCenterSegment.accumulated_phase).
        self._carrier_slip_gap = (
            self._kick_clock_slip_gap + self._segments[-1].accumulated_phase
        )
        self._update_frame_rotations()

        self._replay_backfill_span(n_backfill_centers=span.n_backfill_centers)
        # A per-station phase-loop step happened at THIS passage: the
        # backfill span above replayed the interval before it, the forward
        # span below runs after it.
        self._absorb_phase_loop_step(
            n_backfill_centers=span.n_backfill_centers
        )

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
        then formed as this gap plus the forward segment's accumulated
        phase (``RFCenterSegment.accumulated_phase``), and it is that SUM
        which :meth:`_update_frame_rotations` folds together with the
        station clock ``delta_phi_rf`` -- the generator-component rotation
        uses the full gap, not the kick-clock part alone.
        ``_kick_clock_slip_gap`` is retained as the named intermediate of
        that sum, and as the gap the per-cell backfill rotations add their
        own accumulated phases to (see :meth:`_update_frame_rotations`).
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

        FRAME: every replayed cell is composed, and regulated, in the frame
        of the phase accumulated up to THAT cell -- the per-cell
        ``_backfill_generator_frame_rotations`` and
        ``_backfill_kick_frame_rotations`` -- rather than in the passage's
        final frame, which the forward span keeps. So
        :meth:`_update_frame_rotations` must have run for this passage
        first.
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

    def _absorb_phase_loop_step(self, n_backfill_centers: int) -> None:
        r"""
        Keep the carried beam-induced field in place across a phase step.

        A per-station phase loop moves the station's actual RF phase by
        writing ``phi_rf_loop``. The demodulation/readout chain keeps
        every deposit at a fixed phase *relative to the RF wave* -- the
        demodulation subtracts ``phi_rf + carrier_slip_gap`` and the
        station adds ``phi_rf`` back at the kick -- which is right for a
        frequency slip, where the tuner makes the cavity follow the RF,
        but wrong for a step of the RF reference: the beam-induced field
        in the cavity does not jump. Left alone, the chain would apply
        every deposit carried from before the step ``delta`` further
        along, so the carried beam-sourced component is counter-rotated
        by ``exp(-i delta)`` here, once, at the passage where the change
        is first seen. Deposits of this passage are demodulated in the
        new frame and need nothing.

        The generator-sourced component is not touched: it is anchored to
        the design clock and composed with the station clock, which now
        includes ``phi_rf_loop`` (:meth:`_update_frame_rotations`), so it
        appears at MINUS the step relative to the new RF -- the physical
        walk-off of a drive the reference moved away from, which an
        attached controller then removes.

        Placement: between the backfill replay and the forward span. The
        backfill span reconstructs the interval since the previous passage,
        which lies before the step, so its cells run with the previous
        offset (``_phi_rf_loop_seen``); the rotation is applied to the state
        the forward span starts from -- the last backfill centre, or the
        state carried across the passage boundary when there is no
        backfill -- and to that cell's composed sum, so the fine-grid seed
        (:meth:`_state_before_forward_span`) and the forward recursion both
        read the rotated value.

        Exactly a no-op, to the bit, while the offset does not change --
        every run without such a loop.

        Parameters
        ----------
        n_backfill_centers
            Number of backfill centres of this passage's grid.
        """
        step = self.phi_rf_loop - self._phi_rf_loop_seen
        self._phi_rf_loop_seen = self.phi_rf_loop
        if step == 0.0:
            return
        rotation = complex(np.exp(-1j * step))
        if n_backfill_centers > 0:
            last = n_backfill_centers - 1
            voltage_beam = self.antenna_voltage_beam_coarse_grid[last]
            self.antenna_voltage_beam_coarse_grid[last] = (
                voltage_beam * rotation
            )
            self.antenna_voltage_coarse_grid[last] += (
                rotation - 1.0
            ) * voltage_beam
        else:
            voltage_beam = self._last_val_ant_voltage_beam
            self._last_val_ant_voltage_beam = voltage_beam * rotation
            self._last_val_ant_voltage += (rotation - 1.0) * voltage_beam

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
            e^{-i(\phi_\mathsf{clock} + \mathrm{gap} + \phi_\mathsf{acc})}

        (the station clock + live kick-clock gap + the accumulated
        grid-vs-carrier phase ``phi_acc``). The station clock
        ``phi_clock = delta_phi_rf + phi_rf_loop`` is what the station
        adds to the design phase through ``phi_rf``: the kick clock
        accumulated from the RF-frequency offset plus the per-station
        phase-loop offset. The readout later adds ``gap + phi_acc`` back
        and the station adds the clock, so the generator component nets
        to its design-clock phase -- it appears at MINUS the station
        clock relative to the actual RF, the physical walk-off of a
        design-locked drive under an RF-frequency offset and under a
        phase-loop step alike (see :meth:`_write_station_readout` and
        :meth:`_absorb_phase_loop_step`).

        The kick-frame rotation ``exp(+i (gap + phi_acc))`` rotates the
        demodulation-frame sum into the frame of the applied kick; the PI
        error is formed there, so the loop regulates the voltage the
        station actually applies.

        The PI-error rotation ``exp(+i phi_clock)`` then takes that
        error into the ACTUATOR frame. The controller returns a generator
        current, which drives the design-anchored generator component, so
        ``d(V_kick) / d(I_gen)`` carries the composition's
        ``exp(-i phi_clock)``; rotating the error back cancels it, and
        the open-loop gain stays real instead of turning with the station
        clock. Note the ``gap`` and ``phi_acc`` halves cancel between the two
        rotations, which is why this third one uses the station clock
        alone.

        **Which** ``phi_acc``. The forward span, the fine grid and the
        readout use the forward segment's accumulated phase: the scalars
        ``_generator_frame_rotation`` and ``_kick_frame_rotation``. The
        backfill span replays the interval since this station's previous
        passage, over which the phase is still accumulating, so every
        backfill cell takes the phase accumulated up to THAT cell
        (:meth:`~blond.physics.feedbacks.rf_center_grid.RFCenterGridMixin._backfill_center_phases`),
        running from the previous passage's phase to this one's:
        ``_backfill_generator_frame_rotations`` and
        ``_backfill_kick_frame_rotations``, one entry per backfill centre.
        With the passage's final phase there instead, the beam-induced part
        of the carried voltage would be rotated by the phase still to
        accumulate, and the kick-frame voltage would jump by one passage's
        increment where the previous forward span hands over to this
        backfill span. Both rotations of a cell use the same phase, so the
        generator component still nets to its design-clock phase on every
        cell. The backfill cells also compose with the phase-loop offset
        in force BEFORE this passage (``_phi_rf_loop_seen``): they replay
        the interval before a step this passage absorbs.

        The first two are exactly ``1 + 0j`` without an RF-frequency
        offset, without a phase-loop offset and without multi-section
        acceleration, on every backfill cell too; the third is
        exactly ``1 + 0j`` whenever the station clock is zero, independently
        of ``gap`` and ``phi_acc`` (the zero short-circuits keep the unrotated
        path free of ``exp`` sign dust).

        Notes
        -----
        ORDERING: needs the per-passage station clock (``delta_phi_rf``
        and ``phi_rf_loop``; the backfill cells read ``_phi_rf_loop_seen``
        still unchanged, so this must precede
        :meth:`_absorb_phase_loop_step`), the
        completed ``_kick_clock_slip_gap`` and ``_carrier_slip_gap`` of
        this passage and its complete grid (the backfill segments are
        ``_segments[:-1]``); must precede every :meth:`circuit_track` of the
        passage, whose per-cell sum composition and PI error read the
        rotations off the instance.
        """
        # The station clock: the kick clock accumulated from the
        # RF-frequency offset plus the per-station phase-loop offset, both
        # applied by the station through ``phi_rf`` and both walking the
        # design-anchored generator component off the actual RF.
        station_clock = self.delta_phi_rf + self.phi_rf_loop
        total_generator_slip = station_clock + self._carrier_slip_gap
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
        # The same two rotations per backfill cell, with the phase
        # accumulated up to the cell in place of the forward segment's, and
        # the same exact-zero short-circuit: an unrotated passage composes
        # and regulates every backfill cell with exactly 1 + 0j.
        backfill_carrier_slip_gaps = (
            self._kick_clock_slip_gap + self._backfill_center_phases()
        )
        # The backfill span replays the interval BEFORE this passage, so
        # it carries the phase-loop offset that was in force then (see
        # ``_absorb_phase_loop_step``).
        backfill_generator_slips = (
            self.delta_phi_rf
            + self._phi_rf_loop_seen
            + backfill_carrier_slip_gaps
        )
        self._backfill_generator_frame_rotations = np.where(
            backfill_generator_slips == 0.0,
            1.0 + 0.0j,
            np.exp(-1j * backfill_generator_slips),
        )
        self._backfill_kick_frame_rotations = np.where(
            backfill_carrier_slip_gaps == 0.0,
            1.0 + 0.0j,
            np.exp(1j * backfill_carrier_slip_gaps),
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
            if station_clock == 0.0
            else complex(np.exp(1j * station_clock))
        )

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
        PRECONDITION: ``self._carrier_slip_gap`` must already include the
        forward segment's accumulated phase --
        :meth:`calculate_rf_beam_current_partial` reads the attribute
        directly (``carrier_phase_offset = -(phi_rf +
        _carrier_slip_gap)``, with ``phi_rf = phi_rf_design +
        delta_phi_rf``) and :meth:`_write_station_readout` adds
        ``_carrier_slip_gap`` back on top of the ``phi_rf`` the station
        itself applies -- the identical total -- or the
        demodulation/readout chain no longer closes.
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
                  e^{-i(\Delta\phi_\mathsf{rf} + g + \phi_\mathsf{acc})},

        with ``g`` the live kick-clock gap, ``phi_acc`` the forward
        segment's accumulated phase (``carrier_slip_gap = g + phi_acc``) and
        ``delta_phi_rf`` the station kick clock. The station applies
        ``sin(omega_rf ts + phi_rf_design + delta_phi_rf +
        phase_correction)`` with ``phase_correction = angle(V) +
        carrier_slip_gap``, so each component nets, relative to the
        design RF wave ``omega_rf ts + phi_rf_design``:

        - beam component: ``angle(V_beam) + delta_phi_rf + g + phi_acc``
          against that wave, i.e. ``angle(V_beam) + phi_rf + g + phi_acc``
          in absolute phase -- exactly the total its demodulation
          subtracted (``carrier_phase_offset = -(phi_rf +
          _carrier_slip_gap)``, with ``phi_rf = phi_rf_design +
          delta_phi_rf``; see
          :meth:`calculate_rf_beam_current_partial`); the station
          supplies the ``phi_rf`` half and this readout the ``g + phi_acc``
          half, so the chain closes for every carried deposit,
          byte-for-byte as before the split;
        - generator component: ``angle(V_gen) + 0`` -- design-locked, as
          the klystron drive follows the design frequency. Relative to
          the ACTUAL RF (which leads the design carrier by the kick-clock
          slip ``delta_phi_rf + g``) the driven field therefore appears at
          MINUS that slip: the physical walk-off of a design-locked drive
          under an RF-frequency offset. Without an offset and without
          multi-section acceleration ``delta_phi_rf``, ``g`` and ``phi_acc``
          are all zero, and a driven, beam-free cavity on its setpoint
          reads out ``phase_correction == 0`` -- the feedback is a no-op.
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

    def _state_before_forward_span(
        self, forward_start: int
    ) -> tuple[complex, complex]:
        """
        Coarse state at the centre preceding the forward segment.

        The fine solve is seeded there rather than at the first forward
        centre, so that no deposit of this passage sits inside its own
        initial condition. With backfill cells that centre is the last
        backfill one; without them it is the state carried across the
        passage boundary.

        The two source-split components are composed with the FORWARD
        passage's generator rotation -- the frame the fine grid, and the
        forward span it continues, run in -- rather than with the rotation
        of the backfill cell the state is taken from.

        Parameters
        ----------
        forward_start
            Whole-turn coarse index of the first forward cell.

        Returns
        -------
        seed_voltage
            Demodulation-frame antenna voltage at that centre [V].
        held_generator_current
            Generator command held over the step into the first forward
            cell [A], in the design frame the commands are recorded in.
        """
        if forward_start > 0:
            voltage_beam = self.antenna_voltage_beam_coarse_grid[
                forward_start - 1
            ]
            voltage_gen = self.antenna_voltage_gen_coarse_grid[
                forward_start - 1
            ]
            held_generator_current = self.generator_current_coarse_grid[
                forward_start - 1
            ]
        else:
            voltage_beam = self._last_val_ant_voltage_beam
            voltage_gen = self._last_val_ant_voltage_gen
            held_generator_current = self._last_val_generator_current
        return (
            complex(
                voltage_beam + voltage_gen * self._generator_frame_rotation
            ),
            complex(held_generator_current),
        )

    def _resolve_fine_grid_voltage(self, omega_input: float) -> None:
        """
        Resolve this passage's forward segment onto the fine (profile) grid.

        The second half of :meth:`circuit_track`, run only when the
        segment carries beam. The coarse recursion has just filled the
        forward segment; the fine solve is seeded from the coarse state at
        the centre BEFORE its first cell
        (:meth:`_state_before_forward_span`), propagated to
        ``profile.cut_left`` with the recorded generator drive, and the
        beam-loaded fine response is then integrated at histogram centres.

        Seeding before the span rather than at the first forward centre is
        what lets that first cell carry charge: its deposit drives the
        coarse step into its own centre once, and the fine solve
        integrates the same charge once. A seed taken AT that centre would
        already contain the cell's own deposit, which is why the window
        used to be constrained to start at or after it with a charge-free
        first cell. The seed centre is never later than ``cut_left`` -- it
        lies at or before the passage origin, which ``cut_left`` is
        asserted to be past -- so the propagation always runs forward, and
        no later coarse voltage enters the seed.

        Writes ``generator_current_fine_grid`` (the interpolation) and, via
        :meth:`cavity_response_fine`, ``antenna_voltage_fine_grid``.

        Parameters
        ----------
        omega_input
            Frequency of the segment just tracked [rad/s]; sets both the
            fine-grid step phase ``omega * profile.hist_step`` and the
            normalisation of the cavity detuning.
        """
        init_beam_time = self.profile.cut_left
        assert init_beam_time > 0, (
            f"{init_beam_time=} has to be > 0, shift profile."
        )

        forward_start = len(self._rf_centers) - self._rf_centers_lengths[-1]
        seed_voltage, held_generator_current = self._state_before_forward_span(
            forward_start
        )
        # Coarse command i drives the interval AFTER centre i, so the
        # command held over the step into the first forward cell is the
        # one of the seed centre: prepending both makes the two arrays
        # cover the whole interval the fine window may start in.
        # Reconstructing that interval with these already-computed
        # commands is what keeps the PI from being stepped a second time,
        # which would advance its delay and integral twice.
        rf_centers = np.concatenate(
            (
                [
                    self._rf_centers[forward_start]
                    - self._step_into_first_cell(
                        omega_input, forward_start, len(self._rf_centers)
                    )
                ],
                self._rf_centers[forward_start:],
            )
        )
        generator_current = np.concatenate(
            (
                [held_generator_current],
                self.generator_current_coarse_grid[forward_start:],
            )
        )
        if self._controller is not None:
            generator_current = self._controller.limit(generator_current)

        generator_current_in_frame = (
            generator_current * self._generator_frame_rotation
        )
        antenna_voltage_init = propagate_beam_free_voltage(
            initial_voltage=seed_voltage,
            generator_current=generator_current_in_frame,
            rf_centers=rf_centers,
            end_time=init_beam_time,
            omega=omega_input,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            delta_omega=self.delta_omega,
        )
        initial_current_index = np.clip(
            np.searchsorted(rf_centers, init_beam_time, side="right") - 1,
            0,
            len(rf_centers) - 1,
        )
        generator_current_init = generator_current[initial_current_index]

        omega_times_dt_fine_grid = omega_input * self.profile.hist_step
        # copy_to_cpu: the feedback signal processing is host-side
        # (scipy), so a GPU-backend profile grid must be brought to host.
        self.generator_current_fine_grid = np.interp(
            copy_to_cpu(self.profile.hist_x),
            rf_centers,
            generator_current,
        )

        relative_detuning = self.delta_omega / omega_input
        self.cavity_response_fine(
            initial_voltage_fine_grid=antenna_voltage_init,
            initial_generator_current_fine_grid=generator_current_init,
            omega_times_dt_fine_grid=omega_times_dt_fine_grid,
            relative_detuning=relative_detuning,
            initial_at_bin_edge=True,
        )

    def cavity_response_fine(
        self,
        initial_voltage_fine_grid: float,  # TODO: these should all also be complex
        initial_generator_current_fine_grid: float,
        omega_times_dt_fine_grid: float,
        relative_detuning: float,
        *,
        initial_at_bin_edge: bool = False,
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
        initial_at_bin_edge
            True when the initial state is at ``profile.cut_left`` and
            returned voltages must lie at histogram centres. False keeps
            the direct-call convention: the seed precedes the first
            output sample by one full fine step.
        """
        # No actuator clamp here. The controller clamps every coarse command
        # it returns, and the fine-grid current and its initial value are
        # linear interpolations of those commands: a straight line between
        # two points inside the limit circle cannot leave it. A second clamp
        # measured as a no-op (changes of ~1e-17 A on a 0.05 A current, with
        # 98 % of the coarse cells at the limit).

        # The fine solve runs in the DEMODULATION frame: its seed (the
        # state propagated from the first forward coarse cell) carries the
        # generator component rotated by the generator frame rotation,
        # and its beam current was demodulated in that frame. The raw
        # (design-frame) generator current is rotated the same way into
        # LOCAL inputs -- the solve is linear, so this reproduces the
        # superposition of the two per-component fine solutions exactly.
        # The public ``generator_current_fine_grid`` stays the raw
        # (klystron-limited) design-frame current.
        generator_current_fine_grid = (
            self.generator_current_fine_grid * self._generator_frame_rotation
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
            initial_at_bin_edge=initial_at_bin_edge,
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
        ONLY free phase left in the implemented beam-loading expression.
        Its energy-loss sign requires ``cos(omega_c * dT) < 0``; recovering
        the full magnitude and phase in this convention requires
        ``omega_c * dT == pi`` (mod ``2 pi``) -- the value the segment
        tiling delivers on every passage, turn 0 included (see
        :meth:`~blond.physics.feedbacks.rf_center_grid.RFCenterGridMixin._close_previous_turn_grid`).
        The physical theorem itself does not prescribe this grid offset.

        The demodulation does not consume this product: the frame is
        stated as exactly ``pi`` (``demodulation_phase`` of
        :func:`~blond.physics.feedbacks.beam_current.rf_beam_current`), so
        the residual's frequency lag under a ramp never reaches the beam
        current. This check is what keeps the two consistent -- a grid
        whose own frame is not ``pi`` does not sit where the stated frame
        assumes, and its deposit lands at the wrong phase in the bucket.

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
            "The coarse grid is not aligned with the RF bucket. The "
            "beam current is demodulated at the convention value pi, and "
            "the grid must sit there too: omega_c * dT == pi (mod 2 pi). "
            f"Here omega_c * dT = {theta / np.pi:.9f} pi, off by "
            f"{deviation / np.pi:.9f} pi, so the deposit lands at that "
            "angle in the bucket -- and beyond 0.5 pi of offset "
            f"(cos(omega_c * dT) = {np.cos(theta):+.6f}) the beam-induced "
            "voltage is sign-inverted, so the bunch would be ACCELERATED "
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
            omega_c=self._forward_segment_omega_design,
            sampling_time=sampling_time_frwrd,
            n_points=n_points,
            # The binning shift only: a physical time, the tail this
            # passage's grid leaves before its first forward centre.
            dT=dT_demodulation,
            # The frame is STATED, not taken from that tail. The
            # convention admits exactly one value, ``pi`` (mod 2 pi), and
            # the grid is built to land on it -- but ``dT_demodulation``
            # is the tail left by the PRECEDING segment while ``omega_c``
            # is THIS segment's carrier, so under a ramp their product is
            # short/long by the fractional per-segment frequency change,
            #     frame lag [pi]  ~  (omega_fwd - omega_prod) / omega ,
            # measured 7.9e-8 pi on RCS1 (the fastest shipped ramp) and up
            # to 3.5e-6 pi on the 4 GeV single-section test ramp. Stating
            # the frame keeps that lag out of the beam-loading phase;
            # ``_assert_demodulation_frame_aligned`` above checks the grid
            # against it, so a grid that does not sit at ``pi`` -- a
            # sub-step other than 0.5, a harmonic that is not a whole
            # number of RF periods per segment -- still raises rather than
            # being silently demodulated at the stated value.
            demodulation_phase=np.pi,
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
            # The step from this passage's last centre into the next
            # passage's grid has no cell of its own, so it carries no beam
            # current, and charge past that centre is rejected outright:
            # the window must stay clear of the end of the coarse grid.
            forbid_charge_in_last_coarse_cell=True,
        )

        # Convert RF beam currents to be in units of Amperes
        self.beam_current_fine_grid = (
            self.beam_current_fine_grid / self.profile.hist_step
        )
        self.beam_current_forward_coarse_grid = (
            self.beam_current_forward_coarse_grid / sampling_time_frwrd
        )
