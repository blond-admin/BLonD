"""Cavity-feedback observation-class and diagnostic-flag tests.

The IQ-cavity-feedback timing-class grid tests and the RFCenterSegment
tests moved to test_rf_center_grid.py and test_rf_center_segment.py when
the grid builder and value class were split into their own modules.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pytest
from _pytest import unittest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Numpy64Bit,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    backend,
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass

HARMONIC = 5
CIRCUMFERENCE = 5
STATION_VOLTAGE = 5e6
INITIAL_ANTENNA_VOLTAGE = 30.0e6

# Design RF period [s] of harmonic HARMONIC at the tracked 63 GeV/c
# (beta ~ 1); the multi-harmonic tests below only place profile windows
# and particles with it, which tolerates the ~1e-6 beta error.
T_RF = CIRCUMFERENCE / 299792458.0 / HARMONIC


class TestIQCavityFeedbackObservationClass(unittest.TestCase):
    pass


def _run_one_turn(**feedback_kwargs) -> IQCavityFeedbackTimingClass:
    """
    Track one turn of a single-section ring with a cavity feedback.

    The cavity is undriven and beam-loading free (``R_over_Q = 0``,
    ``generator_current_bias = 0``), so the antenna voltage simply decays
    from ``initial_voltage``. That is enough to tell a real readout from
    the neutral one: the relative voltage correction is
    ``|V_ant| / station voltage``, i.e. ~6 here, and only the neutral
    readout writes exactly 1.

    Parameters
    ----------
    **feedback_kwargs
        Extra keyword arguments for the
        :class:`IQCavityFeedbackTimingClass` under test.

    Returns
    -------
    feedback
        The tracked feedback, after one turn.
    """
    backend.change_backend(Numpy64Bit)
    profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
    rf_station = SingleHarmonicRFStation(
        phi_rf=0.0, harmonic=HARMONIC, voltage=STATION_VOLTAGE
    )
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    ring.add_elements(
        [
            rf_station,
            DriftSimple(CIRCUMFERENCE, momentum_compaction_factor=0),
        ]
    )

    beam = Beam(intensity=1, particle_type=mu_plus, is_counter_rotating=False)
    beam._dt = DistributedArray(np.zeros(5))
    beam._dE = DistributedArray(np.zeros(5))
    beam._ids = DistributedArray(np.arange(5))
    beam._flags = DistributedArray(np.zeros(5))

    feedback = IQCavityFeedbackTimingClass(
        profile=profile,
        n_rf_periods_per_coarse_grid=1,
        R_over_Q=0,
        Q_L=100,
        generator_current_bias=0,
        n_cavities=1,
        initial_voltage=INITIAL_ANTENNA_VOLTAGE,
        **feedback_kwargs,
    )
    rf_station.attach_cavity_feedback(feedback)

    simulation = Simulation(
        ring,
        ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        ),
    )
    simulation.run_simulation(beam, n_turns=1)
    return feedback


def _is_neutral_readout(feedback: IQCavityFeedbackTimingClass) -> bool:
    """
    Whether the feedback wrote the no-correction readout.

    Parameters
    ----------
    feedback
        Feedback whose station readout to inspect.

    Returns
    -------
    is_neutral
        True when the station is told unit gain and zero phase, i.e. it
        kicks exactly as if no feedback were attached.
    """
    return bool(
        np.all(feedback.relative_voltage_correction == 1.0)
        and np.all(feedback.phase_correction == 0.0)
    )


class TestDiagnosticsDoNotDisableTheFeedback:
    """The diagnostic flags must not switch the physics off."""

    def test_default_applies_a_real_correction(self) -> None:
        feedback = _run_one_turn()

        assert not _is_neutral_readout(feedback)

    def test_diagnostics_still_apply_a_real_correction(self) -> None:
        # ``debug=True`` used to short-circuit _track and write the
        # neutral readout, i.e. turning diagnostics on silently turned
        # the feedback off.
        feedback = _run_one_turn(debug=True)

        assert not _is_neutral_readout(feedback)

    def test_grid_validation_still_applies_a_real_correction(self) -> None:
        feedback = _run_one_turn(validate_grid_each_turn=True)

        assert not _is_neutral_readout(feedback)

    def test_diagnostic_flags_leave_the_readout_bit_identical(self) -> None:
        # Both flags are observation-only, so the tracked result must be
        # bit-for-bit the default one.
        reference = _run_one_turn()
        diagnosed = _run_one_turn(debug=True, validate_grid_each_turn=True)

        np.testing.assert_array_equal(
            diagnosed.relative_voltage_correction,
            reference.relative_voltage_correction,
        )
        np.testing.assert_array_equal(
            diagnosed.phase_correction, reference.phase_correction
        )
        np.testing.assert_array_equal(
            diagnosed.antenna_voltage_coarse_grid,
            reference.antenna_voltage_coarse_grid,
        )

    def test_grid_only_mode_applies_no_correction(self) -> None:
        # The one mode that does switch the physics off; it now says so
        # in its name.
        feedback = _run_one_turn(grid_only_no_correction=True)

        assert _is_neutral_readout(feedback)
        # The grid is still built -- that is the point of the mode.
        assert len(feedback._rf_centers) > 0


def _make_bare_feedback(**feedback_kwargs) -> IQCavityFeedbackTimingClass:
    """
    Build a feedback without a simulation, on a mocked profile.

    Parameters
    ----------
    **feedback_kwargs
        Overrides for the :class:`IQCavityFeedbackTimingClass`
        constructor defaults.

    Returns
    -------
    feedback
        A feedback instance not attached to any RF station.
    """
    params = {
        "profile": Mock(StaticProfile),
        "R_over_Q": 0.0,
        "Q_L": 100.0,
        "generator_current_bias": 0.0,
        "n_cavities": 1,
        "initial_voltage": INITIAL_ANTENNA_VOLTAGE,
        "n_rf_periods_per_coarse_grid": 1,
    }
    params.update(feedback_kwargs)
    return IQCavityFeedbackTimingClass(**params)


class _MultiHarmonicStationStub:
    """
    Parent-station stand-in that is not a SingleHarmonicRFStation.

    Carries per-harmonic arrays, the way a multi-harmonic station does,
    so the RF-parameter properties must index them by ``harmonic_index``.
    """

    def __init__(self):
        self.harmonic = np.array([3.0, 7.0])
        self.omega_rf = np.array([1.1e9, 2.2e9])
        self.delta_phi_rf = None


class TestConstructorHarmonicIndexValidation:
    """
    ``harmonic_index`` handling at feedback construction.

    Mirrors the ``attach_cavity_feedback`` rules: plain ``int``,
    ``np.integer`` and integral ``float`` are accepted (silently), a
    fractional value is a hard error -- a harmonic index is a list
    slot, not a physical quantity to be rounded.
    """

    def test_fractional_harmonic_index_raises(self) -> None:
        with pytest.raises(ValueError, match="1.5"):
            _make_bare_feedback(harmonic_index=1.5)

    def test_integral_float_harmonic_index_is_accepted_silently(
        self,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            feedback = _make_bare_feedback(harmonic_index=1.0)
        assert feedback.harmonic_index == 1
        assert type(feedback.harmonic_index) is int

    def test_numpy_integer_harmonic_index_is_accepted(self) -> None:
        # np.int64 indexes per-harmonic arrays fine, but is not an
        # ``int``; the coercion must not reject it.
        feedback = _make_bare_feedback(harmonic_index=np.int64(1))
        assert feedback.harmonic_index == 1

    def test_non_numeric_harmonic_index_raises(self) -> None:
        with pytest.raises(TypeError):
            _make_bare_feedback(harmonic_index="1")


class TestMultiHarmonicParentResolution:
    """RF-parameter accessors on a multi-harmonic parent station."""

    def _feedback_with_stub_parent(self) -> IQCavityFeedbackTimingClass:
        feedback = _make_bare_feedback()
        # Assigned directly: set_parent_rf_station() rejects anything but
        # the real station classes, and only the isinstance dispatch of
        # _resolve_main_harmonic is under test here.
        feedback._parent_rf_station = _MultiHarmonicStationStub()
        feedback.harmonic_index = 1
        return feedback

    def test_harmonic_indexes_the_per_harmonic_array(self) -> None:
        feedback = self._feedback_with_stub_parent()

        assert feedback.harmonic == 7.0

    def test_resolve_main_harmonic_indexes_the_value(self) -> None:
        # omega_rf goes through _resolve_main_harmonic, which must pick
        # the tracked harmonic out of the per-harmonic array.
        feedback = self._feedback_with_stub_parent()

        assert feedback.omega_rf == 2.2e9

    def test_delta_phi_rf_is_zero_before_any_slip(self) -> None:
        # The parent's kick clock is None before the first passage; the
        # accessor must report "no accumulated slip", not crash.
        feedback = self._feedback_with_stub_parent()

        assert feedback.delta_phi_rf == 0.0

    def test_delta_phi_rf_indexes_the_per_harmonic_array(self) -> None:
        feedback = self._feedback_with_stub_parent()
        feedback._parent_rf_station.delta_phi_rf = np.array([0.5, 1.5])

        assert feedback.delta_phi_rf == 1.5


def _make_two_harmonic_station(
    second_harmonic_voltage: float = 0.5 * STATION_VOLTAGE,
) -> MultiHarmonicRFStation:
    """
    Build the two-harmonic RF station the multi-harmonic tests share.

    Parameters
    ----------
    second_harmonic_voltage
        Voltage [V] of the second harmonic; ``0`` makes the station
        physically degenerate with the single-harmonic one.

    Returns
    -------
    rf_station
        A two-harmonic station at harmonics ``[HARMONIC, 2 * HARMONIC]``
        with no feedback attached yet.
    """
    return MultiHarmonicRFStation(
        n_harmonics=2,
        main_harmonic_idx=0,
        voltage=np.array([STATION_VOLTAGE, second_harmonic_voltage]),
        phi_rf=np.array([0.0, 0.0]),
        harmonic=np.array([HARMONIC, 2 * HARMONIC], dtype=float),
    )


def _run_simulation_turns(rf_station, n_turns: int) -> Beam:
    """
    Track ``n_turns`` of a one-station ring around ``rf_station``.

    Same ring and magnetic cycle as ``_run_one_turn``, but the station
    (and its already-attached feedbacks) is the caller's, and the beam
    carries real intensity with its particles spread INSIDE the profile
    window, so the RF kick, the beam loading and the resulting
    corrections are all non-trivial.

    Parameters
    ----------
    rf_station
        RF station (with feedbacks already attached) to track.
    n_turns
        Number of turns to track.

    Returns
    -------
    beam
        The tracked beam, after ``n_turns``.
    """
    backend.change_backend(Numpy64Bit)
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    ring.add_elements(
        [
            rf_station,
            DriftSimple(CIRCUMFERENCE, momentum_compaction_factor=0),
        ]
    )

    beam = Beam(
        intensity=1e12, particle_type=mu_plus, is_counter_rotating=False
    )
    # One RF period into the window (which starts at 0.75 T_RF), so no
    # charge lies in the first coarse-grid cell (rf_beam_current rejects
    # that -- the fine grid would double-count its kick).
    beam._dt = DistributedArray(np.linspace(1.1, 1.9, 5) * T_RF)
    beam._dE = DistributedArray(np.zeros(5))
    beam._ids = DistributedArray(np.arange(5))
    beam._flags = DistributedArray(np.zeros(5))

    simulation = Simulation(
        ring,
        ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        ),
    )
    simulation.run_simulation(beam, n_turns=n_turns)
    return beam


def _make_full_run_feedback(
    profile: StaticProfile, **feedback_kwargs
) -> IQCavityFeedbackTimingClass:
    """
    Build a feedback for the full-simulation multi-harmonic tests.

    Parameters
    ----------
    profile
        Profile the feedback acts on.
    **feedback_kwargs
        Overrides for the constructor defaults.

    Returns
    -------
    feedback
        A feedback instance not attached to any RF station.
    """
    params = {
        "profile": profile,
        "R_over_Q": 518.0,
        "Q_L": 100.0,
        "generator_current_bias": 0.0,
        "n_cavities": 1,
        "initial_voltage": INITIAL_ANTENNA_VOLTAGE,
        "n_rf_periods_per_coarse_grid": 1,
    }
    params.update(feedback_kwargs)
    return IQCavityFeedbackTimingClass(**params)


class TestDegenerateMultiHarmonicMatchesSingleHarmonic:
    """
    The physics anchor of the multi-harmonic-station support.

    A two-harmonic station whose second harmonic has zero voltage is
    physically a single-harmonic station, so a feedback regulating slot 0
    must reproduce the equivalent ``SingleHarmonicRFStation`` run to near
    machine precision: same antenna voltage, same corrections, same
    applied kick.
    """

    N_TURNS = 3

    @classmethod
    def _run(cls, multi_harmonic: bool):
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 64, T_RF)
        feedback = _make_full_run_feedback(profile)
        if multi_harmonic:
            rf_station = _make_two_harmonic_station(
                second_harmonic_voltage=0.0
            )
            rf_station.attach_cavity_feedback(feedback, harmonic_index=0)
        else:
            rf_station = SingleHarmonicRFStation(
                phi_rf=0.0,
                harmonic=HARMONIC,
                voltage=STATION_VOLTAGE,
                cavity_feedback=feedback,
            )
        beam = _run_simulation_turns(rf_station, n_turns=cls.N_TURNS)
        return feedback, beam

    @classmethod
    def setup_class(cls) -> None:
        cls.shc_feedback, cls.shc_beam = cls._run(multi_harmonic=False)
        cls.mhc_feedback, cls.mhc_beam = cls._run(multi_harmonic=True)

    def test_antenna_voltage_matches(self) -> None:
        np.testing.assert_allclose(
            self.mhc_feedback.antenna_voltage_coarse_grid,
            self.shc_feedback.antenna_voltage_coarse_grid,
            rtol=1e-12,
            atol=0.0,
        )

    def test_corrections_match(self) -> None:
        np.testing.assert_allclose(
            self.mhc_feedback.relative_voltage_correction,
            self.shc_feedback.relative_voltage_correction,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            self.mhc_feedback.phase_correction,
            self.shc_feedback.phase_correction,
            rtol=1e-12,
            atol=1e-15,
        )

    def test_applied_kick_matches(self) -> None:
        # The kick the beam actually received over the tracked turns.
        np.testing.assert_allclose(
            np.asarray(self.mhc_beam._dE.array_local),
            np.asarray(self.shc_beam._dE.array_local),
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(self.mhc_beam._dt.array_local),
            np.asarray(self.shc_beam._dt.array_local),
            rtol=1e-12,
            atol=0.0,
        )

    def test_correction_is_real_not_neutral(self) -> None:
        # Guards the anchor against passing vacuously with a switched-off
        # feedback.
        assert not _is_neutral_readout(self.mhc_feedback)


class TestNonMainHarmonicAttachment:
    """
    A feedback attached only at a non-zero slot runs end to end.

    Slot 0 stays empty, so every former ``cavity_feedback_list[0]``
    hardcode in ``MultiHarmonicRFStation`` would crash with
    ``'NoneType' object has no attribute 'profile'``; and the coarse grid
    must be built from harmonic 1's design frequency, not harmonic 0's.
    """

    N_TURNS = 2

    @classmethod
    def setup_class(cls) -> None:
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 64, T_RF)
        cls.feedback = _make_full_run_feedback(profile, harmonic_index=1)
        cls.rf_station = _make_two_harmonic_station()
        cls.rf_station.attach_cavity_feedback(cls.feedback, harmonic_index=1)
        cls.beam = _run_simulation_turns(cls.rf_station, n_turns=cls.N_TURNS)

    def test_runs_and_applies_a_real_correction(self) -> None:
        assert not _is_neutral_readout(self.feedback)

    def test_grid_frequency_is_harmonic_1_design_frequency(self) -> None:
        # The grid must run on THIS feedback's harmonic -- a scalar, not
        # the station's per-harmonic array (whose slot 0 is the main
        # harmonic).
        omega_design = self.rf_station.calc_omega_rf_design(
            self.beam.reference.beta, CIRCUMFERENCE
        )
        assert np.ndim(self.feedback._forward_segment_omega_design) == 0
        np.testing.assert_allclose(
            self.feedback._forward_segment_omega_design,
            omega_design[1],
            rtol=1e-15,
        )


class TestAttachSetsHarmonicIndexFromSlot:
    """
    The blessed convenience case of slot-authoritative attachment.

    A feedback constructed with the DEFAULT ``harmonic_index`` (0) and
    attached at slot 1 must have its index overwritten from the slot,
    run end to end, and read harmonic 1's RF parameters -- no
    construct-time index bookkeeping required of the user.
    """

    N_TURNS = 2

    @classmethod
    def setup_class(cls) -> None:
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 64, T_RF)
        # Deliberately NOT constructed with harmonic_index=1.
        cls.feedback = _make_full_run_feedback(profile)
        cls.rf_station = _make_two_harmonic_station()
        cls.rf_station.attach_cavity_feedback(cls.feedback, harmonic_index=1)
        cls.beam = _run_simulation_turns(cls.rf_station, n_turns=cls.N_TURNS)

    def test_attach_overwrites_the_constructor_index(self) -> None:
        assert self.feedback.harmonic_index == 1

    def test_runs_and_applies_a_real_correction(self) -> None:
        assert not _is_neutral_readout(self.feedback)

    def test_reads_harmonic_1_rf_parameters(self) -> None:
        # The coarse grid must run on the attached slot's harmonic, not
        # on the constructor default's (the main harmonic at slot 0).
        omega_design = self.rf_station.calc_omega_rf_design(
            self.beam.reference.beta, CIRCUMFERENCE
        )
        np.testing.assert_allclose(
            self.feedback._forward_segment_omega_design,
            omega_design[1],
            rtol=1e-15,
        )


class TestHarmonicSlotAgreementIsEnforcedAtRunStart:
    """
    The feedback's ``harmonic_index`` must equal its list slot.

    ``calc_gap_voltage_with_feedbacks`` applies each feedback's
    corrections at its LIST index, while the feedback computes them from
    the RF parameters at its OWN ``harmonic_index``. A disagreement would
    silently apply corrections computed from harmonic A to harmonic B, so
    it is rejected at run start.

    ``attach_cavity_feedback`` now SETS the feedback's index from the
    slot, so a mismatch cannot arise through the attach; the run-start
    guard is reached only by tampering with ``cavity_feedback_list``
    after the attach -- which is what these tests set up, and what the
    attach path cannot see.
    """

    def test_feedback_harmonic_1_in_slot_0_raises(self) -> None:
        feedback = _make_bare_feedback(harmonic_index=1)
        station = _make_two_harmonic_station()
        station.attach_cavity_feedback(feedback, harmonic_index=1)
        station.cavity_feedback_list = [feedback, None]

        with pytest.raises(ValueError) as excinfo:
            feedback.on_run_simulation(
                simulation=Mock(), beam=Mock(), n_turns=1
            )

        message = str(excinfo.value)
        assert "harmonic_index=1" in message
        assert "slot 0" in message
        # The remedy must be followable: re-attaching an already-owned
        # feedback trips the ownership assert, so the message must point
        # at construction-time fixes instead.
        assert "Construct the feedback with harmonic_index=0" in message
        assert "attach_cavity_feedback" not in message

    def test_feedback_harmonic_0_in_slot_1_raises(self) -> None:
        feedback = _make_bare_feedback()
        station = _make_two_harmonic_station()
        station.attach_cavity_feedback(feedback, harmonic_index=0)
        station.cavity_feedback_list = [None, feedback]

        with pytest.raises(ValueError) as excinfo:
            feedback.on_run_simulation(
                simulation=Mock(), beam=Mock(), n_turns=1
            )

        message = str(excinfo.value)
        assert "harmonic_index=0" in message
        assert "slot 1" in message

    def test_matching_slot_passes_the_guard(self) -> None:
        feedback = _make_bare_feedback(harmonic_index=1)
        _make_two_harmonic_station().attach_cavity_feedback(
            feedback, harmonic_index=1
        )

        # Guard only: on_run_simulation would go on to need a real
        # simulation, which the full-tracking tests above provide.
        feedback._validate_multi_harmonic_slot()

    def test_feedback_missing_from_parent_list_raises(self) -> None:
        # A parent whose cavity_feedback_list does not contain this
        # feedback at all (identity check).
        feedback = _make_bare_feedback()
        feedback.set_parent_rf_station(rf_station=_make_two_harmonic_station())

        with pytest.raises(ValueError) as excinfo:
            feedback.on_run_simulation(
                simulation=Mock(), beam=Mock(), n_turns=1
            )

        assert "cavity_feedback_list" in str(excinfo.value)


def _prepare_hand_built_grid(
    feedback: IQCavityFeedbackTimingClass, rf_centers
) -> None:
    """
    Install a hand-built coarse grid and size the IQ arrays for it.

    Parameters
    ----------
    feedback
        Feedback to prepare.
    rf_centers
        Coarse-grid centre times [s] to install.
    """
    feedback._rf_centers = np.asarray(rf_centers, dtype=float)
    feedback._rf_centers_lengths = np.array([len(feedback._rf_centers)])
    feedback.reset_arrays()


class TestCoarseCellStepSizing:
    """Per-cell step sizing of the coarse-grid recursion."""

    # Arbitrary segment frequency; rf_period = 2 pi / omega = 1 s makes
    # the centre times below easy to read.
    omega_input = 2 * np.pi

    def test_single_cell_first_turn_uses_own_coarse_step(self) -> None:
        # First centre ever tracked AND the only centre of the segment:
        # there is no next centre to diff against, so the step proxy must
        # fall back to this segment's own coarse step (n * t_rf).
        single = _make_bare_feedback()
        _prepare_hand_built_grid(single, [0.5])
        single._circuit_track_cells_python(
            self.omega_input, no_beam=True, start_index=0, end_index=1
        )

        # Reference: a two-centre grid whose spacing IS one coarse step;
        # its first cell diffs the two centres and must give the same
        # step, hence the same first-cell voltage.
        step = 2 * np.pi / self.omega_input
        reference = _make_bare_feedback()
        _prepare_hand_built_grid(reference, [0.5, 0.5 + step])
        reference._circuit_track_cells_python(
            self.omega_input, no_beam=True, start_index=0, end_index=2
        )

        assert (
            single.antenna_voltage_coarse_grid[0]
            == reference.antenna_voltage_coarse_grid[0]
        )
        # Non-vacuous: the cell decayed away from the initial voltage.
        assert single.antenna_voltage_coarse_grid[0] != 0
        assert single.antenna_voltage_coarse_grid[0] != INITIAL_ANTENNA_VOLTAGE

    def test_ulp_negative_first_step_is_clamped_and_duplicated(self) -> None:
        # A first-cell step a few ULPs below zero (a centre landing almost
        # exactly on the segment boundary) is floating-point noise, not an
        # ordering violation: it must be clamped to zero and handled as a
        # coincident point, not trip the hard assertion. A coincident FIRST
        # cell has no predecessor in the grid, so the state it duplicates
        # is the value carried over the turn boundary.
        feedback = _make_bare_feedback(
            R_over_Q=518.0, generator_current_bias=0.01
        )
        _prepare_hand_built_grid(feedback, [-1e-12, 1.0])
        feedback._last_rf_centers_entry = 123.0  # not the first turn
        carried_voltage = feedback._last_val_ant_voltage
        carried_current = feedback._last_val_generator_current

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            feedback._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=2
            )

        # The clamped cell carries the turn-boundary state unchanged...
        assert feedback.antenna_voltage_coarse_grid[0] == carried_voltage
        assert feedback.generator_current_coarse_grid[0] == carried_current
        # ...and tracking completed: the next cell was still advanced.
        assert feedback.antenna_voltage_coarse_grid[1] != 0

    def test_coincident_centers_warn_and_duplicate_previous(self) -> None:
        # A duplicated rf_centers value carries zero elapsed time, so the
        # correct antenna voltage at that cell is exactly the previous
        # cell's: V(t + 0) = V(t). The cell must therefore be duplicated,
        # not left at the zeros prefill (which would restart the envelope
        # from V = 0 and destroy the coherent cavity voltage).
        r_over_q = 518.0
        bias = 0.01
        feedback = _make_bare_feedback(
            R_over_Q=r_over_q, generator_current_bias=bias
        )
        _prepare_hand_built_grid(feedback, [0.25, 0.25, 0.5])
        feedback._last_rf_centers_entry = 123.0  # not the first turn

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            feedback._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        assert feedback.antenna_voltage_coarse_grid[0] != 0
        # The coincident cell holds the previous cell's state...
        assert (
            feedback.antenna_voltage_coarse_grid[1]
            == feedback.antenna_voltage_coarse_grid[0]
        )
        assert (
            feedback.generator_current_coarse_grid[1]
            == feedback.generator_current_coarse_grid[0]
        )
        # ...so the following cell advances from the CARRIED voltage.
        omega_times_dt = self.omega_input * 0.25
        expected = feedback._advance_coarse_voltage(
            v_prev=feedback.antenna_voltage_coarse_grid[0],
            generator_current=feedback.generator_current_coarse_grid[1],
            beam_current=0,
            omega_times_dt=omega_times_dt,
            relative_detuning=0.0,
        )
        assert feedback.antenna_voltage_coarse_grid[2] == expected
        # Non-vacuous: this is NOT the pure drive term the old
        # propagate-from-zero behaviour produced.
        assert feedback.antenna_voltage_coarse_grid[2] != (
            r_over_q * omega_times_dt * bias
        )

    def test_coincident_last_cell_does_not_poison_the_next_turn(self) -> None:
        # reset_arrays carries antenna_voltage_coarse_grid[-1] into the
        # next turn. A coincident LAST cell must therefore hold the real
        # voltage, not the zeros prefill, or the whole next turn starts
        # from a dead cavity.
        feedback = _make_bare_feedback(
            R_over_Q=518.0, generator_current_bias=0.01
        )
        _prepare_hand_built_grid(feedback, [0.25, 0.5, 0.5])
        feedback._last_rf_centers_entry = 123.0  # not the first turn

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            feedback._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )
        last_voltage = feedback.antenna_voltage_coarse_grid[1]
        last_current = feedback.generator_current_coarse_grid[1]
        feedback.reset_arrays()

        assert feedback._last_val_ant_voltage == last_voltage
        assert feedback._last_val_generator_current == last_current
        assert feedback._last_val_ant_voltage != 0

    def test_backfill_cells_hold_the_last_commanded_current(self) -> None:
        # The backfill cells replay an interval that has ALREADY elapsed,
        # during which the klystron kept running at whatever command the
        # controller last issued -- it did not snap back to the feedforward
        # bias. Seeding them with the bias instead discards any standing
        # compensation current the loop holds.
        #
        # Why no existing test caught this: every tracked configuration runs
        # a matched bias at delta_omega = 0, where the compensation current
        # the PI settles on IS the bias, so held and bias coincide. They
        # part company exactly when the cavity is detuned, which is the
        # combination the suite never exercised -- so the fixture below
        # deliberately makes held != bias.
        bias = 0.01
        held = 0.037 + 0.011j
        feedback = _make_bare_feedback(generator_current_bias=bias)
        _prepare_hand_built_grid(feedback, [0.25, 0.5, 0.75, 1.0])
        # A previous turn exists, ending on the held command.
        feedback.generator_current_coarse_grid[-1] = held

        feedback.reset_arrays(n_backfill_cells=2)

        np.testing.assert_array_equal(
            feedback.generator_current_coarse_grid[:2], [held, held]
        )
        np.testing.assert_array_equal(
            feedback.generator_current_coarse_grid[2:], [bias, bias]
        )
        # Non-vacuous: filling the whole grid with the bias -- the
        # behaviour before the hold was introduced -- would fail above.
        assert held != bias

    def test_without_backfill_cells_the_grid_is_all_bias(self) -> None:
        # The default (no backfill segments this turn) must leave the whole
        # grid on the feedforward bias, so a constant-current run without a
        # controller stays bit-identical to the pre-hold behaviour.
        bias = 0.01
        feedback = _make_bare_feedback(generator_current_bias=bias)
        _prepare_hand_built_grid(feedback, [0.25, 0.5, 0.75])
        feedback.generator_current_coarse_grid[-1] = 0.5 + 0.25j

        feedback.reset_arrays()

        np.testing.assert_array_equal(
            feedback.generator_current_coarse_grid, [bias, bias, bias]
        )

    def test_vectorised_first_turn_step_matches_reference_path(self) -> None:
        # _coarse_step_sizes is the vectorised twin of the reference loop
        # and must reproduce the first-turn single-cell fallback step.
        feedback = _make_bare_feedback()
        _prepare_hand_built_grid(feedback, [0.5])

        delta_t = feedback._coarse_step_sizes(self.omega_input, 0, 1)

        np.testing.assert_array_equal(delta_t, [2 * np.pi / self.omega_input])

    def test_degenerate_segment_defers_to_the_reference_path(self) -> None:
        # A coincident (zero) step makes the vectorised sizing return
        # None, and the kernel path must then fall back to the reference
        # loop -- reproducing its warning and its result exactly.
        def _degenerate_feedback() -> IQCavityFeedbackTimingClass:
            feedback = _make_bare_feedback(
                R_over_Q=518.0, generator_current_bias=0.01
            )
            _prepare_hand_built_grid(feedback, [0.25, 0.25, 0.5])
            feedback._last_rf_centers_entry = 123.0
            return feedback

        kernel = _degenerate_feedback()
        assert kernel._coarse_step_sizes(self.omega_input, 0, 3) is None

        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            kernel._circuit_track_cells_kernel(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        reference = _degenerate_feedback()
        with pytest.warns(
            UserWarning, match="double taking of rf_centers value"
        ):
            reference._circuit_track_cells_python(
                self.omega_input, no_beam=True, start_index=0, end_index=3
            )

        np.testing.assert_array_equal(
            kernel.antenna_voltage_coarse_grid,
            reference.antenna_voltage_coarse_grid,
        )
        # Non-vacuous: something was tracked.
        assert np.any(kernel.antenna_voltage_coarse_grid != 0)

    def test_kernel_empty_span_is_a_no_op(self) -> None:
        # start_index == end_index: nothing to track. The kernel must
        # return before sizing the (empty) span -- proceeding would index
        # into a zero-length step array -- and leave the grids untouched.
        feedback = _make_bare_feedback()
        _prepare_hand_built_grid(feedback, [0.5, 1.5, 2.5])

        feedback._circuit_track_cells_kernel(
            self.omega_input, no_beam=True, start_index=2, end_index=2
        )

        np.testing.assert_array_equal(
            feedback.antenna_voltage_coarse_grid, np.zeros(3)
        )
