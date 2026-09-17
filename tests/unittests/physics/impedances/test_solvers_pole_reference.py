"""Independent reference for `MultiPoleSparseSolve`'s induced voltage.

The solver splits the wake into a near field (closed form, three taps) and
a far field (a per-pole recursion). This module rebuilds the same voltage
the slow, obvious way -- a direct sum of the exact bin-averaged wake over
every pair of bins -- so a refactor of that split can be judged against
something other than the solver's own previous output.

This is a characterisation test: it is expected to be GREEN on today's
(unmodified) solver. If it goes red, that means either this fixture is
wrong or the solver has a bug the refactor plan did not anticipate --
either way it must be reported, not "fixed" by loosening a tolerance.
"""

import unittest

import numpy as np
from scipy.constants import elementary_charge as e

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.bin_average import triple_box_average_poles
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile

# LHC-like machine, matching
# `test_solvers_multiturn_boundary.py`'s `_make_simulation` fixture.
CIRCUMFERENCE = 26658.883
TRANSITION_GAMMA = 55.759505
HARMONIC = 35640
RF_VOLTAGE = 6e6  # [V]
SYNC_MOMENTUM = 450e9  # [eV/c]

# Narrow-band enough that the wake still lives a few bins after the charge,
# matching `TestMultiPoleSparseSolveTurnBoundaryWake` in that file -- these
# values are already known to exercise both near and far field. Its decay
# per bin, |pole * bin_dt| = pi * f_r / Q * bin_dt, is ~0.87 (with
# N_BINS_FULL_TURN = 32 below): the far-field recursion is down to ~1e-4 of
# peak by bin 5, so this resonator alone barely loads the recursion that
# the refactor rewrites -- see SLOW_DECAY_* below.
SHUNT_IMPEDANCE = 1e6  # [Ohm]
CENTER_FREQUENCY = 1e9  # [Hz]
QUALITY_FACTOR = 1e4

SIGMA_DT_NARROW = 2e-10  # [s], localises a bunch into a single bin.
N_MACROPARTICLES = 500
SEED = 42

N_BINS_FULL_TURN = 32
# First "island" of charge, in bin index, for the sparse (gap_bins > 0)
# fixture. Chosen with margin from bin 0 (near-field/far-field state setup)
# and from the profile end.
FIRST_ISLAND_BIN = 5

_T_REV = ConstantMagneticCycle(
    reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
).get_t_rev_init(CIRCUMFERENCE, particle_type=proton)
BIN_DT = _T_REV / N_BINS_FULL_TURN

# A second resonator whose far field is genuinely exercised: pole real
# part is -pi * f_r / Q, so |pole * BIN_DT| = _SLOW_DECAY_PER_BIN here.
# Chosen at the low end of the "order 0.1-1" range the review asked for,
# so consecutive far-field bins are down by only exp(-0.15) ~= 0.86 each
# -- ~47% of peak still present 5 bins after the charge, ~22% after 10 --
# unlike SHUNT_IMPEDANCE/CENTER_FREQUENCY/QUALITY_FACTOR above.
SLOW_DECAY_SHUNT_IMPEDANCE = 1e6  # [Ohm]
SLOW_DECAY_QUALITY_FACTOR = 100.0
_SLOW_DECAY_PER_BIN = 0.15
SLOW_DECAY_CENTER_FREQUENCY = (
    _SLOW_DECAY_PER_BIN * SLOW_DECAY_QUALITY_FACTOR / (np.pi * BIN_DT)
)  # [Hz]

RESONATOR_PARAMS = {
    "fast_decay": (SHUNT_IMPEDANCE, CENTER_FREQUENCY, QUALITY_FACTOR),
    "slow_decay": (
        SLOW_DECAY_SHUNT_IMPEDANCE,
        SLOW_DECAY_CENTER_FREQUENCY,
        SLOW_DECAY_QUALITY_FACTOR,
    ),
}

# Structural-gap fixture (Finding 1): two active buckets that are NOT
# adjacent, so `EquidistantMultiProfile` packs its memory with a real hole
# in the time axis where buckets 1 and 2 are skipped -- unlike the
# `StaticProfile`-based fixtures above, whose bin spacing is uniform
# everywhere regardless of how charge is placed. Copied in spirit from
# `tests/unittests/physics/impedances/sparse_profile/test_solvers.py`'s
# `_run_sparse` fixture.
SPARSE_N_SLOTS = 8
SPARSE_FILLING_PATTERN = np.zeros(SPARSE_N_SLOTS, dtype=bool)
SPARSE_FILLING_PATTERN[[0, 3]] = True
SPARSE_BINS_PER_PROFILE = 16


def reference_pole_voltage(
    profile_hist_x, profile_hist_y, poles, residues, bin_dt, factor
):
    """
    Induced voltage as a direct double sum over bins, in [V].

    Parameters
    ----------
    profile_hist_x
        The profile's time axis, in [s].
    profile_hist_y
        Beam profile histogram.
    poles
        Complex poles of the model, in [rad/s].
    residues
        Complex residues of the model.
    bin_dt
        Profile bin width, in [s].
    factor
        Conversion from histogram counts to charge per bin.

    Returns
    -------
    voltage
        The induced voltage at every bin, in [V].
    """
    times = copy_to_cpu(profile_hist_x)
    charges = copy_to_cpu(profile_hist_y) * factor
    voltage = np.zeros(len(times))
    for i, t_obs in enumerate(times):
        lags = t_obs - times
        wake = copy_to_cpu(
            triple_box_average_poles(
                backend.array(lags, dtype=backend.float),
                poles,
                residues,
                bin_dt,
            )
        )
        voltage[i] = float(np.sum(wake * charges))
    return voltage


def _first_bin_index_after_gap(hist_x, bin_dt, rtol=1e-6):
    """
    Index of the first bin that follows a gap in the profile's time axis.

    A "gap" is a pair of consecutive bins whose distance exceeds one bin
    width -- e.g. the charge-free hole `EquidistantMultiProfile` leaves
    between two bunches that are not in adjacent buckets.

    Parameters
    ----------
    hist_x
        The profile's time axis, in [s]. May be a NumPy or CuPy array.
    bin_dt
        Profile bin width, in [s].
    rtol
        Relative tolerance above ``bin_dt`` before a spacing counts as a
        gap, to absorb floating-point noise in the bin centres.

    Returns
    -------
    index
        Index of the first bin after a gap, or ``None`` if the time axis
        has no gap (uniform spacing throughout).
    """
    hist_x_cpu = copy_to_cpu(hist_x)
    spacings = hist_x_cpu[1:] - hist_x_cpu[:-1]
    is_gap = spacings > bin_dt * (1.0 + rtol)
    gap_positions = np.nonzero(is_gap)[0]
    if len(gap_positions) == 0:
        return None
    return int(gap_positions[0]) + 1


def _make_simulation(solver, sources, n_bins, cut_right):
    """Assemble an LHC-like ring with a `StaticProfile` of the given span.

    Adapted from `test_solvers_multiturn_boundary._make_simulation`: that
    fixture always spans the full turn (``cut_right = t_rev``); here
    ``cut_right`` is a parameter so the profile can be made shorter than a
    full turn, which is what produces an inter-call gap wider than one bin
    (see `_run_solver_and_reference`).
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    profile = StaticProfile(cut_left=0.0, cut_right=cut_right, n_bins=n_bins)
    wakefield = WakeField(sources=sources, solver=solver, profile=profile)
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=ring.circumference,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=RF_VOLTAGE, phi_rf=0.0
    )
    ring.add_elements((wakefield, drift, rf_station))
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, wakefield, drift, rf_station


def _prepare_and_place_beam(
    simulation, gap_bins, n_bins_profile, bin_dt=BIN_DT
):
    """Prepare a BiGaussian bunch and lay out the deterministic pattern."""
    BIN_DT_LOCAL = bin_dt
    bunch = Beam(intensity=1e11, particle_type=proton)
    simulation.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=SIGMA_DT_NARROW,
            n_macroparticles=N_MACROPARTICLES,
            seed=SEED,
        ),
        beam=bunch,
    )
    if gap_bins == 0:
        # Dense, gap-free: one narrow bunch centred in every bin.
        beam = make_multibunch_beam(
            beam=bunch,
            n_times=n_bins_profile,
            t_distance=BIN_DT_LOCAL,
            common_offset=0.5 * BIN_DT_LOCAL,
        )
    else:
        # Sparse: two narrow bunches, `gap_bins` empty bins apart.
        second_island_bin = FIRST_ISLAND_BIN + gap_bins + 1
        if second_island_bin >= n_bins_profile - 2:
            raise AssertionError(
                "fixture bug: the second island must leave margin before "
                f"the profile end ({second_island_bin=}, {n_bins_profile=})"
            )
        beam = make_multibunch_beam(
            beam=bunch,
            n_times=2,
            t_distance=(gap_bins + 1) * BIN_DT_LOCAL,
            common_offset=(FIRST_ISLAND_BIN + 0.5) * BIN_DT_LOCAL,
        )
    return beam


def _run_solver_and_reference(
    gap_bins,
    n_calls,
    call_gap_bins=1,
    resonator_params=(SHUNT_IMPEDANCE, CENTER_FREQUENCY, QUALITY_FACTOR),
    counter_rotating=False,
):
    """
    Run `MultiPoleSparseSolve` over `n_calls` turns and build its reference.

    The beam is frozen (``orbit_length = 0``, RF voltage off) so every turn
    presents the identical charge pattern; only the recursion's reference
    time advances, exactly as in
    `TestMultiPoleSparseSolveTurnBoundaryWake._run`. ``call_gap_bins``
    controls how many bin widths separate the last bin of one call from
    the first bin of the next: the profile spans
    ``N_BINS_FULL_TURN - (call_gap_bins - 1)`` bins of the full-turn grid,
    which is ``call_gap_bins - 1`` bins short of a full turn, so the gap
    the following call's first bin sees is ``call_gap_bins`` bin widths
    (``call_gap_bins = 1`` reproduces the full-turn, no-slack case in
    `test_solvers_multiturn_boundary.py`).

    Parameters
    ----------
    gap_bins
        Number of charge-free bins between the two islands of charge
        (0 selects the dense, gap-free pattern instead).
    n_calls
        Number of turns (solver calls) to run.
    call_gap_bins
        Bin widths between the last bin of one call and the first bin of
        the next.
    resonator_params
        ``(shunt_impedance, center_frequency, quality_factor)`` passed to
        `Resonators`. Defaults to the fast-decaying resonator; pass
        `RESONATOR_PARAMS["slow_decay"]` to load the far-field recursion.
    counter_rotating
        If true, the resonator gets a sign-flipped counter-rotating shunt
        impedance (so `get_vectorfit` reports a pole sign of -1) and the
        beam is marked counter-rotating. A beam's *own* wake is unchanged
        by that -- the solver applies the flip both to the charge it
        injects and to the amplitude it reads back, so the two cancel --
        which is why `expected` below is valid for either case, and what a
        charge carried across a call boundary with the wrong call's flip
        would break.

    Returns
    -------
    voltages
        List of length `n_calls`, each the solver's induced voltage for
        that call, as a NumPy array.
    expected
        List of length `n_calls`, each the independent reference voltage
        for that call, built by `reference_pole_voltage` accumulated over
        all calls so far (see module docstring and Step 3 of the plan for
        why a call's reference must not see later calls' charge).
    """
    n_bins_profile = N_BINS_FULL_TURN - (call_gap_bins - 1)
    cut_right = n_bins_profile * BIN_DT
    source = Resonators(
        *resonator_params,
        shunt_impedances_counter_rotating=(
            np.array([-resonator_params[0]]) if counter_rotating else None
        ),
    )
    simulation, wakefield, drift, rf_station = _make_simulation(
        solver=MultiPoleSparseSolve(),
        sources=(source,),
        n_bins=n_bins_profile,
        cut_right=cut_right,
    )
    beam = _prepare_and_place_beam(simulation, gap_bins, n_bins_profile)
    if counter_rotating:
        beam._is_counter_rotating = True

    # Freeze the beam: every turn must see the identical charge pattern.
    drift.orbit_length = 0.0
    rf_station.voltage = 0.0
    simulation.check_circumference = "ignore"

    recorded_voltages = []
    recorded_hist_y = []

    def _advance_reference_time_and_record(simulation, beam):
        # Record this turn's output before advancing time for the next
        # one -- callbacks run at the end of each turn, i.e. after this
        # turn's `calc_induced_voltage` but before the next.
        recorded_voltages.append(copy_to_cpu(wakefield.induced_voltage))
        recorded_hist_y.append(copy_to_cpu(wakefield.profile.hist_y))
        # `orbit_length = 0` freezes the beam but also stops
        # reference.time from advancing; this callback restores the
        # increment the pole solver needs to place the previous call's
        # state (see `TestMultiPoleSparseSolveTurnBoundaryWake._run`).
        beam.reference.time += CIRCUMFERENCE / beam.reference.velocity

    simulation.run_simulation(
        beams=beam,
        n_turns=n_calls,
        callbacks=_advance_reference_time_and_record,
    )

    self_hist_x = copy_to_cpu(wakefield.profile.hist_x)
    factor = -(1 * beam.particle_type.charge * e) * (
        beam.intensity * wakefield.profile.hist_y_to_density_factor
    )
    poles, residues, _ = source.get_vectorfit()

    expected = []
    accumulated_times = np.empty(0)
    accumulated_hist_y = np.empty(0)
    for call_index in range(n_calls):
        times_this_call = self_hist_x + call_index * _T_REV
        hist_y_this_call = recorded_hist_y[call_index]
        combined_times = np.concatenate((accumulated_times, times_this_call))
        combined_hist_y = np.concatenate(
            (accumulated_hist_y, hist_y_this_call)
        )
        full_reference = reference_pole_voltage(
            combined_times,
            combined_hist_y,
            poles,
            residues,
            BIN_DT,
            factor,
        )
        expected.append(full_reference[-len(times_this_call) :])
        accumulated_times = combined_times
        accumulated_hist_y = combined_hist_y

    return recorded_voltages, expected


def _run_fractional_call_gap_case(
    call_gap, n_calls=3, resonator_params=None, counter_rotating=False
):
    """Single-gap shorthand for `_run_varying_call_gap_case`.

    Parameters
    ----------
    call_gap
        Distance between one call's last bin and the next call's first, in
        bin widths. May be fractional.
    n_calls
        Number of turns (solver calls) to run.
    resonator_params
        ``(shunt_impedance, center_frequency, quality_factor)``; defaults
        to the slow-decaying resonator, which loads the recursion.
    counter_rotating
        If true, the resonator gets a sign-flipped counter-rotating shunt
        impedance and the beam is marked counter-rotating.

    Returns
    -------
    voltages
        List of length `n_calls`, the solver's induced voltage per call.
    expected
        List of length `n_calls`, the reference voltage per call.
    """
    return _run_varying_call_gap_case(
        (call_gap,),
        n_calls=n_calls,
        resonator_params=resonator_params,
        counter_rotating=counter_rotating,
    )


def _run_varying_call_gap_case(
    call_gaps, n_calls=3, resonator_params=None, counter_rotating=False
):
    """
    Run the solver with inter-call gaps cycling through `call_gaps`.

    Unlike `_run_solver_and_reference`, a gap need not be a whole number:
    the profile keeps `N_BINS_FULL_TURN` bins but their width is chosen so
    that the last bin of one call and the first of the next end up
    ``call_gaps[0]`` bin widths apart --

        t_rev = (n_bins - 1 + call_gaps[0]) * bin_dt

    -- which is what a profile covering slightly less than a full
    revolution period does. Every bin holds a bunch, the last one
    included, so the charge handed to the next call is not zero.

    With more than one gap the reference clock is advanced by a different
    amount before each call, so consecutive hand-overs differ. That is what
    a solver shared by two beams sees, and it is the case a constant gap
    cannot reach: anything the solver caches on the hand-over lag is only
    exercised when the lag changes.

    Parameters
    ----------
    call_gaps
        Distances between one call's last bin and the next call's first, in
        bin widths, cycled over the calls. May be fractional.
    n_calls
        Number of turns (solver calls) to run.
    resonator_params
        ``(shunt_impedance, center_frequency, quality_factor)``; defaults
        to the slow-decaying resonator, which loads the recursion.
    counter_rotating
        If true, the resonator gets a sign-flipped counter-rotating shunt
        impedance and the beam is marked counter-rotating; see
        `_run_solver_and_reference` for why the reference is unchanged by
        that.

    Returns
    -------
    voltages
        List of length `n_calls`, the solver's induced voltage per call.
    expected
        List of length `n_calls`, the reference voltage per call.
    """
    if resonator_params is None:
        resonator_params = RESONATOR_PARAMS["slow_decay"]
    n_bins_profile = N_BINS_FULL_TURN
    bin_dt = _T_REV / (n_bins_profile - 1 + call_gaps[0])
    source = Resonators(
        *resonator_params,
        shunt_impedances_counter_rotating=(
            np.array([-resonator_params[0]]) if counter_rotating else None
        ),
    )
    simulation, wakefield, drift, rf_station = _make_simulation(
        solver=MultiPoleSparseSolve(),
        sources=(source,),
        n_bins=n_bins_profile,
        cut_right=n_bins_profile * bin_dt,
    )
    beam = _prepare_and_place_beam(
        simulation, gap_bins=0, n_bins_profile=n_bins_profile, bin_dt=bin_dt
    )
    if counter_rotating:
        beam._is_counter_rotating = True

    drift.orbit_length = 0.0
    rf_station.voltage = 0.0
    simulation.check_circumference = "ignore"

    recorded_voltages = []
    recorded_hist_y = []
    call_times = []
    # The profile spans `(n_bins - 1) * bin_dt` from first bin centre to
    # last, so a hand-over of `gap` bin widths means advancing the clock by
    # that plus `gap * bin_dt`.
    span_dt = (n_bins_profile - 1) * bin_dt

    def _advance_reference_time_and_record(simulation, beam):
        recorded_voltages.append(copy_to_cpu(wakefield.induced_voltage))
        recorded_hist_y.append(copy_to_cpu(wakefield.profile.hist_y))
        call_times.append(float(beam.reference.time))
        gap = call_gaps[len(call_times) % len(call_gaps)]
        beam.reference.time += span_dt + gap * bin_dt

    simulation.run_simulation(
        beams=beam,
        n_turns=n_calls,
        callbacks=_advance_reference_time_and_record,
    )

    self_hist_x = copy_to_cpu(wakefield.profile.hist_x)
    factor = -(1 * beam.particle_type.charge * e) * (
        beam.intensity * wakefield.profile.hist_y_to_density_factor
    )
    poles, residues, _ = source.get_vectorfit()

    expected = []
    accumulated_times = np.empty(0)
    accumulated_hist_y = np.empty(0)
    for call_index in range(n_calls):
        times_this_call = self_hist_x + (
            call_times[call_index] - call_times[0]
        )
        combined_times = np.concatenate((accumulated_times, times_this_call))
        combined_hist_y = np.concatenate(
            (accumulated_hist_y, recorded_hist_y[call_index])
        )
        full_reference = reference_pole_voltage(
            combined_times, combined_hist_y, poles, residues, bin_dt, factor
        )
        expected.append(full_reference[-len(times_this_call) :])
        accumulated_times = combined_times
        accumulated_hist_y = combined_hist_y

    return recorded_voltages, expected


class TestDenseProfileAgainstConvolution(unittest.TestCase):
    """`MultiPoleSparseSolve` on a gap-free profile equals the direct sum."""

    def test_single_call_matches_reference(self) -> None:
        for resonator_name, resonator_params in RESONATOR_PARAMS.items():
            with self.subTest(resonator=resonator_name):
                voltages, expected = _run_solver_and_reference(
                    gap_bins=0,
                    n_calls=1,
                    resonator_params=resonator_params,
                )
                voltage = voltages[0]
                reference = expected[0]
                np.testing.assert_allclose(
                    voltage,
                    reference,
                    rtol=1e-9,
                    atol=1e-9 * float(np.max(np.abs(reference))),
                    err_msg=(
                        "the near-field taps plus the far-field recursion "
                        "must sum to the direct bin-averaged convolution"
                    ),
                )


class TestSparseAndMultiCallAgainstConvolution(unittest.TestCase):
    """`MultiPoleSparseSolve` matches the direct sum with gaps and calls.

    Covers charge-free bins within a `StaticProfile` (``gap_bins`` -- a
    uniformly-spaced axis with some zero-charge bins, *not* a structural
    hole in the time axis; see `TestStructuralGapAgainstConvolution` for
    that) and gaps between successive calls (``call_gap_bins``), including
    calls that hand over state to each other (``n_calls > 1``).
    """

    def test_gap_and_call_combinations(self) -> None:
        for resonator_name, resonator_params in RESONATOR_PARAMS.items():
            for gap_bins in (0, 1, 5):
                with self.subTest(
                    resonator=resonator_name, gap_bins=gap_bins, n_calls=1
                ):
                    voltages, expected = _run_solver_and_reference(
                        gap_bins=gap_bins,
                        n_calls=1,
                        resonator_params=resonator_params,
                    )
                    scale = float(np.max(np.abs(expected[0])))
                    np.testing.assert_allclose(
                        voltages[0],
                        expected[0],
                        rtol=1e-9,
                        atol=1e-9 * scale,
                    )

                for call_gap_bins in (1, 2, 3):
                    with self.subTest(
                        resonator=resonator_name,
                        gap_bins=gap_bins,
                        n_calls=3,
                        call_gap_bins=call_gap_bins,
                    ):
                        voltages, expected = _run_solver_and_reference(
                            gap_bins=gap_bins,
                            n_calls=3,
                            call_gap_bins=call_gap_bins,
                            resonator_params=resonator_params,
                        )
                        for call_index in range(3):
                            scale = float(np.max(np.abs(expected[call_index])))
                            # A call with no charge in it (e.g. the
                            # trailing island falls outside a very short
                            # profile) has scale == 0; guard against a
                            # zero atol turning the comparison bit-exact.
                            atol = 1e-9 * scale if scale > 0.0 else 1e-12
                            np.testing.assert_allclose(
                                voltages[call_index],
                                expected[call_index],
                                rtol=1e-9,
                                atol=atol,
                                err_msg=(
                                    f"call {call_index} of "
                                    f"resonator={resonator_name} "
                                    f"{gap_bins=} {call_gap_bins=}"
                                ),
                            )


def _make_sparse_profile_simulation(solver, sources):
    """Assemble the LHC-like ring with a structurally sparse profile.

    Copied in spirit from
    `test_solvers.py._run_sparse` (`tests/unittests/physics/impedances/
    sparse_profile/test_solvers.py`): buckets 0 and 3 of `SPARSE_N_SLOTS`
    are filled, buckets 1 and 2 are skipped, leaving a real hole in the
    packed memory's time axis between them.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    )
    profile = EquidistantMultiProfile(
        filling_pattern=SPARSE_FILLING_PATTERN,
        bins_per_profile=SPARSE_BINS_PER_PROFILE,
        offset=0.0,
    )
    wakefield = WakeField(sources=sources, solver=solver, profile=profile)
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=ring.circumference,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=RF_VOLTAGE, phi_rf=0.0
    )
    ring.add_elements((wakefield, drift, rf_station))
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, wakefield, drift, rf_station


def _sparse_profile_bin_dt():
    """Bin width of the structural-gap fixture's profile, in [s]."""
    return _T_REV / SPARSE_N_SLOTS / SPARSE_BINS_PER_PROFILE


def _prepare_sparse_gap_beam(simulation):
    """
    Two narrow bunches straddling the structural gap.

    One bunch sits in the very last bin of bucket 0, the other in the
    very first bin of bucket 3, so the near-field lag terms either side
    of the gap actually see charge -- not just the far-field recursion
    carrying state across it.
    """
    profile_width = _T_REV / SPARSE_N_SLOTS
    sparse_bin_dt = _sparse_profile_bin_dt()
    starts = np.linspace(0.0, _T_REV, SPARSE_N_SLOTS, endpoint=False)
    last_bin_of_bucket0 = starts[0] + profile_width - 0.5 * sparse_bin_dt
    first_bin_of_bucket3 = starts[3] + 0.5 * sparse_bin_dt

    bunch = Beam(intensity=1e11, particle_type=proton)
    simulation.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=SIGMA_DT_NARROW,
            n_macroparticles=N_MACROPARTICLES,
            seed=SEED,
        ),
        beam=bunch,
    )
    beam = make_multibunch_beam(
        beam=bunch,
        n_times=2,
        t_distance=first_bin_of_bucket3 - last_bin_of_bucket0,
        common_offset=last_bin_of_bucket0,
    )
    return beam


def _run_structural_gap_case(n_calls, resonator_params):
    """
    Run `MultiPoleSparseSolve` on the structural-gap fixture and its
    reference, following the same sequential-per-call accumulation as
    `_run_solver_and_reference` (see its docstring for why).

    Returns
    -------
    hist_x
        The profile's time axis, in [s].
    bin_dt
        The profile's bin width, in [s].
    voltages
        List of length `n_calls`, the solver's induced voltage per call.
    expected
        List of length `n_calls`, the reference voltage per call.
    """
    source = Resonators(*resonator_params)
    simulation, wakefield, drift, rf_station = _make_sparse_profile_simulation(
        solver=MultiPoleSparseSolve(), sources=(source,)
    )
    beam = _prepare_sparse_gap_beam(simulation)

    drift.orbit_length = 0.0
    rf_station.voltage = 0.0
    simulation.check_circumference = "ignore"

    recorded_voltages = []
    recorded_hist_y = []

    def _advance_reference_time_and_record(simulation, beam):
        recorded_voltages.append(copy_to_cpu(wakefield.induced_voltage))
        recorded_hist_y.append(copy_to_cpu(wakefield.profile.hist_y))
        beam.reference.time += CIRCUMFERENCE / beam.reference.velocity

    simulation.run_simulation(
        beams=beam,
        n_turns=n_calls,
        callbacks=_advance_reference_time_and_record,
    )

    hist_x = copy_to_cpu(wakefield.profile.hist_x)
    bin_dt = _sparse_profile_bin_dt()
    factor = -(1 * beam.particle_type.charge * e) * (
        beam.intensity * wakefield.profile.hist_y_to_density_factor
    )
    poles, residues, _ = source.get_vectorfit()

    expected = []
    accumulated_times = np.empty(0)
    accumulated_hist_y = np.empty(0)
    for call_index in range(n_calls):
        times_this_call = hist_x + call_index * _T_REV
        hist_y_this_call = recorded_hist_y[call_index]
        combined_times = np.concatenate((accumulated_times, times_this_call))
        combined_hist_y = np.concatenate(
            (accumulated_hist_y, hist_y_this_call)
        )
        full_reference = reference_pole_voltage(
            combined_times,
            combined_hist_y,
            poles,
            residues,
            bin_dt,
            factor,
        )
        expected.append(full_reference[-len(times_this_call) :])
        accumulated_times = combined_times
        accumulated_hist_y = combined_hist_y

    return hist_x, bin_dt, recorded_voltages, expected


class TestStructuralGapAgainstConvolution(unittest.TestCase):
    """`MultiPoleSparseSolve` on a profile with a real hole in its axis.

    `EquidistantMultiProfile` packs only filled buckets into memory, so
    two non-adjacent filled buckets leave consecutive stored bins more
    than one bin width apart -- the case no `StaticProfile`-based
    fixture in this file can reach (its bin spacing is uniform everywhere
    no matter how charge is placed).
    """

    def test_fixture_has_a_real_gap(self) -> None:
        simulation, wakefield, _, _ = _make_sparse_profile_simulation(
            solver=MultiPoleSparseSolve(),
            sources=(Resonators(*RESONATOR_PARAMS["slow_decay"]),),
        )
        hist_x = copy_to_cpu(wakefield.profile.hist_x)
        gap_index = _first_bin_index_after_gap(
            hist_x, _sparse_profile_bin_dt()
        )
        self.assertIsNotNone(
            gap_index,
            "fixture bug: expected a structural gap between buckets 0 "
            "and 3 of the packed profile memory",
        )

    def test_single_call_matches_reference(self) -> None:
        for resonator_name, resonator_params in RESONATOR_PARAMS.items():
            with self.subTest(resonator=resonator_name):
                _, _, voltages, expected = _run_structural_gap_case(
                    n_calls=1, resonator_params=resonator_params
                )
                scale = float(np.max(np.abs(expected[0])))
                np.testing.assert_allclose(
                    voltages[0],
                    expected[0],
                    rtol=1e-9,
                    atol=1e-9 * scale,
                    err_msg=f"resonator={resonator_name}",
                )

    def test_multi_call_matches_reference(self) -> None:
        for resonator_name, resonator_params in RESONATOR_PARAMS.items():
            with self.subTest(resonator=resonator_name):
                _, _, voltages, expected = _run_structural_gap_case(
                    n_calls=3, resonator_params=resonator_params
                )
                for call_index in range(3):
                    scale = float(np.max(np.abs(expected[call_index])))
                    atol = 1e-9 * scale if scale > 0.0 else 1e-12
                    np.testing.assert_allclose(
                        voltages[call_index],
                        expected[call_index],
                        rtol=1e-9,
                        atol=atol,
                        err_msg=(
                            f"call {call_index} of resonator={resonator_name}"
                        ),
                    )


class TestGapHandoverIsNotDoubleCounted(unittest.TestCase):
    """A bin reached across a gap is claimed by exactly one of two paths."""

    def test_first_bin_after_a_gap_matches_reference(self) -> None:
        hist_x, bin_dt, voltages, expected = _run_structural_gap_case(
            n_calls=1, resonator_params=RESONATOR_PARAMS["slow_decay"]
        )
        first_after_gap = _first_bin_index_after_gap(hist_x, bin_dt)
        self.assertIsNotNone(
            first_after_gap,
            "fixture bug: expected a structural gap in the time axis",
        )
        self.assertNotAlmostEqual(
            expected[0][first_after_gap],
            0.0,
            msg="fixture bug: nothing to compare at the bin after the gap",
        )
        self.assertAlmostEqual(
            voltages[0][first_after_gap] / expected[0][first_after_gap],
            1.0,
            places=8,
            msg=(
                "the bin before a gap is at a lag of more than two bins, so "
                "the far-field recursion covers it; the near-field previous"
                "-bin tap must be zero there or its charge is counted twice"
            ),
        )


class TestCounterRotatingChargeAcrossACallBoundary(unittest.TestCase):
    """A carried charge keeps the flip of the call that produced it."""

    def test_counter_rotating_beam_matches_reference(self) -> None:
        # `gap_bins=0` fills every bin, the last one included, so the
        # charge handed to the next call is not zero -- with an empty
        # trailing bin the flip would have nothing to be wrong about.
        # `call_gap_bins` 1 and 2 cover both hand-over routes: the
        # trailing bin injected straight into the state (2), and the one
        # that is not yet due and rides the kernel's delayed charge (1).
        for call_gap_bins in (1, 2):
            with self.subTest(call_gap_bins=call_gap_bins):
                self._assert_matches_reference(call_gap_bins)

    def _assert_matches_reference(self, call_gap_bins: int) -> None:
        voltages, expected = _run_solver_and_reference(
            gap_bins=0,
            n_calls=3,
            call_gap_bins=call_gap_bins,
            resonator_params=RESONATOR_PARAMS["slow_decay"],
            counter_rotating=True,
        )
        for call_index in range(3):
            scale = float(np.max(np.abs(expected[call_index])))
            self.assertGreater(scale, 0.0, "fixture bug: no wake at all")
            np.testing.assert_allclose(
                voltages[call_index],
                expected[call_index],
                rtol=1e-9,
                atol=1e-9 * scale,
                err_msg=(
                    f"call {call_index}: a counter-rotating beam's own wake "
                    "must equal the co-rotating one -- the flip is applied "
                    "to both the injected charge and the amplitude read "
                    "back, so it cancels. A charge carried across a call "
                    "boundary must therefore be signed with the flip of "
                    "the call that produced it, not of the call reading it."
                ),
            )


def _run_two_beam_shared_solver(call_gaps, n_calls=6, beam_order=(0, 1)):
    """
    Drive one solver with two beams that rotate opposite ways.

    The muon-collider case the carry logic is written for: a single
    `MultiPoleSparseSolve` sees calls from a co-rotating and a
    counter-rotating beam in turn, each at its own reference time, so
    consecutive hand-overs differ in length *and* in which beam produced
    the charge the next call inherits.

    The beams are driven through `calc_induced_voltage` directly rather
    than through a mainloop: what is under test is the solver's behaviour
    across a sequence of calls, not the execution model that orders them.
    One turn is tracked first, to fill the profile and wire the wakefield
    up; that call is part of the sequence the reference accounts for, but
    its own voltage is not compared.

    Parameters
    ----------
    call_gaps
        Distances between one call's last bin and the next call's first,
        in bin widths, cycled over the calls.
    n_calls
        Number of driven calls.
    beam_order
        Which beam takes each call, cycled: ``(0, 1)`` alternates,
        ``(0, 0, 1, 1)`` gives each beam two turns in a row. The order
        decides how often the counter-rotation sign pattern of a hand-over
        changes, which a strictly alternating order never does.

    Returns
    -------
    voltages
        List of length `n_calls`, the solver's induced voltage per call.
    expected
        List of length `n_calls`, the reference voltage per call.
    """
    n_bins_profile = N_BINS_FULL_TURN
    bin_dt = _T_REV / (n_bins_profile - 1 + call_gaps[0])
    shunt_impedance = RESONATOR_PARAMS["slow_decay"][0]
    source = Resonators(
        *RESONATOR_PARAMS["slow_decay"],
        # Sign-flipped for a counter-rotating beam, so that the two beams
        # see each other's charge with the opposite sign.
        shunt_impedances_counter_rotating=np.array([-shunt_impedance]),
    )
    simulation, wakefield, drift, rf_station = _make_simulation(
        solver=MultiPoleSparseSolve(),
        sources=(source,),
        n_bins=n_bins_profile,
        cut_right=n_bins_profile * bin_dt,
    )
    beams = [
        _prepare_and_place_beam(
            simulation,
            gap_bins=0,
            n_bins_profile=n_bins_profile,
            bin_dt=bin_dt,
        )
        for _ in range(2)
    ]
    beams[1]._is_counter_rotating = True

    drift.orbit_length = 0.0
    rf_station.voltage = 0.0
    simulation.check_circumference = "ignore"
    simulation.run_simulation(beams=beams[0], n_turns=1)

    solver = wakefield.solver
    profile = wakefield.profile
    hist_y = copy_to_cpu(profile.hist_y)
    self_hist_x = copy_to_cpu(profile.hist_x)
    span_dt = (n_bins_profile - 1) * bin_dt
    poles, residues, pole_signs = source.get_vectorfit()
    counter_rotating_sign = float(copy_to_cpu(pole_signs)[0])

    # The tracked turn is the sequence's first call, by the co-rotating
    # beam, at whatever time it happened.
    call_times = [float(beams[0].reference.time)]
    call_is_counter_rotating = [False]
    voltages = []
    reference_time = call_times[0]
    for call_index in range(n_calls):
        beam = beams[beam_order[call_index % len(beam_order)]]
        reference_time += (
            span_dt + call_gaps[call_index % len(call_gaps)] * bin_dt
        )
        beam.reference.time = reference_time
        voltages.append(copy_to_cpu(solver.calc_induced_voltage(beam)))
        call_times.append(reference_time)
        call_is_counter_rotating.append(beam.is_counter_rotating)

    factor = -(1 * proton.charge * e) * (
        beams[0].intensity * profile.hist_y_to_density_factor
    )
    # A pole whose counter-rotating sign is -1 contributes with the product
    # of the two beams' signs: +1 where the charge and the bin observing it
    # belong to beams rotating the same way, the pole's sign where they do
    # not. With one resonator that is a single sign per pair of calls, so
    # it folds into the charge.
    expected = []
    for call_index in range(1, len(call_times)):
        times_this_call = self_hist_x + call_times[call_index]
        voltage = np.zeros(len(times_this_call))
        for source_index in range(call_index + 1):
            sign = (
                1.0
                if call_is_counter_rotating[source_index]
                == call_is_counter_rotating[call_index]
                else counter_rotating_sign
            )
            source_times = self_hist_x + call_times[source_index]
            for bin_index, t_obs in enumerate(times_this_call):
                wake = copy_to_cpu(
                    triple_box_average_poles(
                        backend.array(
                            t_obs - source_times, dtype=backend.float
                        ),
                        poles,
                        residues,
                        bin_dt,
                    )
                )
                voltage[bin_index] += sign * float(
                    np.sum(wake * hist_y * factor)
                )
        expected.append(voltage)

    return voltages, expected


class TestFractionalCallGap(unittest.TestCase):
    """The gap between two calls is a time, not a whole number of bins.

    A profile that covers slightly less than a full revolution period
    hands over on a fraction of a bin. Between one and two bin widths the
    carried charge comes due *between* two of the recursion's clocks,
    which the recursion cannot express on its own -- the solver has to
    inject it from Python and pay the first bin as a near lag.
    """

    def test_gaps_match_reference(self) -> None:
        for call_gap in (1.0, 1.05, 1.5, 1.9999, 2.0, 3.0):
            with self.subTest(call_gap=call_gap):
                self._assert_matches_reference(call_gap)

    def _assert_matches_reference(
        self, call_gap: float, counter_rotating: bool = False
    ) -> None:
        voltages, expected = _run_fractional_call_gap_case(
            call_gap, counter_rotating=counter_rotating
        )
        self._assert_all_calls_match(
            voltages,
            expected,
            f"a hand-over of {call_gap} bin widths: the carry must be "
            "advanced by the true elapsed time and injected at the true "
            "instant, and the first bin's near tap evaluated at the true "
            "lag",
        )

    def _assert_all_calls_match(self, voltages, expected, message) -> None:
        """
        Compare every call's voltage with its reference.

        Parameters
        ----------
        voltages
            The solver's induced voltage per call.
        expected
            The reference voltage per call.
        message
            Added to the failure message.
        """
        for call_index in range(len(voltages)):
            scale = float(np.max(np.abs(expected[call_index])))
            self.assertGreater(scale, 0.0, "fixture bug: no wake at all")
            np.testing.assert_allclose(
                voltages[call_index],
                expected[call_index],
                rtol=1e-9,
                atol=1e-9 * scale,
                err_msg=f"call {call_index} of {message}",
            )

    def test_alternating_gaps_match_reference(self) -> None:
        # Every other fixture in this file hands over the same distance
        # every turn, which is exactly what a lag-keyed cache cannot be
        # caught by. These alternate, so consecutive hand-overs land on
        # different sides of the causal onset and of the whole-bin case.
        for call_gaps in (
            (1.2, 1.7),
            (1.0, 1.7),
            (1.2, 1.9999),
            (1.05, 1.5),
            (1.5, 2.5),
            (1.0, 3.0),
        ):
            with self.subTest(call_gaps=call_gaps):
                voltages, expected = _run_varying_call_gap_case(
                    call_gaps, n_calls=6
                )
                self._assert_all_calls_match(
                    voltages,
                    expected,
                    f"{call_gaps=}: nothing may be cached on the hand-over "
                    "lag without the lag itself being part of the key",
                )

    def test_counter_rotating_beam_matches_reference(self) -> None:
        # The carried charge keeps the flip of the call that produced it,
        # in the kernel and in the closed-form tap on the first bin alike.
        # A beam's own wake is unchanged by counter-rotation -- the two
        # cancel -- so the same reference holds. 1.05 and 1.5 bin widths
        # land on either side of the near-field onset.
        for call_gap in (1.05, 1.5):
            with self.subTest(call_gap=call_gap):
                self._assert_matches_reference(call_gap, counter_rotating=True)


class TestTwoBeamsSharingOneSolver(unittest.TestCase):
    """One solver, a co-rotating and a counter-rotating beam, in turn.

    The case `MultiPoleSparseSolve`'s carry is written for, and the only
    one in which the counter-rotation flip of the call that *produced* a
    carried charge can be told apart from the flip of the call reading it:
    with a single beam the two are the same number. Both flips the solver
    applies by hand -- the sign it injects a carried charge with, and the
    amplitude it reads back for the first bin -- are pinned here against
    the convolution reference, which knows that a pole contributes with the
    product of the two beams' signs.

    The gaps alternate as well, because two beams meeting a cavity do not
    meet it at the same spacing.
    """

    def test_alternating_beams_match_reference(self) -> None:
        for call_gaps in ((1.5, 1.2), (1.0, 1.7), (2.5, 1.05)):
            with self.subTest(call_gaps=call_gaps):
                self._assert_matches_reference(call_gaps)

    def test_two_sub_onset_gaps_match_reference(self) -> None:
        # Both hand-overs below the causal onset and different from each
        # other, so both take the trailing tap's memo branch under the
        # *same* sign pattern -- the one case where a memo of one lag per
        # pattern would evict itself every turn.
        self._assert_matches_reference((1.05, 1.2))

    def test_beams_taking_two_turns_each_match_reference(self) -> None:
        # A, A, B, B rather than A, B, A, B: the sign pattern of the
        # hand-over now changes every other call while the lag stays put.
        # A strictly alternating order never exercises that, and the
        # whole-bin hand-over below is the commonest geometry there is.
        for call_gaps in ((1.0,), (1.2,), (1.5, 1.05)):
            with self.subTest(call_gaps=call_gaps):
                self._assert_matches_reference(
                    call_gaps, beam_order=(0, 0, 1, 1)
                )

    def _assert_matches_reference(self, call_gaps, beam_order=(0, 1)) -> None:
        """
        Compare every call of a two-beam run with its reference.

        Parameters
        ----------
        call_gaps
            Hand-over distances, in bin widths, cycled over the calls.
        beam_order
            Which beam takes each call, cycled.
        """
        voltages, expected = _run_two_beam_shared_solver(
            call_gaps, n_calls=8, beam_order=beam_order
        )
        for call_index in range(len(voltages)):
            scale = float(np.max(np.abs(expected[call_index])))
            self.assertGreater(scale, 0.0, "fixture bug: no wake at all")
            np.testing.assert_allclose(
                voltages[call_index],
                expected[call_index],
                rtol=1e-9,
                atol=1e-9 * scale,
                err_msg=(
                    f"call {call_index} with {call_gaps=}, {beam_order=}: a "
                    "charge one beam leaves behind is seen by the other "
                    "with the product of their counter-rotation signs, in "
                    "the recursion and in the near tap that reaches across "
                    "the hand-over alike"
                ),
            )


if __name__ == "__main__":
    unittest.main()
