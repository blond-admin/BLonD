"""Tests for the SequentialMultiBunchMatcher (and matcher clone/extra_voltage)."""

import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.experimental.beam_preparation.analytic_distributions import (
    line_density,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    AnalyticDistributionMatcher,
    LineDensityMatcher,
)
from blond.experimental.beam_preparation.analytic_multibunch import (
    SelfConsistentMultiBunchMatcher,
    SequentialMultiBunchMatcher,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import (
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)

RF_PERIOD = 2.0 * np.pi / 2518229887.224505


def _build_simulation(
    resonator_r_shunt=None,
    intensity=3e11,
    n_buckets=16,
    resonator_frequency=8e8,
    resonator_quality=1.0,
    solver=None,
):
    ring = Ring(26658.883)
    rf_station = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    elements = [rf_station, drift]
    if resonator_r_shunt is not None:
        profile = StaticProfile(
            cut_left=0.0, cut_right=n_buckets * RF_PERIOD, n_bins=512
        )
        wakefield = WakeField(
            sources=(
                Resonators(
                    resonator_r_shunt,
                    resonator_frequency,
                    resonator_quality,
                ),
            ),
            solver=solver if solver is not None else TimeDomainFftSolver(),
            profile=profile,
        )
        elements += [wakefield, profile]
    ring.add_elements(elements, reorder=True)
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=intensity, particle_type=proton)
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, beam


def _template(**overrides):
    matcher = AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=1.2e-9,
        seed=0,
        n_points_grid=300,
        allow_inner_buckets=True,
    )
    return matcher.clone(**overrides) if overrides else matcher


def _bunch_positions_and_lengths(dt, bucket_indices):
    positions, lengths = [], []
    for bucket_index in bucket_indices:
        selection = (dt > bucket_index * RF_PERIOD) & (
            dt < (bucket_index + 1) * RF_PERIOD
        )
        positions.append(float(np.mean(dt[selection])))
        lengths.append(float(4.0 * np.std(dt[selection])))
    return np.array(positions), np.array(lengths)


# ------------------------------- clone ------------------------------------


def test_clone_overrides_and_independence():
    template = _template()
    varied = template.clone(bunch_length=1.0e-9, seed=7)
    assert varied is not template
    assert varied._bunch_length == 1.0e-9
    assert varied._seed == 7
    # Untouched arguments are inherited; the original is unmodified.
    assert varied._distribution_type == "parabolic_amplitude"
    assert template._bunch_length == 1.2e-9
    assert template._seed == 0


def test_clone_works_for_line_density_matcher():
    time_measured = np.linspace(-1e-9, 1e-9, 101)
    profile = line_density(
        time_measured, "binomial", 1.6e-9, bunch_position=0.0, exponent=1.5
    )
    matcher = LineDensityMatcher(
        n_macroparticles=1_000,
        time_array=time_measured,
        line_density_values=profile,
        seed=0,
    )
    varied = matcher.clone(half_option="both", seed=3)
    assert varied._half_option == "both"
    assert varied._seed == 3
    np.testing.assert_array_equal(varied._input_time, time_measured)


def test_clone_rejects_unknown_argument():
    with pytest.raises(TypeError, match="not_a_parameter"):
        _template().clone(not_a_parameter=1.0)


# ---------------------------- extra_voltage --------------------------------


def test_extra_voltage_shifts_synchronous_position():
    # A small constant extra voltage V0 moves the zero crossing of the
    # total voltage: sin(omega t) V + V0 = 0 -> dt = -asin(V0/V)/omega.
    extra_time = np.linspace(-2.0 * RF_PERIOD, 3.0 * RF_PERIOD, 100)
    v_0, v_rf = 2e5, 6e6
    omega_rf = 2.0 * np.pi / RF_PERIOD

    positions = {}
    for label, extra in (
        ("bare", None),
        ("offset", (extra_time, v_0 * np.ones_like(extra_time))),
    ):
        simulation, beam = _build_simulation()
        matcher = _template(extra_voltage=extra)
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        positions[label] = float(np.mean(copy_to_cpu(beam.read_partial_dt())))

    expected_shift = -np.arcsin(v_0 / v_rf) / omega_rf
    measured_shift = positions["offset"] - positions["bare"]
    assert np.isclose(measured_shift, expected_shift, rtol=0.05)


def test_extra_voltage_validation():
    with pytest.raises(ValueError, match="pair"):
        _template(extra_voltage=(np.zeros(4),))
    with pytest.raises(AssertionError, match="increasing"):
        _template(extra_voltage=(np.array([1.0, 0.0]), np.array([0.0, 0.0])))


# ------------------------ SequentialMultiBunchMatcher ----------------------


def test_train_positions_lengths_and_independent_noise():
    simulation, beam = _build_simulation()
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=_template(),
        n_bunches=3,
        bunch_spacing_buckets=5,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)

    np.testing.assert_array_equal(matcher.bucket_indices, [0, 5, 10])
    dt = copy_to_cpu(beam.read_partial_dt())
    assert len(dt) == 3 * 2_000
    positions, lengths = _bunch_positions_and_lengths(
        dt, matcher.bucket_indices
    )
    np.testing.assert_allclose(
        positions,
        (matcher.bucket_indices + 0.5) * RF_PERIOD,
        atol=0.02e-9,
    )
    np.testing.assert_allclose(lengths, 1.2e-9, rtol=3e-2)
    # Template mode derives per-bunch seeds: independent noise, so the
    # local coordinates must differ bunch to bunch.
    local_first = dt[:2_000]
    local_second = dt[2_000:4_000] - 5 * RF_PERIOD
    assert not np.allclose(local_first, local_second, atol=1e-13)
    assert [m._seed for m in matcher.bunch_matchers] == [0, 1, 2]


def test_per_bunch_parameters_and_mixed_types():
    time_measured = np.linspace(-1e-9, 1e-9, 101)
    profile = line_density(
        time_measured, "binomial", 1.6e-9, bunch_position=0.0, exponent=1.5
    )
    bunch_matchers = [
        _template(seed=1),
        _template(bunch_length=1.0e-9, seed=2),
        LineDensityMatcher(
            n_macroparticles=2_000,
            time_array=time_measured,
            line_density_values=profile,
            half_option="both",
            n_points_abel=2_000,
            seed=3,
            n_points_grid=300,
        ),
    ]
    simulation, beam = _build_simulation()
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=bunch_matchers,
        bucket_indices=[0, 4, 9],
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)

    dt = copy_to_cpu(beam.read_partial_dt())
    _, lengths = _bunch_positions_and_lengths(dt, matcher.bucket_indices)
    assert np.isclose(lengths[0], 1.2e-9, rtol=3e-2)
    assert np.isclose(lengths[1], 1.0e-9, rtol=3e-2)
    assert np.isclose(
        lengths[2], matcher.bunch_matchers[2].matched_bunch_length, rtol=3e-2
    )
    # The user's spec instances were deep-copied, not run.
    assert bunch_matchers[0].matched_bunch_length is None
    assert matcher.bunch_matchers[0].matched_bunch_length is not None


def test_wake_of_predecessor_shifts_next_bunch():
    # A long-memory resonator (decay over several buckets) so the
    # predecessor's wake reaches the next bucket. Reference: a single
    # bunch alone. In the two-bunch train, the first bunch (no
    # predecessor) must reproduce the reference exactly, while the
    # second must sit at a measurably different position.
    def run(bucket_indices):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e5,
            intensity=2e11 * len(bucket_indices),
            resonator_frequency=2e8,
            resonator_quality=10.0,
        )
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=_template(relaxation_factor=0.5),
            bucket_indices=bucket_indices,
            bunch_intensities=2e11,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        dt = copy_to_cpu(beam.read_partial_dt())
        positions, _ = _bunch_positions_and_lengths(dt, matcher.bucket_indices)
        shifts = (matcher.bucket_indices + 0.5) * RF_PERIOD - positions
        return shifts, matcher

    shift_single, _ = run([0])
    shifts_train, matcher = run([0, 1])

    # Self-wake shift is real and reproduced for the first bunch.
    assert abs(shift_single[0]) > 0.005e-9
    assert np.isclose(shifts_train[0], shift_single[0], atol=0.002e-9)
    # The predecessor's wake moves the second bunch measurably.
    assert abs(shifts_train[1] - shift_single[0]) > 0.01e-9
    # Each bunch ran its own converged self-wake iteration.
    for bunch_matcher in matcher.bunch_matchers:
        assert bunch_matcher.n_intensity_iterations >= 1
        assert bunch_matcher.final_potential_well_error < 1e-6


def test_intensity_handling():
    # None: equal split of beam.intensity, no warning.
    simulation, beam = _build_simulation(intensity=3e11)
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=_template(),
        n_bunches=3,
        bunch_spacing_buckets=2,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    np.testing.assert_allclose(matcher.bunch_intensities, 1e11)

    # Mismatching per-bunch sum: warn and overwrite (BLonD 2).
    simulation, beam = _build_simulation(intensity=3e11)
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=_template(),
        n_bunches=2,
        bunch_spacing_buckets=2,
        bunch_intensities=[1e11, 2.5e11],
    )
    with pytest.warns(UserWarning, match="overwritten"):
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    assert beam.intensity == 3.5e11


def test_input_validation():
    template = _template()
    with pytest.raises(ValueError, match="exactly one"):
        SequentialMultiBunchMatcher(bunch_matchers=template)
    with pytest.raises(ValueError, match="exactly one"):
        SequentialMultiBunchMatcher(
            bunch_matchers=template, bucket_indices=[0, 5], n_bunches=2
        )
    with pytest.raises(ValueError, match="bunch_spacing_buckets"):
        SequentialMultiBunchMatcher(bunch_matchers=template, n_bunches=2)
    with pytest.raises(ValueError, match="increasing"):
        SequentialMultiBunchMatcher(
            bunch_matchers=template, bucket_indices=[5, 0]
        )
    with pytest.raises(ValueError, match="bunch matchers"):
        SequentialMultiBunchMatcher(
            bunch_matchers=[template], bucket_indices=[0, 5]
        )
    with pytest.raises(TypeError, match="single-bunch matcher"):
        SequentialMultiBunchMatcher(
            bunch_matchers=[template, "not_a_matcher"],
            bucket_indices=[0, 5],
        )
    with pytest.raises(ValueError, match="bunch intensities"):
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=template,
            n_bunches=2,
            bunch_spacing_buckets=2,
            bunch_intensities=[1e11, 1e11, 1e11],
        )
        simulation, beam = _build_simulation()
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)


def test_verbose_and_plot_smoke(capsys):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    simulation, beam = _build_simulation(resonator_r_shunt=1e4, intensity=2e11)
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=_template(n_macroparticles=1_000, n_points_grid=200),
        n_bunches=2,
        bunch_spacing_buckets=3,
        verbose=True,
        plot=True,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    assert "SequentialMultiBunchMatcher" in capsys.readouterr().out
    plt.close("all")


def test_train_is_stationary_over_turns():
    simulation, beam = _build_simulation(resonator_r_shunt=1e4, intensity=2e11)
    matcher = SequentialMultiBunchMatcher(
        bunch_matchers=_template(),
        n_bunches=2,
        bunch_spacing_buckets=4,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    dt = copy_to_cpu(beam.read_partial_dt())
    initial_positions, initial_lengths = _bunch_positions_and_lengths(
        dt, matcher.bucket_indices
    )
    simulation.run_simulation(
        beams=(beam,), n_turns=30, show_progressbar=False
    )
    final_dt = copy_to_cpu(beam.read_partial_dt())
    final_positions, final_lengths = _bunch_positions_and_lengths(
        final_dt, matcher.bucket_indices
    )
    np.testing.assert_allclose(final_lengths, initial_lengths, rtol=5e-2)
    np.testing.assert_allclose(
        final_positions, initial_positions, atol=0.05e-9
    )


# ---------------------- SelfConsistentMultiBunchMatcher --------------------


def _train_specs():
    """EX_31-like per-bunch specs (reduced resolution)."""
    lengths = [1.2e-9, 1.1e-9, 1.3e-9, 1.2e-9]
    return [
        _template(bunch_length=length, seed=bunch_i, relaxation_factor=0.5)
        for bunch_i, length in enumerate(lengths)
    ]


TRAIN_INTENSITIES = [2.0e11, 1.6e11, 2.4e11, 2.0e11]


def test_self_consistent_agrees_with_sequential():
    # With causal (open-boundary) wakes the sequential method already
    # sits at the self-consistent fixed point: both matchers must give
    # the same train. Same seeds -> sampling noise cancels in the
    # comparison.
    results = {}
    for label, matcher_class in (
        ("sequential", SequentialMultiBunchMatcher),
        ("self_consistent", SelfConsistentMultiBunchMatcher),
    ):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e5,
            intensity=sum(TRAIN_INTENSITIES),
            n_buckets=41,
            resonator_frequency=2e8,
            resonator_quality=10.0,
        )
        kwargs = dict(
            bunch_matchers=_train_specs(),
            n_bunches=4,
            bunch_spacing_buckets=10,
            bunch_intensities=TRAIN_INTENSITIES,
        )
        if matcher_class is SelfConsistentMultiBunchMatcher:
            kwargs["relaxation_factor"] = 0.5
        matcher = matcher_class(**kwargs)
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        dt = copy_to_cpu(beam.read_partial_dt())
        results[label] = _bunch_positions_and_lengths(
            dt, matcher.bucket_indices
        )
        if matcher_class is SelfConsistentMultiBunchMatcher:
            assert matcher.final_potential_well_error < 1e-6

    np.testing.assert_allclose(
        results["self_consistent"][0],
        results["sequential"][0],
        atol=0.5e-12,
    )
    np.testing.assert_allclose(
        results["self_consistent"][1],
        results["sequential"][1],
        rtol=2e-3,
    )


def test_self_consistent_periodic_wraps_the_wake():
    # With a periodic solver and train_periodicity, the wake of the
    # trailing bunches wraps around onto the first bunch — a
    # configuration the open-boundary methods cannot represent: the
    # first bunch's position must differ measurably.
    n_buckets_period = 40
    train_periodicity = n_buckets_period * RF_PERIOD

    positions = {}
    for label, solver, periodicity in (
        ("open", None, None),
        (
            "periodic",
            PeriodicFreqSolver(t_periodicity=train_periodicity),
            train_periodicity,
        ),
    ):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e5,
            intensity=sum(TRAIN_INTENSITIES),
            n_buckets=n_buckets_period if periodicity else 41,
            resonator_frequency=2e8,
            resonator_quality=10.0,
            solver=solver,
        )
        matcher = SelfConsistentMultiBunchMatcher(
            bunch_matchers=_train_specs(),
            n_bunches=4,
            bunch_spacing_buckets=10,
            bunch_intensities=TRAIN_INTENSITIES,
            relaxation_factor=0.5,
            train_periodicity=periodicity,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        assert matcher.final_potential_well_error < 1e-6
        dt = copy_to_cpu(beam.read_partial_dt())
        positions[label], _ = _bunch_positions_and_lengths(
            dt, matcher.bucket_indices
        )

    # The first bunch now feels the wrapped wake of the whole train.
    assert abs(positions["periodic"][0] - positions["open"][0]) > 2e-12


def test_self_consistent_without_wakefields():
    simulation, beam = _build_simulation()
    matcher = SelfConsistentMultiBunchMatcher(
        bunch_matchers=_template(),
        n_bunches=2,
        bunch_spacing_buckets=5,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    assert matcher.n_intensity_iterations == 0
    dt = copy_to_cpu(beam.read_partial_dt())
    positions, lengths = _bunch_positions_and_lengths(
        dt, matcher.bucket_indices
    )
    np.testing.assert_allclose(
        positions, (matcher.bucket_indices + 0.5) * RF_PERIOD, atol=0.02e-9
    )
    np.testing.assert_allclose(lengths, 1.2e-9, rtol=3e-2)


def test_self_consistent_validation():
    template = _template()
    with pytest.raises(ValueError, match="relaxation_factor"):
        SelfConsistentMultiBunchMatcher(
            bunch_matchers=template,
            n_bunches=2,
            bunch_spacing_buckets=5,
            relaxation_factor=0.0,
        )
    # train_periodicity shorter than the occupied buckets.
    simulation, beam = _build_simulation(resonator_r_shunt=1e4, intensity=2e11)
    matcher = SelfConsistentMultiBunchMatcher(
        bunch_matchers=template,
        n_bunches=2,
        bunch_spacing_buckets=5,
        train_periodicity=3 * RF_PERIOD,
    )
    with pytest.raises(ValueError, match="train_periodicity"):
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
