"""Tests for the multi-bunch matchers and the matcher clone helper."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

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
from blond.core.backends.backend import backend
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

TRAIN_INTENSITIES = [2.0e11, 1.6e11, 2.4e11, 2.0e11]


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


def _train_specs():
    """Build EX_31-like per-bunch specs at reduced resolution.

    Returns
    -------
    list
        One single-bunch matcher per bunch of the train.
    """
    lengths = [1.2e-9, 1.1e-9, 1.3e-9, 1.2e-9]
    return [
        _template(bunch_length=length, seed=bunch_i, relaxation_factor=0.5)
        for bunch_i, length in enumerate(lengths)
    ]


def _bunch_positions_and_lengths(dt, bucket_indices):
    positions, lengths = [], []
    for bucket_index in bucket_indices:
        selection = (dt > bucket_index * RF_PERIOD) & (
            dt < (bucket_index + 1) * RF_PERIOD
        )
        positions.append(float(np.mean(dt[selection])))
        lengths.append(float(4.0 * np.std(dt[selection])))
    return np.array(positions), np.array(lengths)


def _measured_line_density():
    time_measured = backend.linspace(-1e-9, 1e-9, 101, dtype=backend.float)
    profile = line_density(
        time_measured, "binomial", 1.6e-9, bunch_position=0.0, exponent=1.5
    )
    return time_measured, profile


class TestMatcherClone(unittest.TestCase):
    def test_overrides_and_independence(self):
        template = _template()
        varied = template.clone(bunch_length=1.0e-9, seed=7)
        self.assertIsNot(varied, template)
        self.assertEqual(varied._bunch_length, 1.0e-9)
        self.assertEqual(varied._seed, 7)
        # Untouched arguments are inherited; the original is unmodified.
        self.assertEqual(varied._distribution_type, "parabolic_amplitude")
        self.assertEqual(template._bunch_length, 1.2e-9)
        self.assertEqual(template._seed, 0)

    def test_works_for_line_density_matcher(self):
        time_measured, profile = _measured_line_density()
        matcher = LineDensityMatcher(
            n_macroparticles=1_000,
            time_array=time_measured,
            line_density_values=profile,
            seed=0,
        )
        varied = matcher.clone(half_option="both", seed=3)
        self.assertEqual(varied._half_option, "both")
        self.assertEqual(varied._seed, 3)
        np.testing.assert_array_equal(
            copy_to_cpu(varied._input_time), copy_to_cpu(time_measured)
        )

    def test_rejects_unknown_argument(self):
        with self.assertRaisesRegex(TypeError, "not_a_parameter"):
            _template().clone(not_a_parameter=1.0)


class TestSequentialMultiBunchMatcher(unittest.TestCase):
    def test_train_positions_lengths_and_independent_noise(self):
        simulation, beam = _build_simulation()
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=_template(),
            n_bunches=3,
            bunch_spacing_buckets=5,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)

        np.testing.assert_array_equal(matcher.bucket_indices, [0, 5, 10])
        dt = copy_to_cpu(beam.read_partial_dt())
        self.assertEqual(len(dt), 3 * 2_000)
        positions, lengths = _bunch_positions_and_lengths(
            dt, matcher.bucket_indices
        )
        np.testing.assert_allclose(
            positions,
            (matcher.bucket_indices + 0.5) * RF_PERIOD,
            atol=0.02e-9,
        )
        np.testing.assert_allclose(lengths, 1.2e-9, rtol=3e-2)
        # Template mode derives per-bunch seeds: independent noise, so
        # the local coordinates must differ bunch to bunch.
        local_first = dt[:2_000]
        local_second = dt[2_000:4_000] - 5 * RF_PERIOD
        self.assertFalse(np.allclose(local_first, local_second, atol=1e-13))
        self.assertEqual([m._seed for m in matcher.bunch_matchers], [0, 1, 2])

    def test_per_bunch_parameters_and_mixed_types(self):
        time_measured, profile = _measured_line_density()
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
        np.testing.assert_allclose(lengths[0], 1.2e-9, rtol=3e-2)
        np.testing.assert_allclose(lengths[1], 1.0e-9, rtol=3e-2)
        np.testing.assert_allclose(
            lengths[2],
            matcher.bunch_matchers[2].matched_bunch_length,
            rtol=3e-2,
        )
        # The user's spec instances were deep-copied, not run.
        self.assertIsNone(bunch_matchers[0].matched_bunch_length)
        self.assertIsNotNone(matcher.bunch_matchers[0].matched_bunch_length)

    def test_wake_of_predecessor_shifts_next_bunch(self):
        # A long-memory resonator (decay over several buckets) so the
        # predecessor's wake reaches the next bucket. Reference: a
        # single bunch alone. In the two-bunch train, the first bunch
        # (no predecessor) must reproduce the reference exactly, while
        # the second must sit at a measurably different position.
        shift_single, _ = self._run_train([0])
        shifts_train, matcher = self._run_train([0, 1])

        # Self-wake shift is real and reproduced for the first bunch.
        self.assertGreater(abs(shift_single[0]), 0.005e-9)
        np.testing.assert_allclose(
            shifts_train[0], shift_single[0], atol=0.002e-9
        )
        # The predecessor's wake moves the second bunch measurably.
        self.assertGreater(abs(shifts_train[1] - shift_single[0]), 0.01e-9)
        # Each bunch ran its own converged self-wake iteration.
        for bunch_matcher in matcher.bunch_matchers:
            self.assertGreaterEqual(bunch_matcher.n_intensity_iterations, 1)
            self.assertLess(bunch_matcher.final_potential_well_error, 1e-6)

    @staticmethod
    def _run_train(bucket_indices):
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

    def test_intensity_is_split_equally_by_default(self):
        simulation, beam = _build_simulation(intensity=3e11)
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=_template(),
            n_bunches=3,
            bunch_spacing_buckets=2,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        np.testing.assert_allclose(matcher.bunch_intensities, 1e11)

    def test_mismatching_per_bunch_sum_warns_and_overwrites(self):
        # BLonD 2 behaviour: warn and overwrite the beam intensity.
        simulation, beam = _build_simulation(intensity=3e11)
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=_template(),
            n_bunches=2,
            bunch_spacing_buckets=2,
            bunch_intensities=[1e11, 2.5e11],
        )
        with self.assertWarnsRegex(UserWarning, "overwritten"):
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assertEqual(beam.intensity, 3.5e11)

    def test_train_is_stationary_over_turns(self):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e4, intensity=2e11
        )
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

    def test_verbose_and_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        simulation, beam = _build_simulation(
            resonator_r_shunt=1e4, intensity=2e11
        )
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=_template(
                n_macroparticles=1_000, n_points_grid=200
            ),
            n_bunches=2,
            bunch_spacing_buckets=3,
            verbose=True,
            plot=True,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assertIn("SequentialMultiBunchMatcher", stdout.getvalue())
        plt.close("all")


class TestSequentialMultiBunchMatcherValidation(unittest.TestCase):
    def setUp(self):
        self.template = _template()

    def test_no_bucket_specification_raises(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            SequentialMultiBunchMatcher(bunch_matchers=self.template)

    def test_two_bucket_specifications_raise(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            SequentialMultiBunchMatcher(
                bunch_matchers=self.template,
                bucket_indices=[0, 5],
                n_bunches=2,
            )

    def test_n_bunches_without_spacing_raises(self):
        with self.assertRaisesRegex(ValueError, "bunch_spacing_buckets"):
            SequentialMultiBunchMatcher(
                bunch_matchers=self.template, n_bunches=2
            )

    def test_decreasing_bucket_indices_raise(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            SequentialMultiBunchMatcher(
                bunch_matchers=self.template, bucket_indices=[5, 0]
            )

    def test_matcher_count_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "bunch matchers"):
            SequentialMultiBunchMatcher(
                bunch_matchers=[self.template], bucket_indices=[0, 5]
            )

    def test_non_matcher_entry_raises(self):
        with self.assertRaisesRegex(TypeError, "single-bunch matcher"):
            SequentialMultiBunchMatcher(
                bunch_matchers=[self.template, "not_a_matcher"],
                bucket_indices=[0, 5],
            )

    def test_intensity_count_mismatch_raises(self):
        matcher = SequentialMultiBunchMatcher(
            bunch_matchers=self.template,
            n_bunches=2,
            bunch_spacing_buckets=2,
            bunch_intensities=[1e11, 1e11, 1e11],
        )
        simulation, beam = _build_simulation()
        with self.assertRaisesRegex(ValueError, "bunch intensities"):
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)


class TestSelfConsistentMultiBunchMatcher(unittest.TestCase):
    def test_agrees_with_sequential(self):
        # With causal (open-boundary) wakes the sequential method
        # already sits at the self-consistent fixed point: both
        # matchers must give the same train. Same seeds -> sampling
        # noise cancels in the comparison.
        results = {}
        for label, matcher_class in (
            ("sequential", SequentialMultiBunchMatcher),
            ("self_consistent", SelfConsistentMultiBunchMatcher),
        ):
            with self.subTest(matcher=label):
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
                    self.assertLess(matcher.final_potential_well_error, 1e-6)

        np.testing.assert_allclose(
            results["self_consistent"][0],
            results["sequential"][0],
            atol=3e-12,
        )
        np.testing.assert_allclose(
            results["self_consistent"][1],
            results["sequential"][1],
            rtol=2e-3,
        )

    def test_periodic_wraps_the_wake(self):
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
            with self.subTest(boundary=label):
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
                self.assertLess(matcher.final_potential_well_error, 1e-6)
                dt = copy_to_cpu(beam.read_partial_dt())
                positions[label], _ = _bunch_positions_and_lengths(
                    dt, matcher.bucket_indices
                )

        # The first bunch now feels the wrapped wake of the whole train.
        self.assertGreater(
            abs(positions["periodic"][0] - positions["open"][0]), 2e-12
        )

    def test_without_wakefields(self):
        simulation, beam = _build_simulation()
        matcher = SelfConsistentMultiBunchMatcher(
            bunch_matchers=_template(),
            n_bunches=2,
            bunch_spacing_buckets=5,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assertEqual(matcher.n_intensity_iterations, 0)
        dt = copy_to_cpu(beam.read_partial_dt())
        positions, lengths = _bunch_positions_and_lengths(
            dt, matcher.bucket_indices
        )
        np.testing.assert_allclose(
            positions,
            (matcher.bucket_indices + 0.5) * RF_PERIOD,
            atol=0.02e-9,
        )
        np.testing.assert_allclose(lengths, 1.2e-9, rtol=3e-2)

    def test_invalid_relaxation_factor_raises(self):
        with self.assertRaisesRegex(ValueError, "relaxation_factor"):
            SelfConsistentMultiBunchMatcher(
                bunch_matchers=_template(),
                n_bunches=2,
                bunch_spacing_buckets=5,
                relaxation_factor=0.0,
            )

    def test_too_short_train_periodicity_raises(self):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e4, intensity=2e11
        )
        matcher = SelfConsistentMultiBunchMatcher(
            bunch_matchers=_template(),
            n_bunches=2,
            bunch_spacing_buckets=5,
            train_periodicity=3 * RF_PERIOD,
        )
        with self.assertRaisesRegex(ValueError, "train_periodicity"):
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)


if __name__ == "__main__":
    unittest.main()
