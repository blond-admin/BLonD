"""Tests for the AnalyticDistributionMatcher and LineDensityMatcher."""

import unittest

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
    _total_rf_voltage,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import TimeDomainFftSolver

RF_PERIOD = 2.0 * np.pi / 2518229887.224505


def _build_simulation(resonator_r_shunt=None, intensity=1e11):
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
        profile = StaticProfile(cut_left=0.0, cut_right=RF_PERIOD, n_bins=512)
        wakefield = WakeField(
            sources=(Resonators(resonator_r_shunt, 8e8, 1.0),),
            solver=TimeDomainFftSolver(),
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


def _build_simulation_local_wakefield(resonator_r_shunt, intensity=1e11):
    """Build a simulation whose wakefield is local to the RF station.

    Parameters
    ----------
    resonator_r_shunt : float
        Resonator shunt impedance in ohm.
    intensity : float
        Beam intensity in number of particles.

    Returns
    -------
    tuple
        The ``Simulation`` and the ``Beam`` to prepare.
    """
    ring = Ring(26658.883)
    profile = StaticProfile(cut_left=0.0, cut_right=RF_PERIOD, n_bins=512)
    wakefield = WakeField(
        sources=(Resonators(resonator_r_shunt, 8e8, 1.0),),
        solver=TimeDomainFftSolver(),
        profile=profile,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=35640,
        voltage=6e6,
        phi_rf=0,
        local_wakefield=wakefield,
    )
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    ring.add_elements([rf_station, drift, profile], reorder=True)
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=intensity, particle_type=proton)
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    return simulation, beam


def _intensity_matcher(relaxation_factor=1.0, maxiter=100, target=1.2e-9):
    return AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=target,
        seed=0,
        n_points_grid=300,
        maxiter_intensity_effects=maxiter,
        relaxation_factor=relaxation_factor,
        allow_inner_buckets=True,
    )


def _measured_profile(
    full_length=1.6e-9, position=0.15e-9, baseline=0.0, n_samples=201
):
    """Build a synthetic profile on its own, bucket-unrelated, axis.

    Parameters
    ----------
    full_length : float
        Full bunch length in seconds.
    position : float
        Bunch position on the measured axis, in seconds.
    baseline : float
        Constant offset added to the profile.
    n_samples : int
        Number of samples on the measured axis.

    Returns
    -------
    tuple
        The measured time axis and the line density values.
    """
    time_measured = backend.linspace(
        -1.0e-9, 1.0e-9, n_samples, dtype=backend.float
    )
    profile = (
        line_density(
            time_measured,
            "binomial",
            full_length,
            bunch_position=position,
            exponent=1.5,
        )
        + baseline
    )
    return time_measured, profile


def _parabolic_peak_time(time_array, density):
    """Locate a density peak to sub-bin accuracy.

    A three-point parabolic fit around the maximum sample, so the
    result does not quantise to the grid.

    Parameters
    ----------
    time_array : NumpyArray | CupyArray
        Sample times, in [s].
    density : NumpyArray | CupyArray
        Density sampled on ``time_array``.

    Returns
    -------
    float
        Time of the fitted maximum, in [s].
    """
    peak_index = int(backend.argmax(density))
    if peak_index in (0, len(density) - 1):
        return float(time_array[peak_index])
    lower = float(density[peak_index - 1])
    centre = float(density[peak_index])
    upper = float(density[peak_index + 1])
    curvature = lower - 2.0 * centre + upper
    if curvature == 0.0:
        return float(time_array[peak_index])
    bin_width = float(time_array[peak_index + 1] - time_array[peak_index])
    offset = 0.5 * (lower - upper) / curvature
    return float(time_array[peak_index]) + offset * bin_width


def _extra_voltage_matcher(extra_voltage=None):
    return AnalyticDistributionMatcher(
        n_macroparticles=2_000,
        distribution_type="parabolic_amplitude",
        bunch_length=1.2e-9,
        seed=0,
        n_points_grid=300,
        allow_inner_buckets=True,
        extra_voltage=extra_voltage,
    )


class TestTotalInputVoltage(unittest.TestCase):
    # Waveform shapes, normalised to unit amplitude over the frame.
    @staticmethod
    def _shapes(span, omega_rf):
        return {
            "linear": lambda t: (t - t[0]) / span,
            "quadratic": lambda t: ((t - t[0]) / span) ** 2,
            "sinusoid": lambda t: backend.sin(3.0 * omega_rf * t),
        }

    def test_extra_waveform_is_added_to_the_rf_voltage(self):
        # The extra voltage must be summed onto the RF waveform as
        # given, for any shape and amplitude. Query points are every
        # second node of the extra-voltage grid, so the interpolation
        # returns node values exactly and the comparison stays exact.
        simulation, _ = _build_simulation()
        extra_time = backend.linspace(
            -RF_PERIOD, 2.0 * RF_PERIOD, 401, dtype=backend.float
        )
        time_array = extra_time[::2]
        rf_voltage = _total_rf_voltage(simulation, time_array)
        span = float(extra_time[-1] - extra_time[0])
        omega_rf = 2.0 * np.pi / RF_PERIOD

        for name, shape in self._shapes(span, omega_rf).items():
            # Far below and comparable to the 6 MV main harmonic.
            for amplitude in (1.0, 5.0e6):
                with self.subTest(waveform=name, amplitude=amplitude):
                    matcher = _extra_voltage_matcher(
                        extra_voltage=(
                            extra_time,
                            amplitude * shape(extra_time),
                        )
                    )
                    total = matcher._total_input_voltage(
                        simulation, time_array
                    )
                    expected = rf_voltage + amplitude * shape(time_array)
                    scale = float(backend.max(backend.abs(expected)))
                    np.testing.assert_allclose(
                        copy_to_cpu(total),
                        copy_to_cpu(expected),
                        rtol=1e-12,
                        atol=1e-12 * scale,
                    )

    def test_without_extra_voltage_returns_the_rf_voltage(self):
        simulation, _ = _build_simulation()
        time_array = backend.linspace(0.0, RF_PERIOD, 201, dtype=backend.float)
        total = _extra_voltage_matcher()._total_input_voltage(
            simulation, time_array
        )
        np.testing.assert_array_equal(
            copy_to_cpu(total),
            copy_to_cpu(_total_rf_voltage(simulation, time_array)),
        )


class TestExtraVoltage(unittest.TestCase):
    def test_shifts_synchronous_position(self):
        # A constant extra voltage V0 moves the zero crossing of the
        # total voltage. At the mid-frame fixed point
        # V sin(omega t) + V0 = 0 gives dt = +asin(V0/V)/omega.
        # V0 is a sixth of the RF voltage, so the ~66 ps shift is an
        # order of magnitude above the centroid sampling noise.
        extra_time = backend.linspace(
            -2.0 * RF_PERIOD, 3.0 * RF_PERIOD, 100, dtype=backend.float
        )
        v_0, v_rf = 1e6, 6e6
        omega_rf = 2.0 * np.pi / RF_PERIOD

        peaks, centroids = {}, {}
        for label, extra in (
            ("bare", None),
            ("offset", (extra_time, v_0 * backend.ones_like(extra_time))),
        ):
            simulation, beam = _build_simulation()
            matcher = _extra_voltage_matcher(extra_voltage=extra)
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)
            peaks[label] = _parabolic_peak_time(
                matcher.matched_time_array, matcher.matched_line_density
            )
            centroids[label] = float(
                np.mean(copy_to_cpu(beam.read_partial_dt()))
            )

        expected_shift = np.arcsin(v_0 / v_rf) / omega_rf
        # The matched density peaks at the well minimum, so it tracks
        # the fixed point directly.
        np.testing.assert_allclose(
            peaks["offset"] - peaks["bare"], expected_shift, rtol=0.02
        )
        # The sampled beam follows, but its centroid sits ~50 % beyond
        # the minimum: the tilted well is asymmetric. The loose bound
        # checks the particles moved with the bucket, without pinning
        # that asymmetry.
        np.testing.assert_allclose(
            centroids["offset"] - centroids["bare"],
            expected_shift,
            rtol=0.6,
        )

    def test_requires_a_pair(self):
        with self.assertRaisesRegex(ValueError, "pair"):
            _extra_voltage_matcher(extra_voltage=(np.zeros(4),))

    def test_requires_increasing_time(self):
        with self.assertRaisesRegex(AssertionError, "increasing"):
            _extra_voltage_matcher(
                extra_voltage=(
                    np.array([1.0, 0.0]),
                    np.array([0.0, 0.0]),
                )
            )


class _StationarityMixin:
    """Shared check that a matched bunch neither blows up nor drifts."""

    def assert_stationary_over_turns(self, simulation, beam, n_turns=30):
        dt = copy_to_cpu(beam.read_partial_dt())
        initial_length = 4.0 * float(np.std(dt))
        initial_position = float(np.mean(dt))
        simulation.run_simulation(
            beams=(beam,), n_turns=n_turns, show_progressbar=False
        )
        final_dt = copy_to_cpu(beam.read_partial_dt())
        final_length = 4.0 * float(np.std(final_dt))
        final_position = float(np.mean(final_dt))
        self.assertLess(
            abs(final_length - initial_length) / initial_length, 0.05
        )
        self.assertLess(abs(final_position - initial_position), 0.05e-9)


class TestAnalyticDistributionMatcher(_StationarityMixin, unittest.TestCase):
    def test_matched_bunch_length_and_position(self):
        simulation, beam = _build_simulation()
        target = 1.2e-9  # 4-sigma rms
        matcher = AnalyticDistributionMatcher(
            n_macroparticles=20_000,
            distribution_type="parabolic_amplitude",
            bunch_length=target,
            seed=0,
            n_points_grid=500,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)

        dt = copy_to_cpu(beam.read_partial_dt())
        dE = copy_to_cpu(beam.read_partial_dE())
        self.assertEqual(len(dt), 20_000)
        # Matched density bunch length equals the target within the grid.
        np.testing.assert_allclose(
            matcher.matched_bunch_length, target, rtol=1e-2
        )
        # Sampled bunch length within statistics (20k particles ~ 1%).
        np.testing.assert_allclose(4.0 * np.std(dt), target, rtol=3e-2)
        # Centred on the stable phase (half an RF period for phi_rf=0
        # above transition).
        np.testing.assert_allclose(np.mean(dt), RF_PERIOD / 2.0, atol=0.02e-9)
        # All particles inside the bucket frame, energies inside the
        # separatrix half height (~390 MeV).
        self.assertGreater(dt.min(), 0.0)
        self.assertLess(dt.max(), RF_PERIOD)
        self.assertLess(np.max(np.abs(dE)), 4.0e8)

    def test_emittance_target(self):
        simulation, beam = _build_simulation()
        matcher = AnalyticDistributionMatcher(
            n_macroparticles=5_000,
            distribution_type="gaussian",
            emittance=0.7,  # eV.s, inside the 1.24 eV.s bucket
            seed=0,
            n_points_grid=400,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assertIsNotNone(matcher.fitted_x_0)
        self.assertGreater(matcher.fitted_x_0, 0.0)
        self.assertLess(matcher.fitted_x_0, 53.6)
        self.assertGreater(matcher.matched_bunch_length, 0.0)

    def test_seed_reproducibility(self):
        dts = []
        for _ in range(2):
            simulation, beam = _build_simulation()
            matcher = AnalyticDistributionMatcher(
                n_macroparticles=2_000,
                distribution_type="gaussian",
                bunch_length=1.0e-9,
                seed=42,
                n_points_grid=300,
            )
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)
            dts.append(copy_to_cpu(beam.read_partial_dt()).copy())
        np.testing.assert_array_equal(dts[0], dts[1])

    def test_matched_emittance_round_trip(self):
        # The bunch-length target reports the emittance of the matched
        # contour; targeting that emittance must recover the length.
        simulation, beam = _build_simulation()
        matcher_length = AnalyticDistributionMatcher(
            n_macroparticles=2_000,
            distribution_type="parabolic_amplitude",
            bunch_length=1.2e-9,
            seed=0,
            n_points_grid=400,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher_length)
        self.assertIsNotNone(matcher_length.matched_emittance)
        self.assertGreater(matcher_length.matched_emittance, 0.0)
        self.assertLess(matcher_length.matched_emittance, 1.24)  # bucket area

        simulation, beam = _build_simulation()
        matcher_emittance = AnalyticDistributionMatcher(
            n_macroparticles=2_000,
            distribution_type="parabolic_amplitude",
            emittance=matcher_length.matched_emittance,
            seed=0,
            n_points_grid=400,
        )
        simulation.prepare_beam(
            beam=beam, preparation_routine=matcher_emittance
        )
        np.testing.assert_allclose(
            matcher_emittance.matched_bunch_length, 1.2e-9, rtol=1e-2
        )
        np.testing.assert_allclose(
            matcher_emittance.matched_emittance,
            matcher_length.matched_emittance,
            rtol=1e-3,
        )

    def test_matched_bunch_is_stationary_over_turns(self):
        simulation, beam = _build_simulation()
        matcher = AnalyticDistributionMatcher(
            n_macroparticles=2_000,
            distribution_type="parabolic_amplitude",
            bunch_length=1.2e-9,
            seed=1,
            n_points_grid=400,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assert_stationary_over_turns(simulation, beam)

    def test_target_validation(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            AnalyticDistributionMatcher(
                n_macroparticles=1000,
                distribution_type="gaussian",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            AnalyticDistributionMatcher(
                n_macroparticles=1000,
                distribution_type="gaussian",
                bunch_length=1e-9,
                emittance=0.5,
            )

    def test_relaxation_factor_validation(self):
        for bad_value in (0.0, 1.5, -0.3):
            with self.subTest(relaxation_factor=bad_value):
                with self.assertRaisesRegex(ValueError, "relaxation_factor"):
                    AnalyticDistributionMatcher(
                        n_macroparticles=1000,
                        distribution_type="gaussian",
                        bunch_length=1e-9,
                        relaxation_factor=bad_value,
                    )

    def test_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        simulation, beam = _build_simulation()
        matcher = AnalyticDistributionMatcher(
            n_macroparticles=1_000,
            distribution_type="gaussian",
            bunch_length=1.0e-9,
            seed=0,
            n_points_grid=200,
            plot=True,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        plt.close("all")


class TestAnalyticMatcherIntensityEffects(
    _StationarityMixin, unittest.TestCase
):
    def _assert_converged(self, matcher):
        self.assertGreaterEqual(matcher.n_intensity_iterations, 1)
        self.assertLessEqual(matcher.n_intensity_iterations, 20)
        self.assertLess(matcher.final_potential_well_error, 1e-6)
        np.testing.assert_allclose(
            matcher.matched_bunch_length, 1.2e-9, rtol=1e-2
        )
        self.assertEqual(
            len(matcher.intensity_residuals), matcher.n_intensity_iterations
        )
        # The contour emittance is evaluated in the distorted well.
        self.assertIsNotNone(matcher.matched_emittance)
        self.assertGreater(matcher.matched_emittance, 0.0)
        self.assertLess(matcher.matched_emittance, 1.24)

    def test_intensity_effects_converge(self):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e4, intensity=2e11
        )
        matcher = _intensity_matcher()
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self._assert_converged(matcher)

    def test_intensity_effects_converge_local_wakefield(self):
        # Same setup, but the wakefield is attached to the cavity's
        # local_wakefield instead of a top-level ring.elements entry.
        simulation, beam = _build_simulation_local_wakefield(
            resonator_r_shunt=1e4, intensity=2e11
        )
        matcher = _intensity_matcher()
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self._assert_converged(matcher)

    def test_weak_intensity_matches_zero_intensity_limit(self):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1.0, intensity=2e11
        )
        matcher_weak = _intensity_matcher()
        simulation.prepare_beam(beam=beam, preparation_routine=matcher_weak)
        simulation_0, beam_0 = _build_simulation()
        matcher_0 = _intensity_matcher()
        simulation_0.prepare_beam(beam=beam_0, preparation_routine=matcher_0)
        # A vanishing impedance must reproduce the no-wakefield result.
        # (Frames differ: the wakefield branch adds the legacy 40 %
        # margin.)
        np.testing.assert_allclose(
            matcher_weak.fitted_x_0, matcher_0.fitted_x_0, rtol=1e-3
        )
        np.testing.assert_allclose(
            matcher_weak.matched_bunch_length,
            matcher_0.matched_bunch_length,
            rtol=1e-3,
        )

    def test_relaxation_reaches_same_fixed_point(self):
        results = {}
        for relaxation_factor in (1.0, 0.5):
            with self.subTest(relaxation_factor=relaxation_factor):
                simulation, beam = _build_simulation(
                    resonator_r_shunt=1e5, intensity=2e11
                )
                matcher = _intensity_matcher(
                    relaxation_factor=relaxation_factor, maxiter=200
                )
                simulation.prepare_beam(beam=beam, preparation_routine=matcher)
                self.assertLess(matcher.final_potential_well_error, 1e-6)
                results[relaxation_factor] = matcher
        # Different relaxations reach the same self-consistent match.
        np.testing.assert_allclose(
            results[1.0].fitted_x_0, results[0.5].fitted_x_0, rtol=1e-3
        )
        np.testing.assert_allclose(
            results[1.0].matched_bunch_length,
            results[0.5].matched_bunch_length,
            rtol=1e-3,
        )
        # Under-relaxation takes more iterations.
        self.assertGreater(
            results[0.5].n_intensity_iterations,
            results[1.0].n_intensity_iterations,
        )

    def test_relaxation_stabilizes_strong_intensity(self):
        # At this impedance the full-correction (BLonD 2) iteration
        # oscillates without converging (residual plateaus ~1e-3);
        # under-relaxation converges below 1e-6 in ~50 iterations.
        r_shunt, intensity = 2.5e5, 2e11
        simulation, beam = _build_simulation(
            resonator_r_shunt=r_shunt, intensity=intensity
        )
        matcher_full = _intensity_matcher(relaxation_factor=1.0, maxiter=60)
        with self.assertWarnsRegex(UserWarning, "did not converge"):
            simulation.prepare_beam(
                beam=beam, preparation_routine=matcher_full
            )
        self.assertGreater(matcher_full.final_potential_well_error, 1e-4)

        simulation, beam = _build_simulation(
            resonator_r_shunt=r_shunt, intensity=intensity
        )
        matcher_relaxed = _intensity_matcher(relaxation_factor=0.5, maxiter=60)
        simulation.prepare_beam(beam=beam, preparation_routine=matcher_relaxed)
        self.assertLess(matcher_relaxed.final_potential_well_error, 1e-6)

    def test_intensity_matched_bunch_is_stationary(self):
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e4, intensity=2e11
        )
        matcher = _intensity_matcher()
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assert_stationary_over_turns(simulation, beam)


class TestLineDensityMatcher(_StationarityMixin, unittest.TestCase):
    def test_family_mode(self):
        simulation, beam = _build_simulation()
        full_length = 1.5e-9
        matcher = LineDensityMatcher(
            n_macroparticles=20_000,
            line_density_type="parabolic_amplitude",
            bunch_length=full_length,
            seed=0,
            n_points_grid=500,
            n_points_abel=5_000,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)

        dt = copy_to_cpu(beam.read_partial_dt())
        self.assertEqual(len(dt), 20_000)
        # Full length -> 4 sigma rms: sqrt(6)/2 for parabolic_amplitude.
        expected_4sigma = full_length / (np.sqrt(6.0) / 2.0)
        np.testing.assert_allclose(
            matcher.matched_bunch_length, expected_4sigma, rtol=2e-2
        )
        np.testing.assert_allclose(
            4.0 * np.std(dt), expected_4sigma, rtol=3e-2
        )
        # Centred on the stable phase, everything inside the bucket.
        np.testing.assert_allclose(np.mean(dt), RF_PERIOD / 2.0, atol=0.02e-9)
        self.assertGreater(dt.min(), 0.0)
        self.assertLess(dt.max(), RF_PERIOD)
        # The Abel closure reproduces the requested profile.
        self.assertLess(matcher.profile_reconstruction_error, 0.02)

    def test_measured_mode_recenters(self):
        # An arbitrarily positioned profile with a constant baseline, on
        # its own time axis, is recentred onto the bucket and reproduced.
        time_measured, profile = _measured_profile(
            position=0.15e-9, baseline=0.02
        )
        simulation, beam = _build_simulation()
        matcher = LineDensityMatcher(
            n_macroparticles=20_000,
            time_array=time_measured,
            line_density_values=profile,
            half_option="both",
            seed=0,
            n_points_grid=500,
            n_points_abel=5_000,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)

        dt = copy_to_cpu(beam.read_partial_dt())
        np.testing.assert_allclose(
            matcher.matched_bunch_position, RF_PERIOD / 2.0, atol=0.01e-9
        )
        np.testing.assert_allclose(np.mean(dt), RF_PERIOD / 2.0, atol=0.02e-9)
        self.assertLess(matcher.profile_reconstruction_error, 0.02)
        # Matched length agrees with the input profile's own 4 sigma rms
        # (computed on the baseline-subtracted input).
        clean = profile - profile.min()
        mean_time = backend.sum(clean * time_measured) / backend.sum(clean)
        input_4sigma = 4.0 * float(
            backend.sqrt(
                backend.sum(clean * (time_measured - mean_time) ** 2)
                / backend.sum(clean)
            )
        )
        np.testing.assert_allclose(
            matcher.matched_bunch_length, input_4sigma, rtol=3e-2
        )
        # The input arrays were not mutated.
        self.assertGreater(float(profile.min()), 0.0)

    def test_half_options_consistent(self):
        # In the bare (symmetric) bucket the three half options must
        # give the same match.
        time_measured, profile = _measured_profile(position=0.0)
        lengths = {}
        for half_option in ("first", "second", "both"):
            with self.subTest(half_option=half_option):
                simulation, beam = _build_simulation()
                matcher = LineDensityMatcher(
                    n_macroparticles=2_000,
                    time_array=time_measured,
                    line_density_values=profile,
                    half_option=half_option,
                    seed=0,
                    n_points_grid=400,
                    n_points_abel=3_000,
                )
                simulation.prepare_beam(beam=beam, preparation_routine=matcher)
                lengths[half_option] = matcher.matched_bunch_length
        np.testing.assert_allclose(
            lengths["first"], lengths["second"], rtol=1e-2
        )
        np.testing.assert_allclose(
            lengths["first"], lengths["both"], rtol=1e-2
        )

    def test_barycenter_centering(self):
        # The barycenter mode must centre a (noise-free) profile like
        # the peak mode does.
        time_measured, profile = _measured_profile(position=0.2e-9)
        positions = {}
        for profile_centering in ("peak", "barycenter"):
            with self.subTest(profile_centering=profile_centering):
                simulation, beam = _build_simulation()
                matcher = LineDensityMatcher(
                    n_macroparticles=2_000,
                    time_array=time_measured,
                    line_density_values=profile,
                    profile_centering=profile_centering,
                    seed=0,
                    n_points_grid=400,
                    n_points_abel=3_000,
                )
                simulation.prepare_beam(beam=beam, preparation_routine=matcher)
                positions[profile_centering] = matcher.matched_bunch_position
        np.testing.assert_allclose(
            positions["peak"], positions["barycenter"], atol=0.01e-9
        )

    def test_seed_reproducibility(self):
        time_measured, profile = _measured_profile()
        dts = []
        for _ in range(2):
            simulation, beam = _build_simulation()
            matcher = LineDensityMatcher(
                n_macroparticles=2_000,
                time_array=time_measured,
                line_density_values=profile,
                seed=42,
                n_points_grid=300,
                n_points_abel=2_000,
            )
            simulation.prepare_beam(beam=beam, preparation_routine=matcher)
            dts.append(copy_to_cpu(beam.read_partial_dt()).copy())
        np.testing.assert_array_equal(dts[0], dts[1])

    def test_matched_bunch_is_stationary_over_turns(self):
        time_measured, profile = _measured_profile()
        simulation, beam = _build_simulation()
        matcher = LineDensityMatcher(
            n_macroparticles=2_000,
            time_array=time_measured,
            line_density_values=profile,
            seed=1,
            n_points_grid=400,
            n_points_abel=3_000,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assert_stationary_over_turns(simulation, beam)

    def test_intensity_effects(self):
        # With a wakefield: centering + induced potential iterate to a
        # fixed point, the bunch sits at the wake-shifted position, and
        # the generated bunch is stationary when tracked with the wake.
        time_measured, profile = _measured_profile(position=0.0)
        simulation, beam = _build_simulation(
            resonator_r_shunt=1e5, intensity=2e11
        )
        matcher = LineDensityMatcher(
            n_macroparticles=5_000,
            time_array=time_measured,
            line_density_values=profile,
            half_option="both",
            seed=0,
            n_points_grid=400,
            n_points_abel=3_000,
            allow_inner_buckets=True,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        self.assertGreaterEqual(matcher.n_intensity_iterations, 1)
        self.assertLessEqual(matcher.n_intensity_iterations, 30)
        self.assertLess(matcher.final_potential_well_error, 1e-6)
        # The wake shifts the stable position off the bare-bucket centre.
        self.assertLess(
            matcher.matched_bunch_position, RF_PERIOD / 2.0 - 0.01e-9
        )
        self.assert_stationary_over_turns(simulation, beam)

    def test_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        time_measured, profile = _measured_profile()
        simulation, beam = _build_simulation()
        matcher = LineDensityMatcher(
            n_macroparticles=1_000,
            time_array=time_measured,
            line_density_values=profile,
            seed=0,
            n_points_grid=200,
            n_points_abel=1_000,
            plot=True,
        )
        simulation.prepare_beam(beam=beam, preparation_routine=matcher)
        plt.close("all")


class TestLineDensityMatcherValidation(unittest.TestCase):
    def setUp(self):
        self.time_measured, self.profile = _measured_profile()

    def test_no_target_raises(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            LineDensityMatcher(n_macroparticles=1000)

    def test_two_targets_raise(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            LineDensityMatcher(
                n_macroparticles=1000,
                time_array=self.time_measured,
                line_density_values=self.profile,
                line_density_type="gaussian",
                bunch_length=1e-9,
            )

    def test_unknown_half_option_raises(self):
        with self.assertRaisesRegex(ValueError, "half_option"):
            LineDensityMatcher(
                n_macroparticles=1000,
                time_array=self.time_measured,
                line_density_values=self.profile,
                half_option="not_a_half",
            )

    def test_unknown_profile_centering_raises(self):
        with self.assertRaisesRegex(ValueError, "profile_centering"):
            LineDensityMatcher(
                n_macroparticles=1000,
                time_array=self.time_measured,
                line_density_values=self.profile,
                profile_centering="not_a_mode",
            )

    def test_invalid_relaxation_factor_raises(self):
        with self.assertRaisesRegex(ValueError, "relaxation_factor"):
            LineDensityMatcher(
                n_macroparticles=1000,
                time_array=self.time_measured,
                line_density_values=self.profile,
                relaxation_factor=0.0,
            )

    def test_decreasing_time_array_raises(self):
        with self.assertRaisesRegex(AssertionError, "increasing"):
            LineDensityMatcher(
                n_macroparticles=1000,
                time_array=self.time_measured[::-1],
                line_density_values=self.profile,
            )


if __name__ == "__main__":
    unittest.main()
