import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

from blond import DerivativeInterpolator


def _knot_aligned_grid(time, n_sub=51):
    """Dense grid containing every knot, so trapezoid is exact."""
    return np.unique(
        np.concatenate(
            [
                np.linspace(time[k], time[k + 1], n_sub)
                for k in range(len(time) - 1)
            ]
        )
    )


def _reference_on_grid(time, values):
    """
    Independent brute-force implementation of the same scheme.

    Returns the dense grid together with the reference values *on that
    grid*, so the comparison never has to resample the (quadratic) result
    and is not polluted by the resampling error.
    """
    derivative = np.gradient(values, time)
    grid = _knot_aligned_grid(time)
    derivative_grid = np.interp(grid, time, derivative)
    # the integrand is piecewise linear and every knot is a grid point,
    # so the trapezoid rule is exact here
    integral = values[0] + cumulative_trapezoid(
        derivative_grid, grid, initial=0.0
    )
    drift = values[-1] - integral[-1]
    duration = time[-1] - time[0]
    return grid, integral + drift * (grid - time[0]) / duration


class TestDerivativeInterpolator(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 1.2, 61)
        # smooth, monotonic acceleration ramp 1 GeV/c -> 25 GeV/c
        self.values = 1e9 + 24e9 * (
            0.5 - 0.5 * np.cos(np.pi * self.time / self.time[-1])
        )

    def test_linear_program_is_reproduced_exactly(self):
        time = np.linspace(0.0, 2.0, 25)
        values = 3.0e9 + 7.0e9 * time

        interpolator = DerivativeInterpolator(time, values)

        time_eval = np.linspace(0.0, 2.0, 977)
        np.testing.assert_allclose(
            interpolator(time_eval),
            3.0e9 + 7.0e9 * time_eval,
            rtol=1e-12,
        )

    def test_matches_brute_force_integration_of_the_derivative(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        grid, reference = _reference_on_grid(self.time, self.values)
        np.testing.assert_allclose(interpolator(grid), reference, rtol=1e-10)

    def test_handles_non_equidistant_samples(self):
        # nothing in the scheme may assume a uniform grid: every spacing
        # enters per segment, so a quadratically stretched grid must give
        # the same answer as the brute-force reference.
        time = 1.2 * np.linspace(0.0, 1.0, 41) ** 2
        values = 1e9 + 24e9 * (0.5 - 0.5 * np.cos(np.pi * time / time[-1]))

        interpolator = DerivativeInterpolator(time, values)

        grid, reference = _reference_on_grid(time, values)
        np.testing.assert_allclose(interpolator(grid), reference, rtol=1e-10)

    def test_linear_program_is_exact_on_non_equidistant_samples(self):
        time = np.array([0.0, 0.05, 0.2, 0.21, 0.9, 2.0])
        values = 3.0e9 + 7.0e9 * time

        interpolator = DerivativeInterpolator(time, values)

        time_eval = np.linspace(0.0, 2.0, 977)
        np.testing.assert_allclose(
            interpolator(time_eval),
            3.0e9 + 7.0e9 * time_eval,
            rtol=1e-12,
        )

    def test_converges_at_second_order_on_stretched_grids(self):
        # The scheme does not reproduce the samples exactly, but the
        # deviation must vanish at second order in the sample spacing. Had
        # the implementation assumed an equidistant grid anywhere, the
        # stretched grids below would drop to first order (or stall).
        def deviation(n_samples, stretch):
            time = 1.2 * np.linspace(0.0, 1.0, n_samples) ** stretch
            values = 1e9 + 24e9 * (0.5 - 0.5 * np.cos(np.pi * time / time[-1]))
            interpolator = DerivativeInterpolator(time, values)
            return np.max(np.abs(interpolator(time) - values)) / (
                values[-1] - values[0]
            )

        for stretch in (1, 2, 3):  # equidistant, quadratic, cubic spacing
            coarse = deviation(81, stretch)
            fine = deviation(321, stretch)
            order = np.log2(coarse / fine) / 2.0
            self.assertGreater(order, 1.9, f"{stretch=} gave {order=}")

    def test_endpoints_are_pinned_to_the_input_values(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        np.testing.assert_allclose(
            interpolator(self.time[0]), self.values[0], rtol=1e-14
        )
        np.testing.assert_allclose(
            interpolator(self.time[-1]), self.values[-1], rtol=1e-14
        )

    def test_bump_program_keeps_its_excursion(self):
        # returns to its starting value: BLonD 2's multiplicative endpoint
        # rescale divides by the (zero) total swing and flattens this.
        time = np.linspace(0.0, 1.0, 81)
        amplitude = 4e9
        values = 1e9 + amplitude * np.sin(np.pi * time)

        interpolator = DerivativeInterpolator(time, values)

        peak = np.max(interpolator(np.linspace(0.0, 1.0, 2001)))
        self.assertGreater(peak - 1e9, 0.9 * amplitude)
        np.testing.assert_allclose(interpolator(0.0), 1e9, rtol=1e-14)
        np.testing.assert_allclose(interpolator(1.0), 1e9, rtol=1e-14)

    def test_flat_program_stays_constant(self):
        time = np.linspace(0.0, 1.0, 11)
        values = np.full_like(time, 4.5e9)

        interpolator = DerivativeInterpolator(time, values)

        result = interpolator(np.linspace(0.0, 1.0, 101))
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(result, 4.5e9, rtol=1e-14)

    def test_derivative_is_continuous_across_the_knots(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        step = 1e-7
        for knot in self.time[1:-1]:
            slope_before = (
                interpolator(knot) - interpolator(knot - step)
            ) / step
            slope_after = (
                interpolator(knot + step) - interpolator(knot)
            ) / step
            np.testing.assert_allclose(slope_before, slope_after, rtol=1e-4)

    def test_result_does_not_depend_on_the_evaluation_grid(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        time_probe = np.array([0.13, 0.4, 0.777, 1.05])
        dense = np.linspace(self.time[0], self.time[-1], 100_001)

        np.testing.assert_allclose(
            interpolator(time_probe),
            [interpolator(float(t)) for t in time_probe],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            interpolator(np.array([dense[7000]])),
            interpolator(dense)[7000],
            rtol=1e-14,
        )

    def test_scalar_input_gives_scalar_output(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        self.assertEqual(np.ndim(interpolator(0.5)), 0)
        self.assertEqual(interpolator(np.array([0.5, 0.6])).shape, (2,))

    def test_out_of_range_clamps_to_the_edge_values_by_default(self):
        interpolator = DerivativeInterpolator(self.time, self.values)

        np.testing.assert_allclose(
            interpolator(-1.0), self.values[0], rtol=1e-14
        )
        np.testing.assert_allclose(
            interpolator(99.0), self.values[-1], rtol=1e-14
        )

    def test_out_of_range_uses_the_given_left_and_right_values(self):
        interpolator = DerivativeInterpolator(
            self.time, self.values, left=-5.0, right=7.0
        )

        np.testing.assert_allclose(interpolator(-1.0), -5.0)
        np.testing.assert_allclose(interpolator(99.0), 7.0)

    def test_out_of_range_raises_when_left_or_right_is_nan(self):
        interpolator = DerivativeInterpolator(
            self.time, self.values, left=np.nan, right=np.nan
        )

        # inside the range this must not raise
        interpolator(0.5)

        with self.assertRaises(ValueError):
            interpolator(-1.0)
        with self.assertRaises(ValueError):
            interpolator(99.0)
        with self.assertRaises(ValueError):
            interpolator(np.array([0.5, 99.0]))

    def test_rejects_non_monotonic_time(self):
        with self.assertRaises(AssertionError):
            DerivativeInterpolator(
                np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])
            )

    def test_rejects_duplicated_time(self):
        with self.assertRaises(AssertionError):
            DerivativeInterpolator(
                np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])
            )

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(AssertionError):
            DerivativeInterpolator(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0])
            )

    def test_rejects_too_few_points(self):
        with self.assertRaises(AssertionError):
            DerivativeInterpolator(np.array([0.0]), np.array([1.0]))

    def test_rejects_nan_input(self):
        with self.assertRaises(AssertionError):
            DerivativeInterpolator(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, np.nan, 3.0])
            )


class TestDerivativeInterpolatorInMagneticCycle(unittest.TestCase):
    def test_drives_a_magnetic_cycle_by_time(self):
        from blond import MagneticCycleByTime, proton

        time = np.linspace(0.0, 1.0, 51)
        momentum = 26e9 + 4e9 * (0.5 - 0.5 * np.cos(np.pi * time))

        cycle = MagneticCycleByTime.headless(
            reference_particle=proton,
            base_time=time,
            base_values=momentum,
            in_unit="momentum",
            interpolator=DerivativeInterpolator,
        )

        energy_start = cycle.get_target_total_energy(
            turn_i=0, section_i=0, reference_time=0.0, particle_type=proton
        )
        energy_end = cycle.get_target_total_energy(
            turn_i=0, section_i=0, reference_time=1.0, particle_type=proton
        )

        np.testing.assert_allclose(
            energy_start, np.sqrt(26e9**2 + proton.mass**2), rtol=1e-12
        )
        np.testing.assert_allclose(
            energy_end, np.sqrt(30e9**2 + proton.mass**2), rtol=1e-12
        )


class TestAgreementWithBlond2(unittest.TestCase):
    def test_reproduces_the_blond2_derivative_preprocessing(self):
        from blond import proton
        from blond.legacy.blond2.input_parameters.ring_options import (
            RingOptions,
        )

        circumference = 2 * np.pi * 100.0
        time = np.linspace(0.0, 0.5, 51)
        momentum = 2e9 + 24e9 * (0.5 - 0.5 * np.cos(np.pi * time / 0.5))

        time_blond2, momentum_blond2 = RingOptions(
            interpolation="derivative"
        ).preprocess(
            mass=proton.mass,
            circumference=circumference,
            time=time,
            momentum=momentum,
        )
        n_common = min(len(time_blond2), len(momentum_blond2))
        time_blond2 = np.asarray(time_blond2[:n_common])
        momentum_blond2 = np.asarray(momentum_blond2[:n_common])

        interpolator = DerivativeInterpolator(time, momentum)

        # The schemes differ only in quadrature order and in how the
        # residual endpoint drift is absorbed, i.e. same physics. BLonD 2
        # integrates with a right-endpoint rectangle rule stepped by the
        # revolution period `dt`, which drifts by ~dt/2 * delta(dp/dt),
        # here some 10^4 eV/c out of 2.6*10^10 eV/c. That first-order
        # error is what the closed-form integration removes, so the two
        # may not agree to better than ~10^-5 relative.
        np.testing.assert_allclose(
            interpolator(time_blond2),
            momentum_blond2,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
