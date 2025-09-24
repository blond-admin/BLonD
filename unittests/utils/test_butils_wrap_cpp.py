import unittest

import numpy as np

from blond.utils import butils_wrap_cpp


class TestFunctions(unittest.TestCase):
    def setUp(self):
        butils_wrap_cpp.load_libblond("double")

    def test_add_cpp(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = np.empty_like(a)
        expected = a + b

        butils_wrap_cpp.add_cpp(a=a, b=b, result=result, inplace=False)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_add_cpp_inplace(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        expected = a + b

        butils_wrap_cpp.add_cpp(a=a, b=b, inplace=True)
        np.testing.assert_allclose(a, expected, rtol=1e-12)

    def test_arange_cpp(self):
        start, stop, step = 0.0, 5.0, 1.0
        expected = np.arange(start, stop, step, dtype=np.float64)
        result = np.full_like(expected,fill_value=np.nan)

        butils_wrap_cpp.arange_cpp(
            start=start, stop=stop, step=step, dtype=float, result=result
        )
        np.testing.assert_array_equal(result, expected)

    def test_cos_cpp(self):
        x = np.linspace(0, np.pi, 5)
        expected = np.cos(x)
        result = np.empty_like(x)

        butils_wrap_cpp.cos_cpp(x=x, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @unittest.skip
    def test_cumtrapz(self):
        x = np.linspace(0, np.pi, 100)
        y = np.sin(x)
        from scipy.integrate import cumulative_trapezoid as cumtrapz

        expected = cumtrapz(y, x, initial=0)

        result = np.empty_like(x)
        butils_wrap_cpp.cumtrapz(y=y, x=x, dx=x[1] - x[0], initial=0.0, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @unittest.skip
    def test_distribution_from_tomoscope(self):
        # TODO: implement test for `distribution_from_tomoscope`
        butils_wrap_cpp.distribution_from_tomoscope(
            dt=None,
            dE=None,
            probDistr=None,
            seed=None,
            profLen=None,
            cutoff=None,
            x0=None,
            y0=None,
            dtBin=None,
            dEBin=None,
        )

    def test_exp_cpp(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        expected = np.exp(x)
        result = np.empty_like(x)

        butils_wrap_cpp.exp_cpp(x=x, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_interp_const_bin(self):
        # Piecewise constant interpolation between bin edges
        xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)  # doesnt need
        # more entries
        yp = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        x = np.array([0.50, 1.50, 2.50, 3.50], dtype=np.float64)
        expected = np.array([15.0, 25.0, np.nan, np.nan])  # Out-of-bounds

        result = np.full_like(x, fill_value=np.nan)
        butils_wrap_cpp.interp_const_bin(
            x=x, xp=xp, yp=yp, left=np.nan, right=np.nan, result=result
        )
        np.testing.assert_allclose(result, expected, atol=1e-12, equal_nan=True)

    def test_interp_const_space(self):
        # Constant interpolation based on space (step function behavior)
        xp = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        yp = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        x = np.array([-1.0, 0.5, 1.5, 2.5, 3.5], dtype=np.float64)
        expected = np.array(
            [np.nan, 10.0, 20.0, 30.0, np.nan]
        )  # with NaNs outside bounds

        result = np.full_like(x, fill_value=np.nan)
        butils_wrap_cpp.interp_const_space(
            x=x, xp=xp, yp=yp, left=np.nan, right=np.nan, result=result
        )
        np.testing.assert_allclose(result, expected, atol=1e-12, equal_nan=True)

    @unittest.skip("Activate FFTW compilation")
    def test_irfft(self):
        # Use a real FFT -> iFFT round trip to verify correctness
        time_data = np.array([1.0, 2.0, 1.0, 0.0], dtype=np.float64)
        freq_data = np.fft.rfft(time_data)
        n = time_data.size
        result = np.empty(n, dtype=np.float64)

        butils_wrap_cpp.irfft(a=freq_data, n=n, result=result)
        np.testing.assert_allclose(result, time_data, atol=1e-12)

    @unittest.skip("Activate FFTW compilation")
    def test_irfft_packed(self):
        # Packed = real-valued RFFT (half-spectrum) for a known signal
        original_signal = np.array([1.0, 2.0, 1.0, 0.0], dtype=np.float64)
        fft_data = np.fft.rfft(original_signal)
        fftsize = original_signal.size

        # Assuming the function takes packed frequency domain data and recovers time signal
        result = np.empty(fftsize, dtype=np.float64)
        butils_wrap_cpp.irfft_packed(signal=fft_data, fftsize=fftsize, result=result)

        np.testing.assert_allclose(result, original_signal, atol=1e-12)

    def test_linspace_cpp(self):
        start, stop, num = 0.0, 1.0, 5
        expected = np.linspace(start, stop, num, endpoint=True)
        result = np.empty(num, dtype=np.float64)

        # retstep is a placeholder output for the step size (ignored here)
        butils_wrap_cpp.linspace_cpp(
            start=start, stop=stop, num=num, retstep=None, result=result
        )
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_load_libblond(self):
        butils_wrap_cpp.load_libblond(precision="double")

    def test_mean_cpp(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        expected = np.mean(x)
        result = butils_wrap_cpp.mean_cpp(x=x)
        self.assertAlmostEqual(result, expected, places=12)

    def test_mul_cpp(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        expected = a * b
        result = np.empty_like(a)
        butils_wrap_cpp.mul_cpp(a=a, b=b, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @unittest.skip("Activate FFTW compilation")
    def test_rfft(self):
        signal = np.array([1.0, 2.0, 1.0, 0.0], dtype=np.float64)
        expected = np.fft.rfft(signal)
        result = np.empty_like(expected)
        butils_wrap_cpp.rfft(a=signal, n=signal.size, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @unittest.skip("Activate FFTW compilation")
    def test_rfftfreq(self):
        n = 8
        d = 0.1
        expected = np.fft.rfftfreq(n, d)
        result = np.empty(expected.shape, dtype=np.float64)
        butils_wrap_cpp.rfftfreq(n=n, d=d, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sin_cpp(self):
        x = np.linspace(0, 2 * np.pi, 5)
        expected = np.sin(x)
        result = np.empty_like(x)
        butils_wrap_cpp.sin_cpp(x=x, result=result)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sort_cpp(self):
        x = np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float64)
        expected_asc = np.sort(x)
        expected_desc = np.sort(x)[::-1]

        # Ascending
        result_asc = butils_wrap_cpp.sort_cpp(x=x.copy(), reverse=False)
        np.testing.assert_array_equal(result_asc, expected_asc)

        # Descending
        result_desc = butils_wrap_cpp.sort_cpp(x=x.copy(), reverse=True)
        np.testing.assert_array_equal(result_desc, expected_desc)

    def test_std_cpp(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        expected = np.std(x)
        result = butils_wrap_cpp.std_cpp(x=x)
        self.assertAlmostEqual(result, expected, places=12)

    @unittest.skip("TODO")
    def test_where_cpp(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        expected = np.where((x > -1) & (x < 2), x, 0)
        result = np.empty_like(x)
        butils_wrap_cpp.where_cpp(x=x, more_than=-1.0, less_than=2.0, result=result)
        np.testing.assert_array_equal(result, expected)

    def test_argmax_cpp(self):
        x = np.array([1, 3, 2, 5, 4], dtype=np.float64)
        result = butils_wrap_cpp.argmax_cpp(x)
        expected = np.argmax(x)
        self.assertEqual(result, expected)

    def test_argmin_cpp(self):
        x = np.array([1, 3, -2, 5, 4], dtype=np.float64)
        result = butils_wrap_cpp.argmin_cpp(x)
        expected = np.argmin(x)
        self.assertEqual(result, expected)

    def test_interp_cpp(self):
        xp = np.array([0, 1, 2, 3], dtype=np.float64)
        yp = np.array([0, 1, 4, 9], dtype=np.float64)
        x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        expected = np.interp(x, xp, yp)

        result = np.empty_like(x)
        butils_wrap_cpp.interp_cpp(
            x=x, xp=xp, yp=yp, left=None, right=None, result=result
        )
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @unittest.skip
    def test_kick(self):
        # TODO: implement test for `kick`
        butils_wrap_cpp.kick(
            dt=None,
            dE=None,
            voltage=None,
            omega_rf=None,
            phi_rf=None,
            charge=None,
            n_rf=None,
            acceleration_kick=None,
        )

    def test_random_normal(self):
        seed = 42
        size = 1000
        loc = 0.0
        scale = 1.0

        np.random.seed(seed)
        expected = np.random.normal(loc, scale, size)

        result = butils_wrap_cpp.random_normal(
            loc=loc, scale=scale, size=size, seed=seed
        )
        self.assertEqual(len(result), size)
        self.assertAlmostEqual(np.mean(result), np.mean(expected), delta=0.1)
        self.assertAlmostEqual(np.std(result), np.std(expected), delta=0.1)

    @unittest.skip("NotImplemented")
    def test_set_random_seed(self):
        # Check if the seed results in reproducible output
        butils_wrap_cpp.set_random_seed(123)
        r1 = butils_wrap_cpp.random_normal(loc=0, scale=1, size=5)
        butils_wrap_cpp.set_random_seed(123)
        r2 = butils_wrap_cpp.random_normal(loc=0, scale=1, size=5)
        np.testing.assert_array_equal(r1, r2)

    def test_sum_cpp(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = butils_wrap_cpp.sum_cpp(x)
        expected = np.sum(x)
        self.assertAlmostEqual(result, expected, places=12)

    def test_trapz_cpp(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        expected = np.trapezoid(y, x)
        result = butils_wrap_cpp.trapz_cpp(y=y, x=x, dx=None)

        self.assertAlmostEqual(result, expected, places=10)

    def test_trapz_cpp_dx(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        expected = np.trapezoid(y, x, dx=0.1)
        result = butils_wrap_cpp.trapz_cpp(y=y, x=x, dx=0.1)

        self.assertAlmostEqual(result, expected, places=10)


if __name__ == "__main__":
    unittest.main()
