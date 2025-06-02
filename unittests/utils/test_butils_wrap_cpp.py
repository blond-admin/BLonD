import unittest

from blond.utils import butils_wrap_cpp
from blond.utils.butils_wrap_cpp import (argmax_cpp, argmin_cpp, interp_cpp,
                                      kick, \
    random_normal, set_random_seed, sum_cpp, trapz_cpp)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_add_cpp(self):
        # TODO: implement test for `add_cpp`
        butils_wrap_cpp.add_cpp(a=None, b=None, result=None, inplace=None)

    @unittest.skip
    def test_arange_cpp(self):
        # TODO: implement test for `arange_cpp`
        butils_wrap_cpp.arange_cpp(
            start=None, stop=None, step=None, dtype=None, result=None
        )

    @unittest.skip
    def test_cos_cpp(self):
        # TODO: implement test for `cos_cpp`
        butils_wrap_cpp.cos_cpp(x=None, result=None)

    @unittest.skip
    def test_cumtrapz(self):
        # TODO: implement test for `cumtrapz`
        butils_wrap_cpp.cumtrapz(y=None, x=None, dx=None, initial=None, result=None)

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

    @unittest.skip
    def test_exp_cpp(self):
        # TODO: implement test for `exp_cpp`
        butils_wrap_cpp.exp_cpp(x=None, result=None)

    @unittest.skip
    def test_interp_const_bin(self):
        # TODO: implement test for `interp_const_bin`
        butils_wrap_cpp.interp_const_bin(
            x=None, xp=None, yp=None, left=None, right=None, result=None
        )

    @unittest.skip
    def test_interp_const_space(self):
        # TODO: implement test for `interp_const_space`
        butils_wrap_cpp.interp_const_space(
            x=None, xp=None, yp=None, left=None, right=None, result=None
        )

    @unittest.skip
    def test_irfft(self):
        # TODO: implement test for `irfft`
        butils_wrap_cpp.irfft(a=None, n=None, result=None)

    @unittest.skip
    def test_irfft_packed(self):
        # TODO: implement test for `irfft_packed`
        butils_wrap_cpp.irfft_packed(signal=None, fftsize=None, result=None)

    @unittest.skip
    def test_linspace_cpp(self):
        # TODO: implement test for `linspace_cpp`
        butils_wrap_cpp.linspace_cpp(
            start=None, stop=None, num=None, retstep=None, result=None
        )

    @unittest.skip
    def test_load_libblond(self):
        # TODO: implement test for `load_libblond`
        butils_wrap_cpp.load_libblond(precision=None)

    @unittest.skip
    def test_mean_cpp(self):
        # TODO: implement test for `mean_cpp`
        butils_wrap_cpp.mean_cpp(x=None)

    @unittest.skip
    def test_mul_cpp(self):
        # TODO: implement test for `mul_cpp`
        butils_wrap_cpp.mul_cpp(a=None, b=None, result=None)

    @unittest.skip
    def test_rfft(self):
        # TODO: implement test for `rfft`
        butils_wrap_cpp.rfft(a=None, n=None, result=None)

    @unittest.skip
    def test_rfftfreq(self):
        # TODO: implement test for `rfftfreq`
        butils_wrap_cpp.rfftfreq(n=None, d=None, result=None)

    @unittest.skip
    def test_sin_cpp(self):
        # TODO: implement test for `sin_cpp`
        butils_wrap_cpp.sin_cpp(x=None, result=None)

    @unittest.skip
    def test_sort_cpp(self):
        # TODO: implement test for `sort_cpp`
        butils_wrap_cpp.sort_cpp(x=None, reverse=None)

    @unittest.skip
    def test_std_cpp(self):
        # TODO: implement test for `std_cpp`
        butils_wrap_cpp.std_cpp(x=None)

    @unittest.skip
    def test_where_cpp(self):
        # TODO: implement test for `where_cpp`
        butils_wrap_cpp.where_cpp(x=None, more_than=None, less_than=None, result=None)

    @unittest.skip
    def test_argmax_cpp(self):
        # TODO: implement test for `argmax_cpp`
        argmax_cpp(x=None)

    @unittest.skip
    def test_argmin_cpp(self):
        # TODO: implement test for `argmin_cpp`
        argmin_cpp(x=None)

    @unittest.skip
    def test_interp_cpp(self):
        # TODO: implement test for `interp_cpp`
        interp_cpp(x=None, xp=None, yp=None, left=None, right=None, result=None)

    @unittest.skip
    def test_kick(self):
        # TODO: implement test for `kick`
        kick(dt=None, dE=None, voltage=None, omega_rf=None, phi_rf=None, charge=None, n_rf=None, acceleration_kick=None)

    @unittest.skip
    def test_random_normal(self):
        # TODO: implement test for `random_normal`
        random_normal(loc=None, scale=None, size=None, seed=None)

    @unittest.skip
    def test_set_random_seed(self):
        # TODO: implement test for `set_random_seed`
        set_random_seed(seed=None)

    @unittest.skip
    def test_sum_cpp(self):
        # TODO: implement test for `sum_cpp`
        sum_cpp(x=None)

    @unittest.skip
    def test_trapz_cpp(self):
        # TODO: implement test for `trapz_cpp`
        trapz_cpp(y=None, x=None, dx=None)
