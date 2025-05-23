import unittest

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from blond.utils.fft_helpers import irfftfreq, BufferedFftConvolve1D


class TestIrfftfreq(unittest.TestCase):
    def test_results_correct(self):
        for i in range(2, 12):
            ts_expected = np.arange(i)
            amps = np.random.rand(i)
            if i % 2 == 0:
                return_odd = False
            else:
                return_odd = True
            fs = np.fft.rfftfreq(len(amps), ts_expected[1] - ts_expected[0])
            ts_tested = irfftfreq(fs, return_odd)

            np.testing.assert_allclose(ts_tested, ts_expected)


class TestBufferedFft(unittest.TestCase):
    def test_BufferedFftConcolve1D(self):
        for i in range(2, 22):
            array1 = np.random.rand(21)
            array2 = np.random.rand(i)
            for mode in ("full", "valid", "same"):
                res_expected = fftconvolve(array1, array2, mode=mode)
                bfc = BufferedFftConvolve1D(array1, array2)
                res_tested = bfc.fftconvolve(array2, mode=mode)
                np.testing.assert_allclose(res_tested, res_expected)


if __name__ == "__main__":
    unittest.main()
