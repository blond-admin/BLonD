from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.signal._signaltools import _apply_conv_mode


def irfftfreq(freqs: NDArray, return_odd: bool):
    """Opposite of np.fft.rfftfreq

    Parameters
    ----------
    freqs
        The frequency axis
    return_odd
        If True, returns a time array of odd length
        else even length
    """

    n_in = len(freqs)
    f_max = freqs[-1]
    if return_odd:
        n_out = (n_in - 1) * 2 + 1
        dt = (n_out - 1) / (2 * f_max * n_out)
    else:
        n_out = (n_in - 1) * 2
        dt = 1 / (2 * f_max)

    return dt * np.arange(n_out)


class BufferedFftConvolve1D:
    """Twin method of `scipy.signal.fftconvolve`

    This method precalculates one fft for better runtime
    so `fftconvolve` uses only two instead of three fft calls"""

    def __init__(self, array1: NDArray, array2: NDArray):
        assert len(array1.shape) == 1, "Only 1D arrays allowed!"
        assert len(array2.shape) == 1, "Only 1D arrays allowed!"
        self.size1 = len(array1)
        self.size2 = len(array2)
        assert (
            self.size1 >= self.size2
        ), "`array1` must be bigger or same size than `array2`"
        self.shape = self.size1 + self.size2 - 1
        self.shape_fast = next_fast_len(self.shape)
        self.i_array1 = np.fft.rfft(array1, self.shape_fast, axis=0)

    def fftconvolve(
        self, array2: NDArray, mode: Literal["full", "valid", "same"] = "full"
    ):
        """Calculates the convolution of (buffered) array1 with array2"""
        assert len(array2) == self.size2, (
            f"{len(array2)=}, but must be " f"{self.size2}"
        )
        i_array2 = np.fft.rfft(array2, self.shape_fast, axis=0)
        ret = np.fft.irfft(i_array2 * self.i_array1, self.shape_fast, axis=0)[
            : self.shape
        ]
        return _apply_conv_mode(
            ret,
            (max(self.size1, self.size2),),
            (min(self.size1, self.size2),),
            mode,
            [0],
        )
