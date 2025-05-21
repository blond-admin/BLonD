from __future__ import annotations

# General imports
import numpy as np
import abc
from typing import TYPE_CHECKING

try:
    import cupy as cp
except ImportError:
    _CUPY_AVAILABLE = False
else:
    _CUPY_AVAILABLE = True

# BLonD imports
from ..utils import bmath as bm


if TYPE_CHECKING:
    from typing import Iterable

    from numpy.typing import NDArray

    from blond.beam.beam import Beam


class BarrierBucket:

    def __init__(self, t_center: float | Iterable[float],
                 t_width: float | Iterable[float],
                 peak: float | Iterable[float],
                 beam: Beam, bin_centers: NDArray):

        self.t_center = t_center
        self._cent_is_iter = hasattr(t_center, "__iter__")
        self.t_width = t_width
        self._width_is_iter = hasattr(t_width, "__iter__")
        self.peak = peak
        self._peak_is_iter = hasattr(peak, "__iter__")

        self._beam = beam
        self._bin_centers = bin_centers.copy()
        self._barrier_waveform = bm.zeros_like(self._bin_centers)

    def compute_barrier(self, turn: int):

        if self._cent_is_iter:
            cent = self.t_center[turn]
        else:
            cent = self.t_center

        if self._width_is_iter:
            width = self.t_width[turn]
        else:
            width = self.t_width

        if self._peak_is_iter:
            peak = self.peak[turn]
        else:
            peak = self.peak

        self._barrier_waveform *= 0

        low_bin = bm.where(self._bin_centers >= cent-width/2)[0][0]
        high_bin = bm.where(self._bin_centers >= cent+width/2)[0][0]

        b_time = self._bin_centers[low_bin:high_bin] - self._bin_centers[low_bin]

        barrier = peak * bm.sin(2*np.pi * b_time/self.t_width)

        self._barrier_waveform[low_bin:high_bin] = barrier


    def track(self):

        self._beam.dE += bm.interp(self._beam.dt, self._bin_centers,
                                   self._barrier_waveform)

    def to_gpu(self, recursive: bool = True):

        if not _CUPY_AVAILABLE:
            raise RuntimeError("Cannot send to gpu, cupy not available")

        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return

        if recursive:
            self._beam.to_gpu()

        self._bin_centers = cp.array(self._bin_centers)
        self._barrier_waveform = cp.array(self._barrier_waveform)

        self._device = 'GPU'

    def to_cpu(self, recursive: bool = True):

        if not _CUPY_AVAILABLE:
            raise RuntimeError("Cannot send to gpu, cupy not available")

        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':
            return

        if recursive:
            self._beam.to_cpu()

        self._bin_centers = cp.asnumpy(self._bin_centers)
        self._barrier_waveform = cp.asnumpy(self._barrier_waveform)

        self._device = 'CPU'

def barrier_to_harmonics(waveform: NDArray, harmonics: Iterable[int])\
                                -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Converts an arbitrary waveform to a fourier series in amplitude and
    phase.  Waveform is assumed to be 1 revolution period in length.
    the harmonic numbers must be an integer and are used to select the
    required fourier components.

    The input waveform can be reconstructed with a sin function.

    Returns:
        tuple[tuple[float, ...]]:
            Two tuples of float, length equal to len(harmonics).
            Element 0 is the amplitudes, element 1 is the phases.
    """

    wave_fft = np.fft.rfft(waveform)

    harm_series = np.array([wave_fft[h] for h in harmonics])

    harm_amps = np.abs(harm_series)/(len(waveform)/2)
    harm_phases = np.arctan2(harm_series.real, harm_series.imag) + np.pi

    return harm_amps, harm_phases

def sinc_filtering(harmonic_amplitudes: Iterable[float], m: int=1) -> NDArray:
    """
    Filters the fourier components with a sinc function window as
    described in PhD thesis:
        Beam Loss Reduction by Barrier Buckets in the CERN Accelerator
        Complex:  M. Vadai CERN-THESIS-2021-043 Chapter 3.2.3.2

    Args:
        harmonic_amplitudes (Iterable[float]):
            The amplitudes of the fourier series.  Assumed to be
            uniformly spaced in the range 1..n
        m (int, optional):
            Power applied to the sinc function.  Higher values give more
            aggressive filtering, 0 is equivalent to a square window, or
            no filtering.
            Defaults to 1.

    Returns:
        NDArray: The modified harmonic amplitudes.
    """

    filtered_amplitudes = np.zeros_like(harmonic_amplitudes)
    n_harm = len(harmonic_amplitudes)

    for i, a in enumerate(harmonic_amplitudes):

        filtered_amplitudes[i] = a * np.sinc((i*np.pi) / (2 * (n_harm+1)))**m

    return filtered_amplitudes
