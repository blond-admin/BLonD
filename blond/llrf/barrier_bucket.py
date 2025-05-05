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
