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
    from numpy.typing import NDArray

    from blond.beam.beam import Beam


class BarrierBucket:

    def __init__(self, t_center: float, t_width: float, peak: float,
                 beam: Beam, bin_centers: NDArray):

        self.t_center = t_center
        self.t_width = t_width
        self.peak = peak

        self._beam = beam
        self._bin_centers = bin_centers.copy()
        self._barrier_waveform = bm.zeros_like(self._bin_centers)

    def compute_barrier(self):

        self._barrier_waveform *= 0

        low_bin = bm.where(self._bin_centers
                                        >= self.t_center-self.t_width/2)[0][0]
        high_bin = bm.where(self._bin_centers
                                        >= self.t_center+self.t_width/2)[0][0]

        b_time = self._bin_centers[low_bin:high_bin] - self._bin_centers[low_bin]

        barrier = self.peak * bm.sin(2*np.pi * b_time/self.t_width)

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
