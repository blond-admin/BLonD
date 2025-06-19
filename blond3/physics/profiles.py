from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import Optional as LateInit

import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy.typing import NDArray as NumpyArray

from ..core.backend import backend
from ..core.base import BeamPhysicsRelevant
from ..core.beam.base import BeamBaseClass
from ..core.helpers import int_from_float_with_warning
from ..core.simulation.simulation import Simulation


class ProfileBaseClass(BeamPhysicsRelevant):
    def __init__(self):
        super().__init__()
        self._hist_x: LateInit[NumpyArray | CupyArray] = None
        self._hist_y: LateInit[NumpyArray | CupyArray] = None

        self._beam_spectrum_buffer = {}

    @property
    def hist_x(self):
        return self._hist_x

    @property
    def hist_y(self):
        return self._hist_y

    @cached_property
    def diff_hist_y(self):
        return backend.gradient(self._hist_y, self.hist_step)

    @cached_property
    def hist_step(self):
        return backend.float(self._hist_x[1] - self._hist_x[0])

    @cached_property
    def cut_left(self):
        return backend.float(self._hist_x[0] - self.hist_step / 2.0)

    @cached_property
    def cut_right(self):
        return backend.float(self._hist_x[-1] + self.hist_step / 2.0)

    @cached_property
    def bin_edges(self):
        return backend.linspace(
            self.cut_left, self.cut_right, len(self._hist_x) + 1, backend.float
        )

    def track(self, beam: BeamBaseClass):
        if beam.is_distributed:
            raise NotImplementedError("Impleemt hisogram on distributed array")
        else:
            backend.histogram(
                beam.read_partial_dt(), self.cut_left, self.cut_right, self._hist_y
            )
        self.invalidate_cache()

    def late_init(self, simulation: Simulation, **kwargs) -> None:
        assert self._hist_x is not None
        assert self._hist_y is not None
        self.invalidate_cache()

    @staticmethod
    def get_arrays(cut_left: float, cut_right: float, n_bins: int):
        step = (cut_right - cut_left) / n_bins
        offset = step / 2
        hist_x = backend.linspace(
            cut_left + offset, cut_right - offset, n_bins, dtype=backend.float
        )
        hist_y = backend.zeros(n_bins, dtype=backend.float)
        return hist_x, hist_y

    @property
    def cutoff_frequency(self):
        return backend.float(1 / (2 * self.hist_step))

    def _calc_gauss(self):
        return

    @cached_property
    def gauss_fit_params(self):
        return self._calc_gauss()

    @cached_property
    def beam_spectrum(self, n_fft: int):
        if n_fft in self._beam_spectrum_buffer.keys():
            self._beam_spectrum_buffer = np.fft.irfft(self._hist_y, n_fft)
        else:
            np.fft.irfft(self._hist_y, n_fft, out=self._beam_spectrum_buffer[n_fft])

        return self._beam_spectrum_buffer

    def invalidate_cache(self):
        for attribute in (
            "gauss_fit_params",
            "beam_spectrum",
            "hist_step",
            "cut_left",
            "cut_right",
            "bin_edges",
        ):
            self.__dict__.pop(attribute, None)


class StaticProfile(ProfileBaseClass):
    def __init__(self, cut_left: float, cut_right: float, n_bins: int):
        super().__init__()
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )

    @staticmethod
    def from_cutoff(cut_left: float, cut_right: float, cutoff_frequency: float):
        dt = 1 / (2 * cutoff_frequency)
        n_bins = int(math.ceil((cut_right - cut_left) / dt))
        return StaticProfile(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)

    @staticmethod
    def from_rad(
        cut_left_rad: float, cut_right_rad: float, n_bins: int, t_period: float
    ):
        rad_to_frac = 1 / (2 * np.pi)
        cut_left = cut_left_rad * rad_to_frac * t_period
        cut_right = cut_right_rad * rad_to_frac * t_period
        return StaticProfile(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)


class DynamicProfile(ProfileBaseClass):
    def __init__(self):
        super().__init__()

    def late_init(self, simulation: Simulation, **kwargs) -> None:
        self.update_attributes(beam=simulation.ring.beams[0])
        super().late_init(simulation=simulation)

    @abstractmethod
    def update_attributes(self, beam: BeamBaseClass) -> None:
        pass

    def track(self, beam: BeamBaseClass):
        self.update_attributes(beam=beam)
        super().track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    def __init__(self, timestep: float):
        super().__init__()
        self.timestep = timestep

    def update_attributes(self, beam: BeamBaseClass):
        cut_left = beam.dt_min()  # TODO caching of attribute acess
        cut_right = beam.dt_max()  # TODO caching of attribute acess
        n_bins = int(math.ceil((cut_right - cut_left) / self.timestep))
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )


class DynamicProfileConstNBins(ProfileBaseClass):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = int_from_float_with_warning(n_bins, warning_stacklevel=2)

    def update_attributes(self, beam: BeamBaseClass):
        cut_left = beam.dt_min()  # TODO caching of attribute acess
        cut_right = beam.dt_max()  # TODO caching of attribute acess
        self.hist_x, self.hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )
