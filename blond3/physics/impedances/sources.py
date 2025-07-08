from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Optional

import numpy as np
from cryptography.utils import cached_property
from numpy.typing import NDArray as NumpyArray

from .readers import ImpedanceReader
from ..impedances.base import (
    AnalyticWakeFieldSource,
    FreqDomain,
    TimeDomain,
    DiscreteWakeFieldSource,
)
from ... import Simulation
from ..._core.backends.backend import backend


class InductiveImpedance(AnalyticWakeFieldSource, FreqDomain):
    def get_freq_y(self, freq_x: NumpyArray, sim: Simulation) -> NumpyArray:
        imp = np.zeros(len(freq_x), dtype=backend.complex)
        imp[:] = (
            1j
            * self.Z_over_n
            * sim.energy_cycle.f_rev[
                0 if sim.turn_i.value is None else sim.turn_i.value
            ]
        )
        return imp

    def __init__(self, Z_over_n: float):
        self.Z_over_n = Z_over_n


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    pass


class ImpedanceTable(DiscreteWakeFieldSource):
    @staticmethod
    @abstractmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader) -> ImpedanceTable:
        pass


class ImpedanceTableFreq(ImpedanceTable, FreqDomain):
    def __init__(
        self,
        freq_x: NumpyArray,
        freq_y: NumpyArray,
    ):
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.__at_freq_x: Optional[NumpyArray] = field(
            default=None, init=False, repr=False
        )

    def get_freq_y(self, freq_x: NumpyArray, sim: Simulation) -> NumpyArray:
        if self.__at_freq_x is not None:
            if np.any(freq_x != self.__at_freq_x):
                # reset cache if new array
                self.__at_freq_x = freq_x
                self.__dict__.pop("_get_freq_y", None)
        return self._get_freq_y

    @cached_property
    def _get_freq_y(self):
        return np.interp(
            self.__at_freq_x, self.freq_x, self.freq_y, left=0, right=0
        ).astype(backend.complex)

    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader) -> ImpedanceTableFreq:
        x_array, y_array = reader.load_file(filepath=filepath)
        assert not np.any(np.isnan(x_array))
        assert not np.any(np.isnan(y_array))
        return ImpedanceTableFreq(freq_x=x_array, freq_y=y_array)


@dataclass(frozen=True)
class ImpedanceTableTime(ImpedanceTable, TimeDomain):
    wake_x: NumpyArray
    wake_y: NumpyArray

    @staticmethod
    def from_file(
        filepath: PathLike | str, reader: ImpedanceReader
    ) -> ImpedanceTableTime:
        x_array, y_array = reader.load_file(filepath=filepath)
        return ImpedanceTableTime(wake_x=x_array, wake_y=y_array)
