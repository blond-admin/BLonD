from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from os import PathLike

import numpy as np
from numpy.typing import NDArray as NumpyArray

from blond3.physics.impedances.base import (
    AnalyticWakeFieldSource,
    FreqDomain,
    TimeDomain,
    DiscreteWakeFieldSource,
)
from blond3.physics.impedances.readers import ImpedanceReader, ExampleImpedanceReader1


class InductiveImpedance(AnalyticWakeFieldSource, FreqDomain):
    def get_freq_y(self, freq_x: NumpyArray):
        imp = np.zeros(len(freq_x), dtype=complex)
        imp[:] = 1j * self.Z_over_n * ring.f_rev  # TODO

    def __init__(self, Z_over_n: float):
        self.Z_over_n = Z_over_n


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    pass


class ImpedanceTable(DiscreteWakeFieldSource):
    @abstractmethod
    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader) -> ImpedanceTable:
        reader.load_file(filepath=filepath)


@dataclass(frozen=True)
class ImpedanceTableFreq(ImpedanceTable, FreqDomain):
    freq_x: NumpyArray
    freq_y: NumpyArray

    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader) -> ImpedanceTableFreq:
        x_array, y_array = reader.load_file(filepath=filepath)
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
