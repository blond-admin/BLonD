from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from os import PathLike
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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


class InductiveImpedance(AnalyticWakeFieldSource, FreqDomain, TimeDomain):
    def __init__(self, Z_over_n: float):
        """
        Parameters
        ----------
        Z_over_n : float or array-like
            Constant imaginary Z/n = (Z * f /f0) impedance in [Ω].
            Can be a scalar or a turn-indexed array.
        """
        self.Z_over_n = Z_over_n

    def get_impedance(self, freq_x: NumpyArray, simulation: Simulation) -> NumpyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x : array-like
            Frequency components (harmonics).
        simulation : Simulation
            Simulation object containing turn index and RF info.

        Returns
        -------
        imp : array-like
            Complex impedance array.
        """
        T = (
            simulation.ring.circumference / simulation.beams[0].reference_velocity
        )  # FIXME consider update of this value!

        df = freq_x[1] - freq_x[0]  # frequency spacing
        n = 2 * (len(freq_x) - 1)  # original signal length (for irfft)
        dx = 1 / (n * df)
        h = dx
        k = 2 * np.pi * freq_x
        assert np.isclose(
            np.fft.rfftfreq(n, d=dx)[1] - np.fft.rfftfreq(n, d=dx)[0], df
        ), "Contact dev"  # TODO remove after testing

        # central finite difference (f(x+h) - g(x-h)) / 2h
        # expressed in frequency domain
        derivative = 1j * np.sin(k * h) / h

        return derivative / (2 * np.pi) * self.Z_over_n * T

    def get_wake_impedance(
        self, time: NumpyArray, simulation: Simulation
    ) -> NumpyArray:
        freq = np.fft.rfftfreq(len(time), d=time[1] - time[0])
        return self.get_impedance(freq_x=freq, simulation=simulation) / (
            time[1] - time[0]
        )


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    def __init__(
        self,
        shunt_impedances: NumpyArray,
        center_frequencies: NumpyArray,
        quality_factors: NumpyArray,
    ):
        self._shunt_impedances = shunt_impedances
        self._center_frequencies = center_frequencies
        self._quality_factors = quality_factors

        # Test if one or more quality factors is smaller than 0.5.
        if np.sum(self._quality_factors < 0.5) > 0:
            raise RuntimeError("All quality factors Q must be greater or equal 0.5")

    def get_wake_impedance(
        self, time: NumpyArray, simulation: Simulation
    ) -> NumpyArray:
        wake = np.zeros(len(time), dtype=backend.float, order="C")
        n_centers = len(self._center_frequencies)
        omega = 2 * np.pi * self._center_frequencies
        for i in range(n_centers):
            alpha = omega[i] / (2 * self._quality_factors[i])
            omega_bar = np.sqrt(omega[i] ** 2 - alpha**2)

            wake += (
                (np.sign(time) + 1)
                * (self._shunt_impedances[i] * alpha * np.exp(-alpha * time))
                * (
                    np.cos(omega_bar * time)
                    - alpha / omega_bar * np.sin(omega_bar * time)
                )
            )

        return np.fft.rfft(wake)

    def get_impedance(self, freq_x: NumpyArray, simulation: Simulation) -> NumpyArray:
        impedance = np.zeros(len(freq_x), dtype=complex)

        n_centers = len(self._center_frequencies)

        for i in range(n_centers):
            impedance[1:] += self._shunt_impedances[i] / (
                1
                + (
                    (1j * self._quality_factors[i])
                    * (
                        freq_x[1:] / self._center_frequencies[i]
                        - self._center_frequencies[i] / freq_x[1:]
                    )
                )
            )
        return impedance


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

    def get_impedance(self, freq_x: NumpyArray, simulation: Simulation) -> NumpyArray:
        if self.__at_freq_x is not None:
            if np.any(freq_x != self.__at_freq_x):
                # reset cache if new array
                self.__at_freq_x = freq_x
                self.__dict__.pop("_get_impedance", None)
        return self._get_impedance

    @cached_property
    def _get_impedance(self):
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
