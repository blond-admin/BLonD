from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from ..._core.beam.base import BeamBaseClass


def get_hash(array1d: NumpyArray) -> int:
    return hash((array1d[0], array1d[1], array1d[-1], len(array1d)))


class InductiveImpedance(AnalyticWakeFieldSource, FreqDomain, TimeDomain):
    def __init__(self, Z_over_n: float):
        """
        Inductive impedance, i.e. only complex component in frequency domain


        Parameters
        ----------
        Z_over_n : float or array-like
            Constant imaginary Z/n = (Z * f /f0) impedance in [Ω].
            Can be a scalar or a turn-indexed array.
        """
        super().__init__(is_dynamic=True)
        self.Z_over_n = Z_over_n

        self._cache_derivative = None
        self._cache_derivative_hash = None

        self._cache_wake_impedance = None
        self._cache_wake_impedance_hash = None

    def get_impedance(
        self,
        freq_x: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Return the impedance in the frequency domain.

        Notes
        -----
        In algebraic math, multiplying in frequency domain with -1j Z f
        will correspond to the derivative in time domain.
        Unfortunately, with a truncated discrete signal and FFTs,
        this behaviour can not be reproduced by only multiplication with
        this term.
        Instead, multiplication by 1j * np.sin(k * h) / h must be done to
        achieve the same result (as np.gradient).

        np.gradient(x) in time domain
        ifft(derivative_kernel * fft(x)) in frequency domain


        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        impedance
            Complex impedance array.
        """
        T = simulation.ring.circumference / beam.reference_velocity
        z_over_n = self.Z_over_n
        derivative_kernel = self._get_derivative_impedance(freq_x)
        return derivative_kernel[:] / (2 * np.pi) * z_over_n * T

    def _get_derivative_impedance(self, freq_x: NumpyArray) -> NumpyArray:
        """Get the equivalent of np.gradient(x) in frequency domain ifft(
        derivative*fft(x))"""

        # Recalculate only of `freq_x` is changed
        hash_ = get_hash(freq_x)
        if hash_ is self._cache_derivative_hash:
            return self._cache_derivative

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

        self._cache_derivative_hash = hash_
        self._cache_derivative = derivative

        return derivative

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Get impedance equivalent to the partial wake in time domain


        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        wake_impedance

        """
        # Recalculate only of `time` is changed

        hash_ = get_hash(time)
        if hash_ is self._cache_wake_impedance_hash:
            return self._cache_wake_impedance
        freq = np.fft.rfftfreq(len(time), d=time[1] - time[0])
        wake_impedance = self.get_impedance(
            freq_x=freq, simulation=simulation, beam=beam
        ) / (time[1] - time[0])
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance

        return wake_impedance


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    def __init__(
        self,
        shunt_impedances: NumpyArray,
        center_frequencies: NumpyArray,
        quality_factors: NumpyArray,
    ):
        """
        Multiple resonances of RLC circuits for impedance calculations.

        Parameters
        ----------
        shunt_impedances : array-like
            Shunt impedances of the resonant circuits, in [Ω].
        center_frequencies : array-like
            Center frequencies of the resonances, in [Hz].
        quality_factors : array-like
            Quality factors (Q) of the resonances, dimensionless.

        Notes
        -----
        Ensure that all input arrays have the same length, with each entry
        corresponding to a separate resonance.
        """
        super().__init__(is_dynamic=False)
        assert len(shunt_impedances) == len(
            center_frequencies
        ), f"{len(shunt_impedances)} != {len(center_frequencies)}"
        assert len(shunt_impedances) == len(
            quality_factors
        ), f"{len(shunt_impedances)} != {len(quality_factors)}"
        self._shunt_impedances = shunt_impedances
        self._center_frequencies = center_frequencies
        self._quality_factors = quality_factors

        # Test if one or more quality factors is smaller than 0.5.
        if np.sum(self._quality_factors < 0.5) > 0:
            raise RuntimeError("All quality factors Q must be greater or equal 0.5")

        self._cache_wake_impedance = None
        self._cache_wake_impedance_hash = None

        self._cache_impedance = None
        self._cache_impedance_hash = None

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Get impedance equivalent to the partial single-particle-wake in
        time domain


        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        wake_impedance

        """
        # Recalculate only of `time` is changed
        hash = get_hash(time)
        if hash is self._cache_wake_impedance_hash:
            return self._cache_wake_impedance

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
        wake_impedance = np.fft.rfft(wake)

        self._cache_wake_impedance_hash = hash
        self._cache_wake_impedance = wake_impedance
        return wake_impedance

    def get_impedance(
        self,
        freq_x: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        impedance
            Complex impedance array.
        """
        # Recalculate only of `freq_x` is changed

        hash_ = get_hash(freq_x)
        if hash_ is self._cache_impedance_hash:
            return self._cache_impedance

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
        self._cache_impedance_hash = hash_
        self._cache_impedance = impedance
        return impedance


class ImpedanceTable(DiscreteWakeFieldSource):
    """
    Base class to manage impedance tables
    """

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
        """
        Impedance table in frequency domain

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        freq_y
            Complex amplitudes in frequency domain
        """
        super().__init__(is_dynamic=False)

        self._freq_x = freq_x
        self._freq_y = freq_y

        self._cache_impedance = None
        self._cache_impedance_hash = None

    def get_impedance(
        self,
        freq_x: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        impedance
            Complex impedance array.
        """
        # Recalculate only of `freq_x` is changed
        hash_ = get_hash(freq_x)
        if hash_ is self._cache_impedance_hash:
            return self._cache_impedance
        impedance = np.interp(
            freq_x, self._freq_x, self._freq_y, left=0, right=0
        ).astype(backend.complex)

        self._cache_impedance_hash = hash_
        self._cache_impedance = impedance

        return impedance

    @staticmethod
    def from_file(filepath: PathLike, reader: ImpedanceReader) -> ImpedanceTableFreq:
        """
        Instance table from a file on the disk

        Parameters
        ----------
        filepath
            path of the file to lead
        reader
            `ImpedanceReader` to interpret what's written in the file

        Returns
        -------
        impedance_table_freq

        """
        x_array, y_array = reader.load_file(filepath=filepath)
        assert not np.any(np.isnan(x_array))
        assert not np.any(np.isnan(y_array))
        return ImpedanceTableFreq(freq_x=x_array, freq_y=y_array)


class ImpedanceTableTime(ImpedanceTable, TimeDomain):
    def __init__(
        self,
        wake_x: NumpyArray,
        wake_y: NumpyArray,
    ):
        """
        Impedance table in frequency domain

        Parameters
        ----------
        wake_x
            Wake time axis, in [s]
        wake_y
            Wake amplitude, in [V]
        """
        super().__init__(is_dynamic=False)
        self._wake_x = wake_x
        self._wake_y = wake_y

        self._cache_wake_impedance = None
        self._cache_wake_impedance_hash = None

    @staticmethod
    def from_file(
        filepath: PathLike | str, reader: ImpedanceReader
    ) -> ImpedanceTableTime:
        """
        Instance table from a file on the disk

        Parameters
        ----------
        filepath
            path of the file to lead
        reader
            `ImpedanceReader` to interpret what's written in the file

        Returns
        -------
        impedance_table_time

        """
        x_array, y_array = reader.load_file(filepath=filepath)
        return ImpedanceTableTime(wake_x=x_array, wake_y=y_array)

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Get impedance equivalent to the partial single-particle-wake in
        time domain


        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object

        Returns
        -------
        wake_impedance

        """
        hash_ = get_hash(time)
        if hash_ is self._cache_wake_impedance_hash:
            return self._cache_wake_impedance

        wake = np.interp(time, self._wake_x, self._wake_y)
        wake_impedance = np.fft.rfft(wake)
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance
        return wake_impedance
