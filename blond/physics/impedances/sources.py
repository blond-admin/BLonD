from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ... import Simulation
from ..._core.backends.backend import backend
from ..impedances.base import (
    AnalyticWakeFieldSource,
    DiscreteWakeFieldSource,
    FreqDomain,
    TimeDomain,
)
from .readers import ImpedanceReader

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

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
        n_fft: int,
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
        freq = np.fft.rfftfreq(n_fft, d=time[1] - time[0])
        wake_impedance = self.get_impedance(
            freq_x=freq, simulation=simulation, beam=beam
        ) / (time[1] - time[0])
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance

        return wake_impedance


class Resonators(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    def __init__(
        self,
        shunt_impedances: NumpyArray | float,
        center_frequencies: NumpyArray | float,
        quality_factors: NumpyArray | float,
    ):
        """
        Multiple resonances of RLC circuits for impedance calculations.

        Parameters
        ----------
        shunt_impedances : array-like or float
            Shunt impedances of the resonant circuits, in [Ω].
        center_frequencies : array-like or float
            Center frequencies of the resonances, in [Hz].
        quality_factors : array-like or float
            Quality factors (Q) of the resonances, dimensionless.

        all values must be float, if one is given as float

        Notes
        -----
        Ensure that all input arrays have the same length, with each entry
        corresponding to a separate resonance.
        """
        super().__init__(is_dynamic=False)
        if (
            type(shunt_impedances) == float
            and type(center_frequencies) == float
            and type(quality_factors) == float
        ):
            self._shunt_impedances = np.array([shunt_impedances])
            self._center_frequencies = np.array([center_frequencies])
            self._quality_factors = np.array([quality_factors])
            self._n_resonators = len(self._shunt_impedances)
        else:
            assert len(shunt_impedances) == len(center_frequencies), (
                f"{len(shunt_impedances)} != {len(center_frequencies)}"
            )
            assert len(shunt_impedances) == len(quality_factors), (
                f"{len(shunt_impedances)} != {len(quality_factors)}"
            )
            self._shunt_impedances = shunt_impedances
            self._center_frequencies = center_frequencies
            self._quality_factors = quality_factors
            self._n_resonators = len(shunt_impedances)

        # secondary quantities for wake calculation
        self._omega = 2 * np.pi * self._center_frequencies
        self._alpha = self._omega / (2 * self._quality_factors)
        self._omega_bar = np.sqrt(self._omega**2 - self._alpha**2)

        # Test if one or more quality factors is smaller than 0.5.
        if np.sum(self._quality_factors < 0.5) > 0:
            raise RuntimeError(
                "All quality factors Q must be greater or equal 0.5"
            )
        if np.sum(self._center_frequencies < 0) > 0:
            raise RuntimeError(
                "All center frequencies must be greater or equal 0"
            )

        self._cache_wake_impedance = None
        self._cache_wake_impedance_hash = None

        self._cache_impedance = None
        self._cache_impedance_hash = None

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray:
        """
        Get impedance computed via fft from time domain anayltical formula equivalent
        to the partial single-particle-wake


        Parameters
        ----------
        time
            Time array to get wake, in [s]
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation beam object
        n_fft
            Number of fft bins to use

        Returns
        -------
        wake_impedance

        """
        # Recalculate only if `time` has changed
        hash = get_hash(time)
        if hash is self._cache_wake_impedance_hash:
            return self._cache_wake_impedance

        wake = self.get_wake(time)
        wake_impedance = np.fft.rfft(wake, n=n_fft)

        self._cache_wake_impedance_hash = hash
        self._cache_wake_impedance = wake_impedance
        return wake_impedance

    def get_wake_impedance_freq(self, time):
        """
        Get frequency array corresponding to time used in :func:`get_wake_impedance`
        """
        return np.fft.rfftfreq(
            len(self._cache_wake_impedance), time[1] - time[0]
        )

    def get_wake(self, time: NumpyArray) -> NumpyArray:
        """
        Computes the wake function of all resonators in time domain for the given time and returns the summed potential.

        Parameters
        ----------
        time : NumpyArray
            time array at which the wake is calculated [V]
        """
        wake = np.zeros(len(time), dtype=backend.float, order="C")

        for res_ind in range(self._n_resonators):
            wake += (
                (np.sign(time) + 1)  # heaviside
                * (
                    self._shunt_impedances[res_ind]
                    * self._alpha[res_ind]
                    * np.exp(-self._alpha[res_ind] * time)
                )
                * (
                    np.cos(self._omega_bar[res_ind] * time)
                    - self._alpha[res_ind]
                    / self._omega_bar[res_ind]
                    * np.sin(self._omega_bar[res_ind] * time)
                )
            )
        return wake

    def calculate_envelope(self) -> Tuple[NumpyArray, NumpyArray]:
        """
        Calculates the normalized envelope of all resonators.
        """
        time_axis = np.linspace(
            0, np.max(self._quality_factors / self._omega) * 20, 100000
        )
        envelope = np.zeros_like(time_axis)
        for res_ind in range(len(self._quality_factors)):
            envelope += (
                self._shunt_impedances[res_ind]
                * self._alpha[res_ind]
                * np.exp(-time_axis * self._alpha[res_ind])
            )
        envelope /= np.max(envelope)

        return time_axis, envelope

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
        # Recalculate only if `freq_x` is changed

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
    @abstractmethod  # pragma: no cover
    def from_file(
        filepath: PathLike, reader: ImpedanceReader
    ) -> ImpedanceTable:
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
    def from_file(
        filepath: PathLike, reader: ImpedanceReader
    ) -> ImpedanceTableFreq:
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
        n_fft: int,
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
        wake_impedance = np.fft.rfft(wake, n=n_fft)
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance
        return wake_impedance


# TODO rework docstring
class TravelingWaveCavity(AnalyticWakeFieldSource, TimeDomain, FreqDomain):
    r"""Impedance of travelling wave cavities.
    Notes
    -----
    Impedance contribution from travelling wave cavities,
    analytic formulas for both wake and impedance. The resonance modes (and
    the corresponding R and a) can be inputed as a list in case of several
    modes.

    The model is the following:

    .. math::
        Z &= Z_+ + Z_- \\
        Z_-(f) &= R \left[\left(\frac{\sin{\frac{a(f-f_r)}{2}}}{\frac{a(f-f_r)}{2}}\right)^2 - 2i \frac{a(f-f_r) - \sin{a(f-f_r)}}{\left(a(f-f_r)\right)^2}\right] \\
        Z_+(f) &= R \left[\left(\frac{\sin{\frac{a(f+f_r)}{2}}}{\frac{a(f+f_r)}{2}}\right)^2 - 2i \frac{a(f+f_r) - \sin{a(f+f_r)}}{\left(a(f+f_r)\right)^2}\right]

    .. math::
        W(0<t<\tilde{a}) &= \frac{4R}{\tilde{a}}\left(1-\frac{t}{\tilde{a}}\right)\cos{\omega_r t} \\
        W(0) &= \frac{2R}{\tilde{a}}

    .. math::
        a = 2 \pi \tilde{a}

    Parameters
    ----------
    R_S
        Shunt impepdance, in [Ω]
    frequency_R
        Resonant frequency, in [Hz]
    a_factor
        Damping time `a`, in [s]

    Attributes
    ----------
    R_S
        Shunt impepdance, in [Ω]
    frequency_R
        Resonant frequency, in [Hz]
    a_factor
        Damping time `a`, in [s]

    Examples
    ----------
    >>> R_S = [1, 2, 3]
    >>> frequency_R = [1, 2, 3]
    >>> a_factor = [1, 2, 3]
    >>> twc = TravelingWaveCavity(R_S, frequency_R, a_factor)
    >>> time = np.array(1,2,3)
    >>> twc.wake_calc(time)
    >>> frequency = np.array(1,2,3)
    >>> twc.imped_calc(frequency)
    """

    def __init__(
        self,
        R_S: float | NumpyArray,
        frequency_R: float | NumpyArray,
        a_factor: float | NumpyArray,
    ):
        if isinstance(R_S, np.ndarray):
            assert len(R_S) == len(frequency_R), (
                f"{len(R_S)=}, but {len(frequency_R)=}."
            )
            assert len(R_S) == len(a_factor), (
                f"{len(R_S)=}, but {len(a_factor)=}."
            )
        super().__init__(is_dynamic=False)

        # Shunt impedance in :math:`\Omega`
        self.R_S = np.array([R_S], dtype=float).flatten()

        # Resonant frequency in Hz
        self.frequency_R = np.array([frequency_R], dtype=float).flatten()

        # Damping time a in s
        self.a_factor = np.array([a_factor], dtype=float).flatten()

    def wake_calc(self, time: NumpyArray) -> NumpyArray:
        r"""
        Wake calculation method as a function of time.

        Parameters
        ----------
        time
            Time array to get wake, in [s]
        """

        wake = np.zeros(time.shape, dtype=backend.float, order="C")

        for i in range(0, len(self.R_S)):
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = time <= a_tilde
            wake[indexes] += (
                (np.sign(time[indexes]) + 1)
                * 2
                * self.R_S[i]
                / a_tilde
                * (1 - time[indexes] / a_tilde)
                * np.cos(2 * np.pi * self.frequency_R[i] * time[indexes])
            )
        return wake

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
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
        wake = self.wake_calc(time=time)
        wake_impedance = np.fft.rfft(wake, n=n_fft)
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
        impedance = np.zeros(len(freq_x), dtype=backend.complex, order="C")

        for i in range(0, self.R_S):
            xs_plus = self.a_factor[i] * (freq_x - self.frequency_R[i])
            xs_minus = self.a_factor[i] * (freq_x + self.frequency_R[i])

            Zplus = self.R_S[i] * (
                (np.sin(xs_plus / 2) / xs_plus / 2) ** 2
                - 2j * (xs_plus - np.sin(xs_plus)) / (xs_plus * xs_plus)
            )

            Zminus = self.R_S[i] * (
                (np.sin(xs_minus / 2) / xs_minus / 2) ** 2
                - 2j * (xs_minus - np.sin(xs_minus)) / (xs_minus * xs_minus)
            )

            impedance += Zplus + Zminus
        return impedance
