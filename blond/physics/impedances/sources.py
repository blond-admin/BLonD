# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementations of beam impedance sources.

Module to describe classes for the calculation of wakes and impedances.

Notes
-----
Authors:
Alexandre Lasheen
Danilo Quartullo
Juan F. Esteban Mueller,
Markus Schwarz
Simon Lauber
"""

from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.core.simulation.simulation import Simulation
from blond.generals.formatting_ import si_format
from blond.generals.warnings_ import NotTestedWarning
from blond.physics.impedances.base import (
    FreqDomain,
    TimeDomain,
    TimeDomainCounterRotation,
    WakeFieldSource,
)
from blond.physics.impedances.readers import ImpedanceReader

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass


def get_hash(array1d: NumpyArray | CupyArray) -> int:
    """
    Compute a lightweight, approximate hash value for a 1D NumPy array.

    The function samples a few representative elements of the input array
    (first, second, middle, and last), along with the array length, and computes
    a Python built-in hash from this tuple. The result is intended for quick,
    approximate identification of arrays rather than exact equality or integrity
    verification.

    Parameters
    ----------
    array1d : numpy.ndarray
        One-dimensional NumPy array of numeric values.

    Returns
    -------
    int
        An integer hash value derived from selected elements of the array.

    Warnings
    --------
    - This function is **not collision-resistant**. Different arrays may yield
      identical hash values, especially if they share similar boundary values or
      lengths.
    - Not suitable for **data integrity**, **deduplication**, or **security**
      purposes. Use `hashlib` (e.g., SHA-256) for robust, deterministic hashing.
    - Assumes a 1D numeric array; no validation is performed. Multi-dimensional
      or non-numeric inputs may cause unexpected behavior or errors.
    - Python’s built-in hash is **not stable across sessions** due to hash
      randomization (unless `PYTHONHASHSEED` is fixed).

    Notes
    -----
    - **Time complexity:** O(1) — the function samples only four elements
      regardless of array size.
    - **Memory usage:** O(1) — constant space overhead.
    - Designed for fast, approximate fingerprinting in performance-sensitive
      contexts where occasional collisions are acceptable.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> get_hash(arr)
    2398472938472938  # Example output (varies by session)
    """
    len_ = len(array1d)
    return hash(
        (
            float(array1d[0]),
            float(array1d[1]),
            float(array1d[int(len_ // 2)]),
            float(array1d[-1]),
            len_,
        )
    )


class InductiveImpedance(WakeFieldSource, FreqDomain, TimeDomain):
    """
    Inductive impedance, i.e. only complex component in frequency domain.

    Parameters
    ----------
    Z_over_n : float or array-like
        Constant imaginary Z/n = (Z * f /f0) impedance in [Ω].
        Can be a scalar or a turn-indexed array.
    """

    def __init__(self, Z_over_n: float):
        super().__init__(is_dynamic=True)
        self.Z_over_n = Z_over_n

        self._cache_derivative: NumpyArray | CupyArray | None = None
        self._cache_derivative_hash: int | None = None

        self._cache_wake_impedance: NumpyArray | CupyArray | None = None
        self._cache_wake_impedance_hash: int | None = None

    def get_impedance(
        self,
        freq_x: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray | CupyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.

        Returns
        -------
        impedance
            Complex impedance array.

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
        """
        T = simulation.ring.circumference / beam.reference.velocity
        z_over_n = self.Z_over_n
        derivative_kernel = self._get_derivative_impedance(freq_x)
        return derivative_kernel[:] / (2 * np.pi) * z_over_n * T

    def _get_derivative_impedance(
        self, freq_x: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Get the equivalent of np.gradient(x) in frequency domain ifft(derivative*fft(x)).

        Parameters
        ----------
        freq_x
            Frequency axis.

        Returns
        -------
        derivative
            Derivative impedance in frequency domain.
        """
        # Recalculate only if `freq_x` is changed
        hash_ = get_hash(freq_x)
        if hash_ == self._cache_derivative_hash:
            return self._cache_derivative

        df = float(freq_x[1] - freq_x[0])  # frequency spacing
        n = 2 * (len(freq_x) - 1)  # original signal length (for irfft)
        dx = 1 / (n * df)
        h = dx
        k = 2 * np.pi * freq_x
        assert np.isclose(
            np.fft.rfftfreq(n, d=dx)[1] - np.fft.rfftfreq(n, d=dx)[0], df
        ), "Contact dev"  # TODO remove after testing
        # central finite difference (f(x+h) - g(x-h)) / 2h
        # expressed in frequency domain
        derivative = 1j * backend.sin(k * h) / h

        self._cache_derivative_hash = hash_
        self._cache_derivative = derivative

        return derivative

    def get_wake_impedance(
        self,
        time: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray | CupyArray:
        """
        Get impedance equivalent to the partial wake in time domain.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of FFT points.

        Returns
        -------
        wake_impedance
            Wake impedance.
        """
        # Recalculate only if `time` is changed

        hash_ = get_hash(time)
        if hash_ == self._cache_wake_impedance_hash:
            return self._cache_wake_impedance
        freq = backend.fft.rfftfreq(n_fft, d=time[1] - time[0])
        wake_impedance = self.get_impedance(
            freq_x=freq,
            simulation=simulation,
            beam=beam,
        ) / (time[1] - time[0])
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance

        return wake_impedance


class Resonators(
    WakeFieldSource, TimeDomain, FreqDomain, TimeDomainCounterRotation
):
    r"""
    Multiple resonances of RLC circuits for impedance calculations.

    Parameters
    ----------
    shunt_impedances : array-like or float
        Shunt impedances of the resonant circuits, in [:math:`\omega`].
    center_frequencies : array-like or float
        Center frequencies of the resonances, in [Hz].
    quality_factors : array-like or float
        Quality factors (Q) of the resonances, dimensionless.
    shunt_impedances_counter_rotating : array-like or float or None
        Shunt impedances for counter-rotating mode.
    supersampling
        When using `get_wake_impedance`,
        will create a wake with n-times the time resolution,
        that is averaged into a single bin.
        This feature is used to prevent aliasing effects
        in time domain wake solvers.

    Notes
    -----
    All values must be float, if one is given as float.

    Ensure that all input arrays have the same length, with each entry
    corresponding to a separate resonance.
    """

    def __init__(
        self,
        shunt_impedances: NumpyArray | ArrayLike | float | int,
        center_frequencies: NumpyArray | ArrayLike | float | int,
        quality_factors: NumpyArray | ArrayLike | float | int,
        shunt_impedances_counter_rotating: NumpyArray
        | float
        | ArrayLike
        | None = None,
        supersampling: int = 0,
    ):
        warnings.warn("Untested code", NotTestedWarning, stacklevel=1)
        super().__init__(is_dynamic=False)

        self.supersampling = supersampling
        self._shunt_impedances: NumpyArray
        self._center_frequencies: NumpyArray
        self._quality_factors: NumpyArray
        self._n_resonators: int

        if (
            isinstance(shunt_impedances, float | int)
            and isinstance(center_frequencies, float | int)
            and isinstance(quality_factors, float | int)
        ):
            self._shunt_impedances = backend.array([shunt_impedances])
            self._center_frequencies = backend.array([center_frequencies])
            self._quality_factors = backend.array([quality_factors])
            self._n_resonators = len(self._shunt_impedances)
        else:
            assert len(shunt_impedances) == len(center_frequencies), (
                f"{len(shunt_impedances)} != {len(center_frequencies)}"
            )
            assert len(shunt_impedances) == len(quality_factors), (
                f"{len(shunt_impedances)} != {len(quality_factors)}"
            )
            self._shunt_impedances = np.array(shunt_impedances)
            self._center_frequencies = np.array(center_frequencies)
            self._quality_factors = np.array(quality_factors)
            self._n_resonators = len(shunt_impedances)

        self._shunt_impedances_counter_rotating: (
            NumpyArray | CupyArray | None
        ) = None

        if shunt_impedances_counter_rotating is not None:
            if isinstance(shunt_impedances_counter_rotating, float | int):
                shunt_impedances_counter_rotating = [
                    shunt_impedances_counter_rotating
                ]
            self._shunt_impedances_counter_rotating = backend.array(
                shunt_impedances_counter_rotating
            )

            assert len(self._shunt_impedances_counter_rotating) == len(
                self._shunt_impedances
            ), (
                "Array lengths between co- and counterrotating impedances need to match."
            )

            for imp, imp_cr in zip(
                self._shunt_impedances,
                self._shunt_impedances_counter_rotating,
                strict=False,
            ):
                assert backend.isclose(
                    backend.abs(imp), backend.abs(imp_cr)
                ), (
                    "Absolute value of co- and counter-rotating impedances mismatch, no energy conservation."
                )

        # secondary quantities for wake calculation
        self._omega = 2 * np.pi * self._center_frequencies
        self._alpha = self._omega / (2 * self._quality_factors)
        self._omega_bar = np.sqrt(self._omega**2 - self._alpha**2)

        # Test if one or more quality factors is smaller than 0.5.
        if backend.sum(self._quality_factors < 0.5) > 0:  # NOQA PLR2004
            raise RuntimeError(
                "All quality factors Q must be greater or equal 0.5"
            )
        if backend.sum(self._center_frequencies < 0) > 0:
            raise RuntimeError(
                "All center frequencies must be greater or equal 0"
            )

        self._cache_wake_impedance: NumpyArray | CupyArray | None = None
        self._cache_wake_impedance_hash: int | None = None

        self._cache_wake_impedance_counter_rotation: (
            NumpyArray | CupyArray | None
        ) = None
        self._cache_wake_impedance_counter_rotation_hash: int | None = None

        self._cache_impedance: NumpyArray | CupyArray | None = None
        self._cache_impedance_hash: int | None = None

    def get_wake_impedance(
        self,
        time: NumpyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray | CupyArray:  # Fixme all get_wake_impedance same
        """
        Get the wake function, but converted to frequency domain.

        Get impedance  computed via ``fft(...)`` from time domain
        analytical formula equivalent to the partial single-particle-wake.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of fft bins to use.

        Returns
        -------
        wake_impedance
            Wake impedance in frequency domain.
        """
        f_max = np.max(self._omega) / (2 * np.pi)
        T_max = 1 / f_max
        assert (time[1] - time[0]) <= (T_max / 2), (  # Nyquist-Frequency
            "The time step is not precise enough to consider the resonators"
            f" maximum frequency of {si_format(f_max)}Hz."
        )

        dt_antialias_required = T_max / 10
        dt_antialias = (time[1] - time[0]) / (2 * self.supersampling + 1)
        supersampling_required = int(
            math.ceil((time[1] - time[0]) / dt_antialias_required)
        )
        assert dt_antialias <= dt_antialias_required, (
            f"Use supersampling >= {supersampling_required} to prevent aliasing effects."
        )
        # Recalculate only if `time` has changed
        hash_ = get_hash(time)
        if hash_ == self._cache_wake_impedance_hash:
            return self._cache_wake_impedance

        wake = self.get_wake(time)
        for i in range(self.supersampling):
            offset = dt_antialias * i
            wake += self.get_wake(time + offset)
            wake += self.get_wake(time - offset)
        wake /= 2 * self.supersampling + 1
        wake_impedance = backend.fft.rfft(wake, n=n_fft)

        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance
        return wake_impedance

    def get_wake_impedance_counter_rotation(
        self,
        time: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray | CupyArray:  # Fixme all get_wake_impedance same
        """
        Get the wake function, but converted to frequency domain.

        Get impedance  computed via ``fft(...)`` from time domain
        analytical formula equivalent to the partial single-particle-wake.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation : Simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of fft bins to use.

        Returns
        -------
        wake_impedance
            Wake impedance in frequency domain for counter-rotating mode.
        """
        # Recalculate only if `time` has changed
        hash_ = get_hash(time + 1)  # to distinguish between counterrotation
        if hash_ == self._cache_wake_impedance_counter_rotation_hash:
            return self._cache_wake_impedance_counter_rotation

        wake_counter_rotation = self.get_wake_counter_rotation(time)
        wake_impedance_counter_rotation = backend.fft.rfft(
            wake_counter_rotation, n=n_fft
        )

        self._cache_wake_impedance_counter_rotation_hash = hash_
        self._cache_wake_impedance_counter_rotation = (
            wake_impedance_counter_rotation
        )
        return wake_impedance_counter_rotation

    def get_wake_impedance_freq(self, time):
        """
        Get frequency array corresponding to time used in :func:`get_wake_impedance`.

        Parameters
        ----------
        time
            Time array, in [s].

        Returns
        -------
        frequency_array
            Frequency array corresponding to the wake impedance.
        """
        return backend.fft.rfftfreq(
            len(self._cache_wake_impedance), time[1] - time[0]
        )

    def get_wake(self, time: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        """
        Compute the wake function of all resonators in time domain for the given time and return the summed potential.

        Parameters
        ----------
        time
            Time array at which the wake is calculated, in [s].

        Returns
        -------
        wake
            Wake potential array, in [V].
        """
        wake = backend.zeros(len(time), dtype=backend.float, order="C")

        heaviside_like = self.heaviside_eps_at_0(time)

        for res_ind in range(self._n_resonators):
            wake += (
                heaviside_like
                * (
                    self._shunt_impedances[res_ind]
                    * self._alpha[res_ind]
                    * backend.exp(-self._alpha[res_ind] * time)
                )
                * (
                    backend.cos(self._omega_bar[res_ind] * time)
                    - self._alpha[res_ind]
                    / self._omega_bar[res_ind]
                    * backend.sin(self._omega_bar[res_ind] * time)
                )
            )
        return wake

    def heaviside_eps_at_0(self, time: NumpyArray, eps: float = 0.01):
        """
        Calculate the Heaviside function (this is essentially a bugfix).

        ``cupy.linspace`` can produce slightly
        different results than ``np.linspace``,
        where 0 will be represented another small number like 1e-27.

        For this reason, we take ``eps * abs(time[1] - time[0])`` to
        identify a number close to 0 and tread is as 0 too.

        Parameters
        ----------
        time
            Axis along which to calculate the Heaviside function.
        eps
            Relative region around which a number will be assumed as 0.

        Returns
        -------
        heaviside
            The Heaviside function.
        """
        heaviside_like = (
            backend.sign(time) + 1.0
        )  # heaviside: /2 from heaviside and *2 from linac R/Q cancel
        # protect against numerical noise, where a 0 might be expressed as -1.6155871338926322e-27
        tol = eps * backend.abs(time[1] - time[0])
        bugfix = backend.abs(time) < tol
        heaviside_like[bugfix] = 1
        return heaviside_like

    def get_wake_counter_rotation(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Compute the wake function of all resonators in time domain for the given time and return the summed potential.

        Parameters
        ----------
        time
            Time array at which the wake is calculated, in [s].

        Returns
        -------
        wake_potential
            Potential array, in [V].
        """
        if self._shunt_impedances_counter_rotating is None:
            raise RuntimeError(
                "_shunt_impedances_counter_rotating needs to be set before calling this function."
            )

        wake = backend.zeros(len(time), dtype=backend.float, order="C")

        heaviside_like = self.heaviside_eps_at_0(time)

        for res_ind in range(self._n_resonators):
            wake += (
                (
                    heaviside_like
                )  # heaviside: /2 from heaviside and *2 from linac R/Q cancel
                * (
                    self._shunt_impedances_counter_rotating[res_ind]
                    * self._alpha[res_ind]
                    * backend.exp(-self._alpha[res_ind] * time)
                )
                * (
                    backend.cos(self._omega_bar[res_ind] * time)
                    - self._alpha[res_ind]
                    / self._omega_bar[res_ind]
                    * backend.sin(self._omega_bar[res_ind] * time)
                )
            )
        return wake

    def calculate_envelope(
        self, time_axis: NumpyArray | CupyArray | None = None
    ) -> tuple[NumpyArray, NumpyArray] | tuple[CupyArray, CupyArray]:
        """
        Calculate the normalized envelope of all resonators.

        Parameters
        ----------
        time_axis
            Time axis on which to calculate the envelope, in [s].

        Returns
        -------
        time_axis
            Time axis used for calculation.
        envelope
            Normalized envelope values.
        """
        if time_axis is None:
            time_axis = backend.linspace(
                0,
                backend.max(self._quality_factors / self._omega) * 20,
                100000,
            )
            # Should be sufficient, as the time between turns is
            # usually larger than the required time stepping in here, only gets called on init
        envelope = backend.zeros_like(time_axis)
        for res_ind in range(len(self._quality_factors)):
            envelope += (
                self._shunt_impedances[res_ind]
                * self._alpha[res_ind]
                * backend.exp(-time_axis * self._alpha[res_ind])
            )
        envelope /= backend.max(envelope)

        return time_axis, envelope

    def get_decay_time(self, decay_fraction_threshold: float) -> float:
        """
        Calculate the decay time of the full envelope based on a decay fraction threshold.

        Parameters
        ----------
        decay_fraction_threshold
            Decay fraction threshold.

        Returns
        -------
        storage_time
            Time, after which the decay fraction threshold is reached.
        """
        time_axis = backend.linspace(
            0,
            -2
            * np.max(self._quality_factors / self._omega)
            * np.log(decay_fraction_threshold)
            * 2,
            100000,
        )
        _, envelope = self.calculate_envelope(time_axis=time_axis)

        storage_time = float(
            time_axis[
                backend.abs(envelope - decay_fraction_threshold).argmin()
            ]
        )

        return storage_time

    def get_impedance(
        self,
        freq_x: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        counter_rotation: bool = False,
    ) -> NumpyArray | CupyArray:
        """
        Return the analytically calculated impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        counter_rotation
            Checkbox if the counter-rotating or corotating impedance should be used.

        Returns
        -------
        impedance
            Complex impedance array.
        """
        # Recalculate only if `freq_x` is changed

        hash_ = get_hash(freq_x + counter_rotation)
        if hash_ == self._cache_impedance_hash:
            return self._cache_impedance

        impedance = backend.zeros(len(freq_x), dtype=complex)
        n_centers = len(self._center_frequencies)

        shunt_impedance = (
            self._shunt_impedances_counter_rotating
            if counter_rotation
            else self._shunt_impedances
        )
        for i in range(n_centers):
            impedance[1:] += shunt_impedance[i] / (
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


class ImpedanceTable(WakeFieldSource):
    """Base class to manage impedance tables."""

    @staticmethod
    @abstractmethod  # pragma: no cover
    def from_file(
        filepath: PathLike, reader: ImpedanceReader
    ) -> ImpedanceTable:
        """
        Instance table from a file on the disk.

        Parameters
        ----------
        filepath
            Path of the file to lead.
        reader
            `ImpedanceReader` to interpret what's written in the file.

        Returns
        -------
        impedance_table
            The loaded table.
        """
        pass


class ImpedanceTableFreq(ImpedanceTable, FreqDomain):
    """
    Impedance table in frequency domain.

    Parameters
    ----------
    freq_x
        Frequency axis, in [Hz].
    freq_y
        Complex amplitudes in frequency domain.
    """

    def __init__(
        self,
        freq_x: NumpyArray,
        freq_y: NumpyArray,
    ):
        super().__init__(is_dynamic=False)

        self._freq_x = backend.array(freq_x)
        self._freq_y = backend.array(freq_y)

        self._cache_impedance = None
        self._cache_impedance_hash: int | None = None

    def get_impedance(
        self,
        freq_x: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.

        Returns
        -------
        impedance
            Complex impedance array.
        """
        # Recalculate only if `freq_x` is changed
        hash_ = get_hash(freq_x)
        if hash_ == self._cache_impedance_hash:
            return self._cache_impedance
        impedance = backend.interp(
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
        Instance table from a file on the disk.

        Parameters
        ----------
        filepath
            Path of the file to lead.
        reader
            `ImpedanceReader` to interpret what's written in the file.

        Returns
        -------
        impedance_table_freq
            The loaded impedance table in frequency domain.
        """
        x_array, y_array = reader.load_file(filepath=filepath)
        assert not np.any(np.isnan(x_array))
        assert not np.any(np.isnan(y_array))
        return ImpedanceTableFreq(freq_x=x_array, freq_y=y_array)


class ImpedanceTableTime(ImpedanceTable, TimeDomain):
    """
    Impedance table in time domain..

    Parameters
    ----------
    wake_x
        Wake time axis, in [s].
    wake_y
        Wake amplitude, in [V].
    """

    def __init__(
        self,
        wake_x: NumpyArray,
        wake_y: NumpyArray,
    ):
        super().__init__(is_dynamic=False)
        self._wake_x = backend.array(wake_x)
        self._wake_y = backend.array(wake_y)

        self._cache_wake_impedance: NumpyArray | CupyArray | None = None
        self._cache_wake_impedance_hash: int | None = None

    @staticmethod
    def from_file(
        filepath: PathLike | str, reader: ImpedanceReader
    ) -> ImpedanceTableTime:
        """
        Instance table from a file on the disk.

        Parameters
        ----------
        filepath
            Path of the file to lead.
        reader
            `ImpedanceReader` to interpret what's written in the file.

        Returns
        -------
        impedance_table_time
            The loaded impedance table in time domain.
        """
        x_array, y_array = reader.load_file(filepath=filepath)
        return ImpedanceTableTime(wake_x=x_array, wake_y=y_array)

    def get_wake_impedance(
        self,
        time: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
    ) -> NumpyArray:
        """
        Get impedance equivalent to the partial single-particle-wake in time domain.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of FFT points.

        Returns
        -------
        wake_impedance
            Wake impedance in frequency domain.
        """
        hash_ = get_hash(time)
        if hash_ == self._cache_wake_impedance_hash:
            return self._cache_wake_impedance
        if time.min() < self._wake_x.min():
            warnings.warn(
                "Interpolation of wake outside boundaries",
                stacklevel=1,
            )
        if time.max() > self._wake_x.max():
            warnings.warn(
                "Interpolation of wake outside boundaries",
                stacklevel=1,
            )
        wake = backend.interp(time, self._wake_x, self._wake_y)
        wake_impedance = backend.fft.rfft(wake, n=n_fft)
        self._cache_wake_impedance_hash = hash_
        self._cache_wake_impedance = wake_impedance
        return wake_impedance


# TODO rework docstring
class TravelingWaveCavity(WakeFieldSource, TimeDomain, FreqDomain):
    r"""
    Impedance of travelling wave cavities.

    Parameters
    ----------
    R_S
        Shunt impepdance, in [Ω].
    frequency_R
        Resonant frequency, in [Hz].
    a_factor
        Damping time `a`, in [s].

    Attributes
    ----------
    R_S
        Shunt impepdance, in [Ω].
    frequency_R
        Resonant frequency, in [Hz].
    a_factor
        Damping time `a`, in [s].

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

    Examples
    --------
    >>> R_S = [1, 2, 3]
    >>> frequency_R = [1, 2, 3]
    >>> a_factor = [1, 2, 3]
    >>> twc = TravelingWaveCavity(R_S, frequency_R, a_factor)
    >>> time = np.array(1,2,3)
    >>> twc.wake_calc(time)
    >>> frequency = np.array(1,2,3)
    >>> twc.get_impedance(frequency)
    """

    def __init__(
        self,
        R_S: float | ArrayLike,
        frequency_R: float | ArrayLike,
        a_factor: float | ArrayLike,
    ):
        if hasattr(R_S, "__len__"):
            assert len(R_S) == len(frequency_R), (
                f"{len(R_S)=}, but {len(frequency_R)=}."
            )
            assert len(R_S) == len(a_factor), (
                f"{len(R_S)=}, but {len(a_factor)=}."
            )
        else:
            R_S = float(R_S)
            frequency_R = float(frequency_R)
            a_factor = float(a_factor)
        super().__init__(is_dynamic=False)

        # Shunt impedance in :math:`\Omega`
        self.R_S = backend.array(R_S, dtype=float).flatten()

        # Resonant frequency in Hz
        self.frequency_R = backend.array(frequency_R, dtype=float).flatten()

        # Damping time a in s
        self.a_factor = backend.array(a_factor, dtype=float).flatten()

    def wake_calc(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        r"""
        Wake calculation method as a function of time.

        Parameters
        ----------
        time
            Time array to get wake, in [s].

        Returns
        -------
        wake
            Wake potential array.
        """
        wake = backend.zeros(time.shape, dtype=backend.float, order="C")

        for i in range(0, len(self.R_S)):
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = time <= a_tilde
            wake[indexes] += (
                (backend.sign(time[indexes]) + 1)
                * 2
                * self.R_S[i]
                / a_tilde
                * (1 - time[indexes] / a_tilde)
                * backend.cos(2 * np.pi * self.frequency_R[i] * time[indexes])
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
        Get impedance equivalent to the partial single-particle-wake in time domain.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of FFT points.

        Returns
        -------
        wake_impedance
            Wake impedance in frequency domain.
        """
        wake = self.wake_calc(time=time)
        wake_impedance = backend.fft.rfft(wake, n=n_fft)
        return wake_impedance

    def get_impedance(
        self,
        freq_x: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> NumpyArray | CupyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.

        Returns
        -------
        impedance
            Complex impedance array.
        """
        impedance = backend.zeros(
            len(freq_x), dtype=backend.complex, order="C"
        )

        for i in range(0, len(self.R_S)):
            arg2_minus = self.a_factor[i] * (freq_x - self.frequency_R[i])
            Zminus = self.R_S[i] * (
                (backend.sinc(arg2_minus * 0.5 / np.pi)) ** 2
                - 2j * (arg2_minus - backend.sin(arg2_minus)) / arg2_minus**2
            )

            arg2_plus = self.a_factor[i] * (freq_x + self.frequency_R[i])
            Zplus = self.R_S[i] * (
                (backend.sinc(arg2_plus * 0.5 / np.pi)) ** 2
                - 2j * (arg2_plus - backend.sin(arg2_plus)) / arg2_plus**2
            )
            Zminus[freq_x == self.frequency_R[i]] = self.R_S[i]
            imp = Zplus + Zminus
            impedance += imp
        return impedance
