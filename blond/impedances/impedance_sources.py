# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to describe classes for the calculation of wakes and impedances to
be used either alone or by InducedVoltage objects described in impedance.py.
The module consists of a parent class called _ImpedanceObject and several child
classes, as for example InputTable, Resonators and TravelingWaveCavity.**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Juan F. Esteban Mueller**,
    **Markus Schwarz**
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mpmath
import numpy as np
from scipy import integrate
from scipy.constants import c, physical_constants
from scipy.special import airy, gamma as gamma_func, kv, polygamma

from ..utils import bmath as bm
from ..utils import precision

if TYPE_CHECKING:
    from typing import Optional, Iterable, Tuple, Literal

    from numpy.typing import ArrayLike

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


def _infinite_samples_fourier_transform(
    ts: NumpyArray | CupyArray,
    impedance: NumpyArray | CupyArray,
    freq: NumpyArray | CupyArray,
    out: Optional[NumpyArray | CupyArray] = None,
) -> NumpyArray | CupyArray:
    """Convert the impedance to a wake

    Notes
    -----
    A conventional FFT can only deliver a wave as long as the sampling allows.
    The resulting wake will be a periodic function.
    To get infinitely long wakes, an infinite sampling is required.
    Thus, the impedance is linearly interpolated with infinite points using
    an analytic formula.

    The base formula to interpolate and fourier transform between two
    frequency samples $f_1$ and $f_2$ is:
    $\int_f_1^f_2\, (a_1(f)+ i a_2(f)) \exp(i 2 \pi f t)  \, df$
    with the linear function $a(f) = m f + b$
    The result of this integral is used in the code below.

    Parameters
    ----------
    ts
        Time points to calculate the inverse fourier transform for
    impedance
        Complex amplitudes in the frequency domain, a.k.a. impedance
    freq
        Frequency base points in frequency domain, corresponding to amp_f
    out
        If given, wave will be written to this array instead of creating a
        new array. This is intended for better performance in loops etc.

    Returns
    -------
    wake
        The single particle wakefield along ts
    """
    if out is None:
        wave = bm.zeros_like(ts, dtype=precision.real_t)
    else:
        assert len(out) == len(ts)
        assert out.dtype == precision.real_t
        wave = out

    for i in range(len(wave)):
        t = ts[i]
        f1 = freq[:-1]
        f2 = freq[1:]
        a1 = impedance[:-1]
        a2 = impedance[1:]

        _m = (a2 - a1) / (f2 - f1)
        _n = impedance[:-1] - _m * freq[:-1]
        pi = bm.pi
        cos = bm.cos
        sin = bm.sin
        if t > 0:
            m = _m.real
            n = _n.real
            equation_real = (
                2
                * pi
                * t
                * (
                    (f2 * m + n) * sin(2 * pi * f2 * t)
                    - (f1 * m + n) * sin(2 * pi * f1 * t)
                )
                - m * cos(2 * pi * f1 * t)
                + m * cos(2 * pi * f2 * t)
            ) / (4 * pi**2 * t**2)
            m = _m.imag
            n = _n.imag
            # the real part coming from imaginary contribution, because (
            # i * i) = -1
            equation_imag = (
                m * (sin(2 * pi * f1 * t) - sin(2 * pi * f2 * t))
                - 2 * pi * t * (f1 * m + n) * cos(2 * pi * f1 * t)
                + 2 * pi * t * (f2 * m + n) * cos(2 * pi * f2 * t)
            ) / (4 * pi**2 * t**2)
        elif t == 0:
            m = _m.real
            n = _n.real
            equation_real = -1 / 2 * (f1 - f2) * (f1 * m + f2 * m + 2 * n)
            equation_imag = 0
            equation_real /= 2  # beam loading theorem
        elif t < 0:
            equation_real = 0
            equation_imag = 0
        else:
            msg = f"Unexpected {t=}"
            raise RuntimeError(msg)
        wave[i] = (bm.sum(equation_real) + bm.sum(equation_imag)) / (bm.sum(f2 - f1))
    return wave


class _FftHandler:
    def __init__(
        self, frequencies: NumpyArray | CupyArray, amplitudes: NumpyArray | CupyArray
    ):
        """
        Initialize the wakefield generator with frequency-domain impedance data.

        Parameters
        ----------
        frequencies : NumpyArray | CupyArray
            Array of frequency values corresponding to the impedance data (in Hz).

        amplitudes : NumpyArray | CupyArray
            Array of complex impedance amplitudes at the given frequencies.
            Should be of the same length as `frequencies`.
        """
        assert len(frequencies) == len(amplitudes)
        self._frequencies = frequencies
        self._amplitudes = amplitudes

    def get_periodic_wake_by_time(
        self, time_array: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        r"""
        Generate a periodic wakefield corresponding to the given time array.

        This method reconstructs a time-domain wakefield using the inverse FFT of a
        frequency-domain impedance spectrum, interpolated onto the frequency bins
        associated with the given time samples. The resulting signal is periodic
        with the duration defined by the input `time_array`.

        Parameters
        ----------
        time_array : NumpyArray | CupyArray
            Array of time points at which the wakefield should be evaluated.
            Assumes uniform spacing.

        Returns
        -------
        wake : NumpyArray | CupyArray
            Periodic wakefield signal corresponding to the given `time_array`.

        Notes
        -----
        - The period of the reconstructed wake is inferred from the span of `time_array`,
          i.e., `time_array.max() - time_array.min()`.
        - Frequency-domain impedance values are linearly interpolated over the FFT
          frequency bins before applying the inverse FFT.
        - If `time_array[0] != 0`, the resulting wake is shifted and re-interpolated
          to match the original time offset while maintaining periodicity.
        """
        period = time_array.max() - time_array.min()
        dt = time_array[1] - time_array[0]
        f_itp = bm.fft.rfftfreq(len(time_array), d=dt)
        imp_itp = bm.interp(
            f_itp,
            self._frequencies,
            self._amplitudes,
            left=0,
            right=0,
        )
        wake = bm.irfft(imp_itp)
        assert len(wake) == len(time_array)
        if time_array[0] != 0:
            wake = bm.interp(
                time_array,
                dt * bm.arange(len(wake)),
                wake,
                period=period,
            )
        return wake

    def get_periodic_wake(
        self, t_periodicity: float
    ) -> Tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]:
        """
        Generate a time-domain periodic wake signal over a specified interval.

        This method reconstructs a wake signal in the time domain using the inverse real FFT
        of interpolated frequency-domain amplitudes. The signal is assumed to be periodic
        over the given time interval.

        Parameters
        ----------
        t_periodicity : float
            Duration of the periodic interval over which the wake signal is generated.

        Returns
        -------
        Tuple[NumpyArray | CupyArray, NumpyArray | CupyArray]
            - `ts_itp`: Array of time samples over the interval [0, t_periodicity].
            - `wake_itp`: Reconstructed periodic wake signal corresponding to `ts_itp`.
        """
        ts_itp = np.linspace(0, t_periodicity, 2 * len(self._frequencies) - 2)
        fs_itp = bm.fft.rfftfreq(len(ts_itp), d=ts_itp[1] - ts_itp[0])
        amps_itp = bm.interp(
            fs_itp, self._frequencies, self._amplitudes, left=0, right=0
        )
        wake_itp = bm.fft.irfft(amps_itp)
        assert len(wake_itp) == len(ts_itp)
        return ts_itp, wake_itp

    def get_non_periodic_wake(self, time_array: NumpyArray | CupyArray):
        """Compute the non-periodic wakefield from impedance data.

        This method performs an analytical inverse Fourier transform of the impedance
        spectrum to obtain a time-domain wakefield over arbitrary time points. Unlike
        FFT-based methods, this approach enables computation of non-periodic (infinite-length)
        wakes by integrating over a linearly interpolated impedance function between frequency bins.


        Notes
        -----
        Standard FFT-based reconstruction yields a periodic signal limited by the sampling length.
        To overcome this and simulate an infinitely long wakefield, the impedance is interpolated
        between frequency samples using a linear model and integrated analytically.

        The base formula to interpolate and fourier transform between two
        frequency samples $f_1$ and $f_2$ is:
        $\int_f_1^f_2\, (a_1(f)+ i a_2(f)) \exp(i 2 \pi f t)  \, df$
        with the linear function $a(f) = m f + b$
        The result of this integral is used in the code below.

        Parameters
        ----------
        time_array : NumpyArray | CupyArray
            Array of time points at which to evaluate the wakefield.


        Returns
        -------
        wake : NumpyArray | CupyArray
            The single-particle wakefield evaluated at the given time points.
        """
        # TODO write test, assertion might fail
        return _infinite_samples_fourier_transform(
            impedance=self._amplitudes, ts=time_array, freq=self._frequencies
        )


class _ImpedanceObject(ABC):
    r"""
    Parent impedance object to implement required methods and attributes
    common to all the child classes. The attributes are initialised to 0 but
    they are overwritten by float arrays when the child classes are used.
    """

    def __init__(self):
        # Time array of the wake in s
        self.time_array: NumpyArray | CupyArray = 0

        # Wake array in :math:`\Omega / s`
        self.wake: NumpyArray | CupyArray = 0

        # Frequency array of the impedance in Hz
        self.frequency_array: NumpyArray | CupyArray = 0

        # Impedance array in :math:`\Omega`
        self.impedance: NumpyArray | CupyArray = 0

    @abstractmethod
    def wake_calc(self, time_array: NumpyArray):
        r"""
        Method required to compute the wake function. Returns an error if
        called from an object which does not implement this method.
        """
        # WrongCalcError
        raise NotImplementedError(
            "wake_calc() method not implemented in this class. "
            + "This object is probably meant to be used in the "
            + "frequency domain"
        )

    @abstractmethod
    def imped_calc(self, frequency_array: NumpyArray):
        r"""
        Method required to compute the impedance. Returns an error if called
        from an object which does not implement this method.
        """
        # WrongCalcError
        raise NotImplementedError(
            "imped_calc() method not implemented in this class. "
            + "This object is probably meant to be used in the "
            + "time domain"
        )


class InputTableTimeDomain(_ImpedanceObject):
    r"""
    Intensity effects from impedance and wake tables of a wake table is passed;
    Be careful, the input wake for W(t=0) should be already
    divided by two (beam loading theorem)

    Parameters
    ----------
    time_array : float array
        Time array of the wake in s or frequency array of the impedance in Hz
    wake : float array
        Wake array in :math:`\Omega / s` or real part of impedance
        in :math:`\Omega`


    Attributes
    ----------
    time_array : float array
        Input time array of the wake in s
    wake : float array
        Input wake array in :math:`\Omega / s`

    Examples
    --------
    >>> ##### WAKEFIELD TABLE
    >>> time = np.array([1.1,2.2,3.3])
    >>> wake = np.array([4.4,5.5,6.6])
    >>> table = InputTableTimeDomain(time, wake)
    >>> time_array = np.array([7.7,8.8,9.9])
    >>> table.wake_calc(time_array)
    """

    def __init__(self, time_array: ArrayLike[float], wake: ArrayLike[float]):
        _ImpedanceObject.__init__(self)
        # Time array of the wake in s
        self.time_array = self._time_array_org = time_array
        # Wake array in :math:`\Omega / s`
        self.wake = self._wake_org = wake

    def wake_calc(self, time_array: NumpyArray):
        r"""
        The wake from the table is interpolated using the new time array.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        Attributes
        ----------
        time_array : float array
            Input time array in s
        """

        self.time_array = time_array
        self.wake = np.interp(
            self.time_array, self._time_array_org, self._wake_org, right=0
        )

    def imped_calc(self, frequency_array: NumpyArray):
        dt = 1 / (2 * frequency_array.max())  # Nyquist rate
        ts_itp = dt * np.arange(len(frequency_array) * 2 - 2)
        wake_itp = np.interp(ts_itp, self.time_array, self.wake)
        impedance = np.fft.rfft(wake_itp)
        assert len(frequency_array) == len(impedance)
        self.frequency_array = frequency_array
        self.impedance = impedance


class InputTableFrequencyDomain(_ImpedanceObject):
    r"""
    Intensity effects from impedance and wake tables.
    If the constructor takes just two arguments, then a wake table is passed;
    if it takes three arguments, then an impedance table is passed. Be careful
    that if you input a wake, the input wake for W(t=0) should be already
    divided by two (beam loading theorem) ; and that if you input impedance,
    only the positive  frequencies of the impedance is needed (the impedance
    will be assumed to be Hermitian (Real part symmetric and Imaginary part
    antisymmetric).Note that we add the point (f, Z(f)) = (0, 0) to the
    frequency and impedance arrays derived from the table.

    Parameters
    ----------
    frequency_array : float array
        Time array of the wake in s or frequency array of the impedance in Hz
    Re_Z_array : float array
        Wake array in :math:`\Omega / s` or real part of impedance
        in :math:`\Omega`
    Im_Z_array : float array
        Imaginary part of impedance in :math:`\Omega`


    Attributes
    ----------
    _frequency_array float array
        Input frequency array of the impedance in Hz
    _impedance complex array
        Input impedance array in :math:`\Omega + j \Omega`

    Examples
    --------

    >>> ##### IMPEDANCE TABLE
    >>> frequency = np.array([1.1,2.2,3.3])
    >>> real_part = np.array([4.4,5.5,6.6])
    >>> imaginary_part = np.array([7.7,8.8,9.9])
    >>> table2 = InputTableFrequencyDomain(frequency, real_part, imaginary_part)
    >>> new_freq_array = np.array([7.7,8.8,9.9])
    >>> table2.imped_calc(new_freq_array)

    """

    def __init__(
        self,
        frequency_array: ArrayLike[float],
        Re_Z_array: ArrayLike[float],
        Im_Z_array: Optional[ArrayLike[float]] = None,
    ):
        _ImpedanceObject.__init__(self)

        # Frequency array of the impedance in Hz
        self._frequency_array_org = frequency_array
        # Real part of impedance in :math:`\Omega`
        _Re_Z_array_org = Re_Z_array = Re_Z_array
        # Imaginary part of impedance in :math:`\Omega`
        _Im_Z_array_org = Im_Z_array = Im_Z_array

        if self._frequency_array_org[0] != 0:
            self._frequency_array_org = np.hstack((0, self._frequency_array_org))
            _Re_Z_array_org = np.hstack((0, _Re_Z_array_org))
            _Im_Z_array_org = np.hstack((0, _Im_Z_array_org))

        # Impedance array in :math:`\Omega`
        self._impedance_org = self.impedance = _Re_Z_array_org + 1j * _Im_Z_array_org

        self.t_periodicity: Optional[float, Literal["auto"]] = "auto"  # "auto",
        # to guarantee original `wake_calc` behaviour

    def wake_calc(self, time_array: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        r"""The wake from the table is interpolated using the new time array.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        """
        time_array_dt = time_array[1] - time_array[0]
        time_array_f_cutoff = 1 / (2 * time_array_dt)
        if time_array_f_cutoff > self._frequency_array_org.max():
            msg = (
                f"`time_array` has a cutoff frequency of "
                f"{time_array_f_cutoff} Hz, but the input table ends already "
                f"at {self._frequency_array_org.max()} Hz"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        fft_handler = _FftHandler(
            frequencies=self._frequency_array_org, amplitudes=self._impedance_org
        )

        if self.t_periodicity == "auto":
            # periodicity is same as time array length
            wake = fft_handler.get_periodic_wake_by_time(time_array=time_array)
        elif isinstance(self.t_periodicity, float):
            # this allows periodicity to be different from time_array length
            ts_itp, wake_itp = fft_handler.get_periodic_wake(
                t_periodicity=self.t_periodicity
            )
            wake = np.interp(time_array, ts_itp, wake_itp, period=self.t_periodicity)
        elif self.t_periodicity is None:
            # wave that decays towards infinity
            wake = fft_handler.get_non_periodic_wake(time_array=time_array)
        else:
            raise ValueError(f"{self.t_periodicity=}")
        return wake

    def imped_calc(self, frequency_array: NumpyArray):
        r"""
        The impedance from the table is interpolated using the new frequency
        array.

        Parameters
        ----------
        frequency_array : float array
            frequency array in :math:`\Omega`

        Attributes
        ----------
        frequency_array : float array
            frequency array in :math:`\Omega`
        Re_Z_array : float array
            Real part of the interpolated impedance in :math:`\Omega`
        Im_Z_array : float array
            Imaginary part of the interpolated impedance in :math:`\Omega`
        impedance : complex array
            Output interpolated impedance array in :math:`\Omega + j \Omega`
        """

        Re_Z = np.interp(
            frequency_array,
            self._frequency_array_org,
            self._impedance_org.real,
            right=0,
        )
        Im_Z = np.interp(
            frequency_array,
            self._frequency_array_org,
            self._impedance_org.imag,
            right=0,
        )
        self.frequency_array = frequency_array
        self.impedance = Re_Z + 1j * Im_Z


class Resonators(_ImpedanceObject):
    r"""
    Impedance contribution from resonators, analytic formulas for both wake
    and impedance. The resonant modes (and the corresponding R and Q)
    can be inputed as a list in case of several modes.
    The model is the following:

    .. math::

        Z(f) = \frac{R}{1 + j Q \left(\frac{f}{f_r}-\frac{f_r}{f}\right)}

    .. math::

        W(t>0) = 2\alpha R e^{-\alpha t}\left(\cos{\bar{\omega}t} - \frac{\alpha}{\bar{\omega}}\sin{\bar{\omega}t}\right)

        W(0) = \alpha R

    .. math::

        \omega_r = 2 \pi f_r

        \alpha = \frac{\omega_r}{2Q}

        \bar{\omega} = \sqrt{\omega_r^2 - \alpha^2}

    Parameters
    ----------
    R_S : float list
        Shunt impepdance in :math:`\Omega`
    frequency_R : float list
        Resonant frequency in Hz
    Q : float list
        Quality factor

    Attributes
    ----------
    R_S : float array
        Shunt impepdance in :math:`\Omega`
    frequency_R : float array
        Resonant frequency in Hz
    Q : float array
        Quality factor
    n_resonators : int
        number of resonators

    Examples
    ----------
    >>> R_S = [1, 2, 3]
    >>> frequency_R = [1, 2, 3]
    >>> Q = [1, 2, 3]
    >>> resonators = Resonators(R_S, frequency_R, Q)
    >>> time = np.array(1,2,3)
    >>> resonators.wake_calc(time)
    >>> frequency = np.array(1,2,3)
    >>> resonators.imped_calc(frequency)
    """

    def __init__(
        self,
        R_S: float | list[float] | NumpyArray,
        frequency_R: float | list[float] | NumpyArray,
        Q: float | list[float] | NumpyArray,
    ) -> None:
        _ImpedanceObject.__init__(self)

        # Shunt impedance in :math:`\Omega`
        self.R_S = np.array([R_S], dtype=float).flatten()

        # Resonant frequency in Hz
        self.frequency_R = np.array([frequency_R], dtype=float).flatten()

        # Quality factor
        self.Q = np.array([Q], dtype=float).flatten()

        # Test if one or more quality factors is smaller than 0.5.
        if np.count_nonzero(self.Q < 0.5) > 0:
            # ResonatorError
            raise RuntimeError("All quality factors Q must be greater or equal 0.5")

        # Number of resonant modes
        self.n_resonators = len(self.R_S)

    @property
    def frequency_R(self):
        return self.__frequency_R

    @frequency_R.setter
    def frequency_R(self, frequency_R):
        self.__frequency_R = frequency_R
        self.__omega_R = 2 * np.pi * frequency_R

    @property
    def omega_R(self):
        return self.__omega_R

    # Resonant angular frequency in rad/s
    @omega_R.setter
    def omega_R(self, omega_R):
        self.__frequency_R = omega_R / 2 / np.pi
        self.__omega_R = omega_R

    def wake_calc(self, time_array: NumpyArray) -> None:
        r"""
        Wake calculation method as a function of time.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        Attributes
        ----------
        time_array : float array
            Input time array in s
        wake : float array
            Output wake in :math:`\Omega / s`
        """

        self.time_array = time_array
        self.wake = np.zeros(
            self.time_array.shape, dtype=bm.precision.real_t, order="C"
        )

        for i in range(0, self.n_resonators):
            alpha = self.omega_R[i] / (2 * self.Q[i])
            omega_bar = np.sqrt(self.omega_R[i] ** 2 - alpha**2)

            self.wake += (
                (np.sign(self.time_array) + 1)
                * self.R_S[i]
                * alpha
                * np.exp(-alpha * self.time_array)
                * (
                    bm.cos(omega_bar * self.time_array)
                    - alpha / omega_bar * bm.sin(omega_bar * self.time_array)
                )
            )

    def imped_calc(self, frequency_array: NumpyArray):
        r"""
        Impedance calculation method as a function of frequency optimised in C++

        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz

        Attributes
        ----------
        frequency_array : float array
            input frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        # TODO does bm.fast_resonator exist in all backends?
        self.frequency_array = frequency_array
        self.impedance = bm.fast_resonator(
            self.R_S, self.Q, self.frequency_array, self.frequency_R
        )

    def _imped_calc_python(self, frequency_array: NumpyArray):
        r"""
        Impedance calculation method as a function of frequency using Python.

        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz

        Attributes
        ----------
        frequency_array : float array
            nput frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        warnings.warn(
            f"Use imped_calc(frequency_array=your_array) instead!",
            DeprecationWarning,
            stacklevel=2,
        )

        self.frequency_array = frequency_array
        self.impedance = np.zeros(
            len(self.frequency_array), dtype=bm.precision.complex_t, order="C"
        )

        for i in range(0, self.n_resonators):
            self.impedance[1:] += self.R_S[i] / (
                1
                + (1j * self.Q[i])
                * (
                    self.frequency_array[1:] / self.frequency_R[i]
                    - self.frequency_R[i] / self.frequency_array[1:]
                )
            )

    def _imped_calc_cpp(self, frequency_array: NumpyArray) -> None:
        r"""
        Impedance calculation method as a function of frequency optimised in C++

        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz

        Attributes
        ----------
        frequency_array : float array
            nput frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """
        warnings.warn(
            f"Use imped_calc(frequency_array=your_array) instead!",
            DeprecationWarning,
            stacklevel=2,
        )

        self.frequency_array = frequency_array
        self.impedance = bm.fast_resonator(
            self.R_S, self.Q, self.frequency_array, self.frequency_R
        )


class TravelingWaveCavity(_ImpedanceObject):
    r"""
    Impedance contribution from travelling wave cavities, analytic formulas for
    both wake and impedance. The resonance modes (and the corresponding R and a)
    can be inputed as a list in case of several modes.

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
    R_S : float list
        Shunt impepdance in [:math:`\Omega`]
    frequency_R : float list
        Resonant frequency in [Hz]
    a_factor : float list
        Damping time a in [s]

    Attributes
    ----------
    R_S : float array
        Shunt impepdance in :math:`\Omega`
    frequency_R : float array
        Resonant frequency in Hz
    a_factor : float array
        Damping time a in s
    n_twc : int
        number of resonant modes

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
        R_S: float | Iterable[float] | NumpyArray,
        frequency_R: float | Iterable[float] | NumpyArray,
        a_factor: float | Iterable[float] | NumpyArray,
    ) -> None:
        _ImpedanceObject.__init__(self)

        # Shunt impepdance in :math:`\Omega`
        self.R_S = np.array([R_S], dtype=float).flatten()

        # Resonant frequency in Hz
        self.frequency_R = np.array([frequency_R], dtype=float).flatten()

        # Damping time a in s
        self.a_factor = np.array([a_factor], dtype=float).flatten()

        # Number of resonant modes
        self.n_twc = len(self.R_S)

    def wake_calc(self, time_array: NumpyArray) -> None:
        r"""
        Wake calculation method as a function of time.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        Attributes
        ----------
        time_array : float array
            Input time array in s
        wake : float array
            Output wake in :math:`\Omega / s`
        """

        self.time_array = time_array
        self.wake = np.zeros(
            self.time_array.shape, dtype=bm.precision.real_t, order="C"
        )
        self.wake = np.zeros(self.time_array.shape)

        for i in range(0, self.n_twc):
            a_tilde = self.a_factor[i] / (2 * np.pi)
            indexes = self.time_array <= a_tilde
            self.wake[indexes] += (
                (np.sign(self.time_array[indexes]) + 1)
                * 2
                * self.R_S[i]
                / a_tilde
                * (1 - self.time_array[indexes] / a_tilde)
                * bm.cos(2 * np.pi * self.frequency_R[i] * self.time_array[indexes])
            )

    def imped_calc(self, frequency_array: NumpyArray):
        r"""
        Impedance calculation method as a function of frequency.

        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz

        Attributes
        ----------
        frequency_array : float array
            input frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """

        self.frequency_array = frequency_array
        freq = self.frequency_array
        self.impedance = np.zeros(len(freq), dtype=bm.precision.complex_t, order="C")

        for i in range(0, self.n_twc):
            freq_r = self.frequency_R[i]
            a_factor = self.a_factor[i]
            r_s = self.R_S[i]
            Zplus = r_s * (
                (
                    bm.sin((a_factor / 2) * (freq - freq_r))
                    / ((a_factor / 2) * (freq - freq_r))
                )
                ** 2
                - 2j
                * (a_factor * (freq - freq_r) - bm.sin(a_factor * (freq - freq_r)))
                / (a_factor * (freq - freq_r)) ** 2
            )

            Zminus = r_s * (
                (
                    bm.sin((a_factor / 2) * (freq + freq_r))
                    / ((a_factor / 2) * (freq + freq_r))
                )
                ** 2
                - 2j
                * (a_factor * (freq + freq_r) - bm.sin(a_factor * (freq + freq_r)))
                / (a_factor * (freq + freq_r)) ** 2
            )

            self.impedance += Zplus + Zminus


class ResistiveWall(_ImpedanceObject):
    r"""
    Impedance contribution from resistive wall for a cilindrical beam pipe

    The model is the following:

    .. math::

        Z(f) = \frac{Z_0 c L}{ \pi } \frac{ 1 }{ \left[1 - i \text{sign} f \right] 2  b  c
                                    \sqrt{ \frac{\sigma_c Z_0 c }{ 4 \pi |f| } + i 2 \pi b^2 f } }

    Parameters
    ----------
    pipe_radius : float
        Beam pipe radius in m
    pipe_length : float
        Beam pipe length in m
    resistivity : float
        Beam pipe resistivity in :math:`m / s`
    conductivity : float
        Beam pipe conductivity in :math:`s / m`

    Attributes
    ----------
    pipe_radius : float
        Beam pipe radius in m
    pipe_length : float
        Beam pipe length in m
    conductivity : float
        Beam pipe conductivity in :math:`s / m`
    Z0 : float
        Characteristic impedance of vacuum in* :math:`\Omega`

    Examples
    ----------
    >>> pipe_radius = 1
    >>> pipe_length = 2
    >>> resistivity = 3
    >>> rw = ResistiveWall(pipe_radius, pipe_length, resistivity)
    >>> frequency = np.array(1,2,3)
    >>> rw.imped_calc(frequency)
    """

    def __init__(
        self,
        pipe_radius: float,
        pipe_length: float,
        resistivity: Optional[float] = None,
        conductivity: Optional[float] = None,
    ):
        _ImpedanceObject.__init__(self)

        # Beam pipe radius in m
        self.pipe_radius = float(pipe_radius)

        # Beam pipe length in m
        self.pipe_length = float(pipe_length)

        # Beam pipe conductivity in :math:`s / m`
        if resistivity is not None:
            self.conductivity = 1 / resistivity
        elif conductivity is not None:
            self.conductivity = conductivity
        else:
            # MissingParameterError
            raise RuntimeError(
                "At least one of the following parameters "
                "should be provided: resistivity or conductivity"
            )

        # Characteristic impedance of vacuum in* :math:`\Omega`
        self.Z0 = physical_constants["characteristic impedance of vacuum"][0]

        self.t_periodicity: Optional[float, Literal["auto"]] = "auto"  # "auto",
        # to guarantee original `wake_calc` behaviour

    @property
    def resistivity(self):
        return self.__resistivity

    @resistivity.setter
    def resistivity(self, resistivity):
        self.__resistivity = resistivity
        self.__conductivity = 1 / resistivity

    @property
    def conductivity(self):
        return self.__conductivity

    @conductivity.setter
    def conductivity(self, conductivity):
        self.__resistivity = 1 / conductivity
        self.__conductivity = conductivity

    def wake_calc(self, time_array: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        r"""The wake from the table is interpolated using the new time array.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        """
        time_array_dt = time_array[1] - time_array[0]
        time_array_f_cutoff = 1 / (2 * time_array_dt)
        if time_array_f_cutoff > self.frequency_array.max():
            msg = (
                f"`time_array` has a cutoff frequency of "
                f"{time_array_f_cutoff} Hz, but the input table ends already "
                f"at {self.frequency_array.max()} Hz"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        fft_handler = _FftHandler(
            frequencies=self.frequency_array, amplitudes=self.impedance
        )

        if self.t_periodicity == "auto":
            # periodicity is same as time array length
            wake = fft_handler.get_periodic_wake_by_time(time_array=time_array)
        elif isinstance(self.t_periodicity, float):
            # this allows periodicity to be different from time_array length
            ts_itp, wake_itp = fft_handler.get_periodic_wake(
                t_periodicity=self.t_periodicity
            )
            wake = np.interp(time_array, ts_itp, wake_itp, period=self.t_periodicity)
        elif self.t_periodicity is None:
            # wave that decays towards infinity
            wake = fft_handler.get_non_periodic_wake(time_array=time_array)
        else:
            raise ValueError(f"{self.t_periodicity=}")
        return wake

    def wake_calc_old(self, time_array: NumpyArray):
        r"""
        The wake from the table is interpolated using the new time array.

        Parameters
        ----------
        time_array : float array
            Input time array in s

        Attributes
        ----------
        time_array : float array
            Input time array in s
        wake: float array
            Output interpolated wake in :math:`\Omega / s`
        """
        warnings.warn(
            "Use `self.wake_calc` instead with `self.t_periodicity = auto`!",
            DeprecationWarning,
            stacklevel=2,
        )
        f_itp = np.fft.rfftfreq(len(time_array), d=(time_array[1] - time_array[0]))

        imp_itp = np.interp(
            f_itp,
            self.frequency_array,
            self.impedance,
            left=0,
            right=0,
        )
        self.wake = bm.irfft(imp_itp)
        assert len(self.wake) == len(time_array)
        self.time_array = time_array

    def imped_calc(self, frequency_array: NumpyArray):
        r"""
        Impedance calculation method as a function of frequency.

        Parameters
        ----------
        frequency_array : float array
            Input frequency array in Hz

        Attributes
        ----------
        frequency_array : float array
            Input frequency array in Hz
        impedance : complex array
            Output impedance in :math:`\Omega + j \Omega`
        """

        self.frequency_array = frequency_array

        self.impedance = (
            self.Z0
            * c
            * self.pipe_length
            / (
                (1.0 - 1j * np.sign(self.frequency_array))
                * (np.pi * 2 * self.pipe_radius * c)
                * np.sqrt(
                    (self.conductivity * self.Z0 * c)
                    / (4.0 * np.pi * np.abs(self.frequency_array))
                )
                + 1j * self.pipe_radius**2.0 * 2.0 * np.pi * self.frequency_array
            )
        ).astype(dtype=bm.precision.complex_t, order="C", copy=False)

        self.impedance[np.isnan(self.impedance)] = 0.0


class CoherentSynchrotronRadiation(_ImpedanceObject):
    r"""
    Impedance contribution of coherent synchrotron radiation emitted insisde a dipole magnet with
    bending raidus :math:`R`.

    The free-space impedance increases as :math:`f^{1/3}` until the critical frequency

    .. math::
        f_{\text{crit}} = \frac{3}{4\pi} \gamma^3 \frac{c}{R},

    and is exponentially surpressed above. Notice that the critical frequency strongly depends on
    energy, but the low-frequency regime is energy-independent.

    The parallel-plates impedance models the vacuum chamber as a pair of infinite, perfectly
    conducting plates, separated by a distance `chamber_height` (i.e. distance from top to bottom).
    Frequencies bewlow the cut-off frequency

    .. math::
        f_{\text{cut}} = \sqrt{\frac{2}{3}} \left(\pi \frac{R}{h}\right)^{3/2} \frac{c}{R}\,,

    are exponentially surpressed. For larger frequencies, it approaches the free-space impedance.

    Depending on the input parameters, different models are chosen:

    * Without any optional parameters, the low-frequency approximation to the free-space impedance
      is used, given by eq. 6.18 of [Murphy1997]_
    * If the Lorentz `gamma` is passed as the *only* optional parameter, the exact free-space
      impedance is used, implemented as eqs. A4 and A5 of [Murphy1997]_.
    * If the height of the vacuum chamber (from top to bottom) is passed as the *only* optional
      `chamber_height`, the low-frequency parallel-plates model as given by eq. 8 of
      [Chao2011]_ is used.
    * Finally, if both `gamma` *and* the `chamber_height` are passed, the parallel-plate impedance
      model from eq. B13 in [Murphy1997]_ is used.

    .. image:: csr_impedance_real.png
        :align: right

    The evaluation of the exact impedances is numerically unstable (and time-consuming) for
    frequencies larger than the critical frequency. Therefore,
    the kwarg `high_frequency_transition` sets the frequency (in units of the critical
    frequency) above which a simpler approximate expression is used.

    Note
    ----------
    The (incoherent) energy loss due to synchrotron radiation is *not* included. To include it,
    use :class:`~blond.synchrotron_radiation.synchrotron_radiation.SynchrotronRadiation`.

    References
    ----------
    .. [Murphy1997] J.B. Murphy, S. Krinsky, R.L. Gluckstern, "Longitudinal wakefield for
        an electron moving on a circular orbit", *Particle Accelerators*, vol. 57, p. 9--64, 1997.

    .. [Chao2011] Y. Cai, "Theory of microwave instability and coherent synchrotron radiation
        in electron storage rings", in *Proc. IPAC'11*, San Sebastian, Spain, May 2011,
        pp. 3774--3778

    Examples
    ----------
    >>> from blond.impedances.impedance_sources import CoherentSynchrotronRadiation
    >>> freqs = 10**np.linspace(8,15, num=200)  # frequencies at which to compute impedance

    Specify bending radius and chamber height (both in [m]) and the Lorentz gamma
    for 40MeV electrons

    >>> r_bend, h, gamma = 1.273, 32e-3, 40e6/511e3

    With only the bending radius, only the low-freqeuncy free-space impedance is computed:

    >>> Z_fs_appr = CoherentSynchrotronRadiation(r_bend)
    >>> Z_fs_appr.imped_calc(freqs)

    To compute the free-space impedance, also pass only the Lorentz gamma:

    >>> Z_fs = CoherentSynchrotronRadiation(rBend, gamma=gamma)
    >>> Z_fs.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)

    To compute the approximate parallel-plates impedance, pass the chamber height:

    >>> Z_pp_appr = CoherentSynchrotronRadiation(rBend, chamber_height=h)
    >>> Z_pp_appr.imped_calc(freqs, high_frequency_transition=10)

    For the computation of the parallel-plates impedance you need to pass both the chamber height
    and the Lorentz gamma:

    >>> Z_pp = CoherentSynchrotronRadiation(rBend, chamber_height=h, gamma=gamma)
    >>> Z_pp.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=20)

    """

    def wake_calc(self, time_array: NumpyArray):
        raise NotImplementedError
        pass  # TODO

    def imped_calc(self, frequency_array: NumpyArray):
        raise NotImplementedError
        pass  # TODO

    def __init__(
        self,
        r_bend: float,
        gamma: Optional[float] = None,
        chamber_height: float = np.inf,
    ) -> None:
        r"""


        Parameters
        ----------
        r_bend : float
            Bending radius in m
        gamma : float, optional
            The Lorentz factor is only used when the impedance is computed The default is None.
        chamber_height : float
            The height of the vacuum chamber, i.e. the separation between the parallel plates.
            `chamber_height` is required for the
            The default is None.
        low_frequency_approximation : bool, optional
            If true, the low-frequency approximation for either the free-space or parallel
            plates impedances are used. If false, the Lorentz factor `gamma` must be specified.
            The default is True.
        parallel_plates : TYPE, optional
            If ture, the parallel plates impedance is computed. In this case, `chamber_height`
            must be specified. If false, the free-space impedance is computed. The default is False.

        Raises
        ------
        ValueError
            Raised when input parameters are inconsistent, e.g. using the parallel plates impedance,
            but not specifying the `chamber_height`.
        RuntimeError
            Raised when the method for computing the impedance could not be deterimined.

        Returns
        -------
        None.

        """
        _ImpedanceObject.__init__(self)

        # Characteristic impedance of vacuum in* :math:`\Omega`
        self.Z0 = physical_constants["characteristic impedance of vacuum"][0]

        self.r_bend = r_bend
        self.gamma = gamma
        self.chamber_height = chamber_height

        # test for input consistency
        if self.r_bend <= 0.0:
            raise ValueError("bending radius must be greater 0")
        if self.chamber_height <= 0.0:
            raise ValueError("chamber_height must be greater 0")
        if self.gamma is not None and self.gamma <= 1.0:
            raise ValueError("gamma must be greater 1")

        # TODO use f_0 = f_rev (from ring length) or f_0 = c/r_bend/2pi (only path in dipoles)?
        self.f_0 = 0.5 * c / self.r_bend / np.pi  # assumes beta == 1

        if self.chamber_height < np.inf:
            self.Delta = self.chamber_height / self.r_bend
            # parallel plates cut-off frequency
            self.f_cut = np.sqrt(2 / 3) * (np.pi / self.Delta) ** 1.5 * self.f_0

        if self.gamma is not None:
            # critical frequency in Hz
            self.f_crit = 0.75 * self.gamma**3 * c / self.r_bend / np.pi

            # used for the computation of the free-space impedance
            self.hyper_vec = np.vectorize(mpmath.hyper, excluded=[0, 1], otypes=[float])

        # chose proper impedance method based on input
        if self.gamma is None and np.isinf(self.chamber_height):
            self.imped_calc = self._fs_low_frequency_wrapper

        elif self.gamma is not None and np.isinf(self.chamber_height):
            self.imped_calc = self._fs_spectrum

        elif self.gamma is None and not np.isinf(self.chamber_height):
            self.imped_calc = self._pp_low_frequency

        elif self.gamma is not None and not np.isinf(self.chamber_height):
            self.imped_calc = self._pp_spectrum

        else:
            # WrongCalcError
            raise RuntimeError(
                "method for impedance calculation in CoherentSynchrotronRadiation object "
                "not recognized"
            )

    def _pp_low_frequency(
        self,
        frequency_array: NumpyArray,
        u_max: float = 10.0,
        high_frequency_transition: float = np.inf,
    ):
        """
        Computes the parallel-plates impedance according to eq. 8 of [Chao2011]_. For frequencies
        larger than the cut-off frequencies, it approaches the low-frequency free-space impedance,
        i.e. the impedance increases as :math:`f^{1/3}`.

        Parameters
        ----------
        frequency_array : float, array
            Frequencies at which to compute the impedance.
        u_max : float, optional
            Maximum value of u, for which the summands are evaluated. The Airy functions become
            numerically unstable for u>100. The real part is exponentially suppressed, but
            imaginary part is not and the remaining sum from u_max to infinity is
            estimated analytically. The default is 10.
        high_frequency_transition : float, optional
            Use the low-frequency free-space impedance for frequencies above
            `high_frequency_transition` times cut_off frequency. The default is np.inf.

        Raises
        ------
        ValueError
            Raised if `high_frequency_transition` < 1.

        Yields
        ------
        None.

        """

        # using high_frequency_transition factor of 1 yields unphysical results
        if high_frequency_transition <= 1.0:
            raise ValueError("high frequency transition must be greater than 1")

        n_array = frequency_array / self.f_0

        self.impedance = np.zeros_like(n_array, dtype=complex)

        # parallel plates cut-off frequency in units of f_0
        n_cut = np.sqrt(2 / 3) * (np.pi / self.Delta) ** 1.5

        # use approximate equation for frequencies above high_frequency_transition * n_cut
        approx_indexes = n_array > (high_frequency_transition * n_cut)
        if np.count_nonzero(approx_indexes) > 0:
            self.impedance[approx_indexes] = self._fs_low_frequency(
                frequency_array[approx_indexes]
            )

        # use exact result for all other frequencies (and ignore f=0)
        exact_indexes = np.invert(approx_indexes) * n_array != 0

        # if there is nothing to calculate, we are done ...
        if np.count_nonzero(exact_indexes) == 0:
            return

        n_array = n_array[exact_indexes]  # override for convenience and exclude f=0

        # maximum p(n) to have u(p,n)<u_max(n)
        pMax_array = np.array(
            np.ceil(
                self.Delta
                * n_array ** (2 / 3)
                * np.sqrt(u_max)
                / (2 ** (2 / 3) * np.pi)
                - 0.5
            ),
            dtype=int,
        )

        # maximum p; assumes largest frequency is at last array element
        pMax = pMax_array[-1]

        p_matrix = np.zeros(shape=(len(n_array), pMax), dtype=int)

        # matrix to store the summands
        Z_matrix = np.zeros_like(p_matrix, dtype=complex)

        for nit, n in enumerate(n_array):
            # first element of p_matrix is 1 to ensure evaluation at u_min...
            # ... if n is large enough so that u_min < 100, (i.e. airy(
            # u_min) does not yield np.nan)
            if (
                pMax_array[nit] == 0
                and n > (np.pi / self.Delta) ** 1.5 / np.sqrt(2) / 100**0.75
            ):
                p_matrix[nit, 0] = 1
            else:
                p_matrix[nit, : pMax_array[nit]] = (
                    2 * np.arange(pMax_array[nit]) + 1
                ) ** 2

        # evaluate Airy functions only at these values of p
        indexes = p_matrix > 0

        # argument of the Airy functions
        u_matrix = (
            (np.pi / 2 ** (1 / 3) / self.Delta) ** 2 / n_array ** (4 / 3) * p_matrix.T
        ).T

        # evaluate Airy function only at relevant indexes
        airy_matrix = airy(u_matrix[indexes])  # returns Ai(u), Ai'(u), Bi(u), Bi'(u)

        Ci_matrix = airy_matrix[0] - 1j * airy_matrix[2]
        Ci_prime_matrix = airy_matrix[1] - 1j * airy_matrix[3]

        Z_matrix[indexes] = (
            airy_matrix[1] * Ci_prime_matrix
            + u_matrix[indexes] * airy_matrix[0] * Ci_matrix
        )

        # sum over p
        self.impedance[exact_indexes] = np.sum(Z_matrix, axis=1)

        # the sum in eq. 8 is up to infinity, but we stop at p_max; the real
        # part is exponentially surpressed for large u, but the imaginary
        # part is not; using the assymptotic expression of the imaginary
        # summands for large u, the remaining sum from p_max to infinity can be
        # performed analytically by Mathematica 12.1.0.0.
        self.impedance[exact_indexes] += (
            1j
            * 2 ** (5 / 3)
            / (4096 * np.pi**6)
            * (self.Delta * n_array ** (2 / 3)) ** 5
            * polygamma(4, pMax_array + 1)
        )

        self.impedance[exact_indexes] *= (
            self.Z0 * 4 * np.pi**2 * 2 ** (1 / 3) * 1 / self.Delta / n_array ** (1 / 3)
        )
        # fixed sign of imaginary part
        self.impedance[exact_indexes] = self.impedance[exact_indexes].conj()

    def _pp_spectrum(
        self, frequency_array: NumpyArray, zeta_max: float = 9.0, **kwargs
    ):
        r"""
        Computes the parallel-plates impedance, based on eq. B13 of [Murphy1997]_.

        .. math::
            Z_{pp}(f) = Z_{fs,exact} - Z_{fs,approx} +
            Z_0 \sqrt{8\pi} 3^{2/3} \exp(i \pi/6) \left(\frac{f}{f_0}\right)^{1/3}
            \frac{1}{\alpha(f)}
            \sum_{p=0}^\infty h\left(\frac{1}{2}\left(\frac{\pi(2p+1)}{\alpha(f)}\right)^2 \right)

        Parameters
        ----------
        frequency_array : float array
            Frequency at which to calculate parallel-plates impedance
        zeta_max : float, optional
            Maximum value of zeta, for which :meth:`~_hFun` is evaluated. The result becomes
            numerically unstable for zeta>18. The remaining sum from zeta_max to infinity is
            estimated analytically. The default is 9.
        **kwargs :
            Keyword arguments get passed to :meth:`~_fs_spectrum`

        Returns
        -------
        None.

        """

        non_zero_indexes = frequency_array != 0

        n_array = frequency_array[non_zero_indexes] / self.f_0

        # Murphy et al. convention has plates at +/-h, wheras we use h for the full chamber height
        # (from top to bottom) => we need to use 0.5*self.Delta in their equations

        # based an eq. B8
        alphas = (
            np.sqrt(2)
            * np.exp(-1j * np.pi / 6)
            * n_array ** (2 / 3)
            * 0.5
            * self.Delta
            / 3 ** (1 / 6)
        )

        # sets self.impedance to the full free-space impedance
        self._fs_spectrum(frequency_array, **kwargs)

        # subtract low-frequency free-space impedance
        self.impedance -= self._fs_low_frequency(frequency_array)

        # parallel plates part

        # maximum summation index p(n), such that zeta(p,n) < zeta_max
        pMax_array = np.array(
            np.ceil(
                np.sqrt(zeta_max / 3 ** (1 / 3))
                / np.pi
                * 0.5
                * self.Delta
                * n_array ** (2 / 3)
                - 0.5
            ),
            dtype=int,
        )

        pMax = pMax_array[
            -1
        ]  # maximum p; assumes largest frequency is at last array element

        p_matrix = np.zeros(shape=(len(n_array), pMax), dtype=int)

        # matrix to store the summands
        Z_matrix = np.zeros_like(p_matrix, dtype=complex)

        for nit, n in enumerate(n_array):
            # first element of p_matrix is 1 to ensure evaluation at zeta_min...
            # ... if n is large enough so that zeta_min < zeta_max
            if (
                pMax_array[nit] == 0
                and n > 3**0.25 * (np.pi / (0.5 * self.Delta)) ** 1.5 / zeta_max**0.75
            ):
                p_matrix[nit, 0] = 1
            else:
                p_matrix[nit, : pMax_array[nit]] = (
                    2 * np.arange(pMax_array[nit]) + 1
                ) ** 2

        # evaluate h function only at these values of p
        indexes = p_matrix > 0

        z_matrix = (0.5 * (np.pi / alphas) ** 2 * p_matrix.T).T

        Z_matrix[indexes] = self._hFun(z_matrix[indexes])

        # sum over p
        Z_pp = np.sum(Z_matrix, axis=1)

        # we cut the infinite sum at p_max; using the assymptotic expression of _hFun for large
        # zeta, the remaining sum from p_max to infinity can be performed analytically by
        # Mathematica 12.1.0.0
        Z_pp += (
            np.sqrt(np.pi)
            / (32 * 3 ** (5 / 6))
            * np.exp(1j * np.pi / 6)
            * (0.5 * self.Delta * n_array ** (2 / 3) / np.pi) ** 5
            * polygamma(4, pMax_array + 0.5)
        )

        Z_pp *= (
            self.Z0
            * (8 * np.pi) ** 0.5
            * 3 ** (2 / 3)
            * np.exp(1j * np.pi / 6)
            * n_array ** (1 / 3)
            / alphas
        )
        # fixed sign of imaginary part
        Z_pp = Z_pp.conj()
        self.impedance[non_zero_indexes] += Z_pp

    def _hFun(self, z: NumpyArray):
        r"""
        Implements eq. B14 of [Murphy1997]_.

        The integral in eq. B14 was solved analytically by Mathematica 12.1.0.0. However,
        the analytic solution in terms of Airy functions ist numerically unstable for
        :math:`\zeta>18`, with :math:`z=\exp(i\pi/3)\zeta`. Numerically solving eq. B14,
        gives a negligable contribution for :math:`zeta > 18.0`, and the analytic solution is,
        thus, sufficient for our purposes.

        Parameters
        ----------
        z : complex array
            Argument of the function. Numerically unstable for Re z > 9, but this
            condition is not checked.

        Returns
        -------
        complex array
            h(z)

        """

        airy_array = airy(-z / 12 ** (1 / 3))  # returns Ai(), Ai'(), Bi(), Bi'()
        return (
            -(np.pi**1.5)
            / (2 ** (2 / 3) * 3 ** (5 / 6))
            * (
                z * (airy_array[0] ** 2 + airy_array[2] ** 2) / 12 ** (1 / 3)
                - airy_array[1] ** 2
                - airy_array[3] ** 2
            )
        )

    def _fs_spectrum(
        self,
        frequency_array: NumpyArray,
        epsilon: float = 1e-6,
        low_frequency_transition: float = 1e-5,
        high_frequency_transition: float = 10,
    ) -> None:
        r"""
        Computes the exact free-space synchrotron radiation impedance, based on eqs. A4 and A5 of
        [Murphy1997]_. For computation speed and numerical stability, the approximate expressions
        are used for frequencies lower / higher than the critical frequency `f_crit`.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to evaluate the impedance
        high_frequency_transition : float, optional
            Ratio of f/f_crit above which the high-frequency approximation is used. If it is smaller
            than 1, a `ValueError` is raised. The default is 10.
        low_frequency_transition : float, optional
            Ratio of f/f_crit below which the low-frequency approximation is used. If it is greater
            than 1, a `ValueError` is raised. The default is 0, i.e. the approximation is not used.
        epsilon : float, optional
            The first integral of eq. A5 has an integrable singularity at the upper limit 1,
            which is (currently) not handled by :func:`~scipy.integrate.quad_vec`.
            Therefore, the integration is only up to 1-`epsilon`.

        Raises
        ------
        ValueError
            Raised if
            * `high_frequency_transition` < 1
            * `low_frequency_transition` > 1
            * `high_frequency_transition` < `low_frequency_transition`

        Returns
        -------
        None.

        """

        # check for input consistency
        if high_frequency_transition <= low_frequency_transition:
            raise ValueError(
                "high_frequency_transition ratio must be larger than the "
                + "low_frequency_transition rato"
            )

        if high_frequency_transition < 1.0:
            raise ValueError("high_frequency_transition ratio must be greater than 1")

        if low_frequency_transition > 1.0:
            raise ValueError("low_frequency_transition ratio must be smaller than 1")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be a small positive value")

        self.impedance = np.zeros_like(frequency_array, dtype=complex)

        l_array = frequency_array / self.f_crit

        # use the low frequency approximation where f < LFT * f_c
        # low_indexes = np.where(l_array < low_frequency_transition)[0]
        low_indexes = l_array < low_frequency_transition
        if np.count_nonzero(low_indexes) > 0:
            self.impedance[low_indexes] = self._fs_low_frequency(
                frequency_array[low_indexes]
            )

        # use the high frequency approximation where f > HFT * f_c
        # high_indexes = np.where(l_array > high_frequency_transition)[0]
        high_indexes = l_array > high_frequency_transition
        if np.count_nonzero(high_indexes) > 0:
            self.impedance[high_indexes] = self._fs_high_frequency(
                frequency_array[high_indexes]
            )

        # use full integration for frequencies inbetween
        exact_indexes = np.invert(low_indexes + high_indexes)
        if np.count_nonzero(exact_indexes) == 0:
            return

        # Real part: eq. A4, is solved analytically with Mathematica 12.1.0.0 in terms of
        # generalized hypergeometric functions
        # Imaginary part: quad_vec can't handle the integrable singularity
        # at y=1 for y<1, we need to integrate up to 1-epsilon fixed sign of
        # imaginary part
        self.impedance[exact_indexes] = (
            np.sqrt(3)
            * gamma_func(2 / 3)
            / l_array[exact_indexes] ** (2 / 3)
            / 2 ** (4 / 3)
            * self.hyper_vec(
                [-1 / 3], [-2 / 3, 2 / 3], 0.25 * l_array[exact_indexes] ** 2
            )
            + 81
            * np.pi
            * l_array[exact_indexes] ** (8 / 3)
            / (640 * 2 ** (2 / 3) * gamma_func(-1 / 3))
            * self.hyper_vec(
                [4 / 3], [7 / 3, 8 / 3], 0.25 * l_array[exact_indexes] ** 2
            )
            - 0.25 * np.pi
            - 1j
            * (
                integrate.quad_vec(
                    lambda y: self._fs_integrandImZ1(y, l_array[exact_indexes]),
                    0,
                    1 - epsilon,
                )[0]
                - integrate.quad_vec(
                    lambda y: self._fs_integrandImZ2(y, l_array[exact_indexes]),
                    1,
                    np.inf,
                )[0]
            )
        )

        self.impedance[exact_indexes] *= self.Z0 * self.gamma * l_array[exact_indexes]

    def _fs_low_frequency_wrapper(self, frequency_array: NumpyArray):
        r"""
        Wrapper to compute the free-space low-frequency approximation of the synchrotron
        radiation impedance.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        None.

        """

        self.impedance = self._fs_low_frequency(frequency_array)

    def _fs_low_frequency(self, frequency_array: NumpyArray) -> NumpyArray:
        r"""
        Computes the free-space low-frequency approximation of the synchrotron radiation impedance,
        according to eq. 6.18 of [Murphy1997]_.

        .. math::
          Z_{fs,approx}(f) / Z_0 = \frac{\Gamma\left( 2/3 \right) }{3^{1/3}} e^{i\pi/6}
          \left(\frac{f}{f_0}\right)^{1/3}

        This is a helper function.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        complex array
            impedance
        """
        # fixed sign of imaginary part
        return (
            self.Z0
            * gamma_func(2 / 3)
            / 3 ** (1 / 3)
            * np.exp(-1j * np.pi / 6)
            * (frequency_array / self.f_0) ** (1 / 3)
        )

    def _fs_high_frequency(self, frequency_array: NumpyArray) -> NumpyArray:
        r"""
        Computes the free-space high-frequency approximation of the synchrotron radiation impedance,
        based on eq. 6.20 of [Murphy1997]_. This function is a helper function for
        _fs_full_spectrum.

        Parameters
        ----------
        frequency_array : float array
            Frequencies at which to compute the impedance

        Returns
        -------
        complex array
            impedance
        """
        # fixed sign of imaginary part
        return (
            self.Z0
            * self.gamma
            * (
                np.sqrt(3 * np.pi / 32)
                * np.sqrt(frequency_array / self.f_crit)
                * bm.exp(-frequency_array / self.f_crit)
                + 4j / 9 * self.f_crit / frequency_array
            )
        )

    # todo type hint
    def _fs_integrandReZ(self, x):
        # integrand of real part of free-space impedance
        return kv(5 / 3, x)

    # todo type hint
    def _fs_integrandImZ1(self, y, x):
        # integrand of imaginary part of free-space impedance for y<1
        return np.real(
            bm.exp(-x * y)
            * (
                (1j * y + np.sqrt(1 - y**2)) ** (5 / 3)
                + 1 / (1j * y + np.sqrt(1 - y**2)) ** (5 / 3)
                - 2 * np.sqrt(1 - y**2)
            )
            / (4 * y * np.sqrt(1 - y**2))
        )

    # todo type hint
    def _fs_integrandImZ2(self, y, x):
        # integrand of imaginary part of free-space impedance for y>1
        return (
            bm.exp(-x * y)
            * (
                8 * np.sqrt(y**2 - 1)
                - (y + np.sqrt(y**2 - 1)) ** (5 / 3)
                + 1 / (y + np.sqrt(y**2 - 1)) ** (5 / 3)
            )
            / (8 * y * np.sqrt(y**2 - 1))
        )
