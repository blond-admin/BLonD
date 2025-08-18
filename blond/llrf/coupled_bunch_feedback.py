# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Implementation of Coupled Bunch Feedback**

Heavily based on original mode analysis implementation of A. Lasheen:
(https://gitlab.cern.ch/alasheen/mode_analysis)

:Authors: **Alexandre Lasheen**, **Simon Albright**
"""

from __future__ import annotations

import enum
import collections as coll
import numpy as np
import numpy.fft as npfft
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Optional, Sequence
    from collections import deque

    from numpy.typing import NDArray, ArrayLike

    from blond.beam.profile import Profile


class CBFBModes(enum.Enum):

    DIPOLAR = enum.auto()
    QUADRUPOLAR = enum.auto()


class CoupledBunchFeedback:

    def __init__(self, mode_numbers: ArrayLike[int], phases: ArrayLike[float],
                 gains: ArrayLike[float], n_samples: int, profile: Profile,
                 mode: CBFBModes = CBFBModes.DIPOLAR,
                 max_n_bunch: Optional[int] = None,
                 cba: Optional[CoupledBunchAnalysis] = None,
                 voltage_limit: Optional[float] = None):

        self._mode_numbers = np.array(mode_numbers)
        self._phases = np.array(phases)
        self._gains = np.array(gains)
        self._profile = profile
        self._mode = mode
        self._n_samples = n_samples

        self.total_voltage = np.zeros_like(profile.bin_centers)
        self._bins = np.arange(len(self.total_voltage))/len(self.total_voltage)

        self._max_n = (np.max(mode_numbers) if max_n_bunch is None
                                            else max_n_bunch)

        self.voltage_limit = (voltage_limit if voltage_limit is not None
                                            else np.inf)

        if cba is None:
            self._cba = CoupledBunchAnalysis(self._n_samples, self._profile,
                                             self._max_n, self._mode)
        else:
            self._cba = cba

        if self._cba._mode != self._mode:
            raise ValueError("Feedback mode does not match "
                             +"CoupledBunchAnalysis mode")


    def track(self):

        self.total_voltage *= 0

        self._cba.track()

        for mode in range(self._max_n):
            if mode not in self._mode_numbers:
                continue

            mode_freq = (self._cba._mode_frequencies[mode] + mode*self._max_n)
            mode_phase = self._cba._mode_phases[mode]

            fb_volt = self._gains[mode] * self._cba._mode_amplitudes[mode]

            # TODO: Voltage limit per mode or total?
            if fb_volt > self.voltage_limit:
                fb_volt = self.voltage_limit

            fb_wave = fb_volt * np.sin(2*np.pi * mode_freq
                                       * self._bins
                                       + mode_phase
                                       + self._phases[mode])

            self.total_voltage += fb_wave



class CoupledBunchAnalysis:

    def __init__(self, n_samples: int, profile: Profile, max_n_bunch: int,
                 mode: CBFBModes = CBFBModes.DIPOLAR,
                 n_fft: Optional[int] = None,
                 frequency_limits: Optional[tuple[float]] = None):

        self._n_samples = n_samples
        self._profile = profile

        self._n_fft = n_samples if n_fft is None else n_fft
        self._fft_freqs = npfft.rfftfreq(self._n_fft)

        self._max_n = max_n_bunch

        self._bunch_data = np.zeros([self._max_n, n_samples])

        self._fft_matrix = np.zeros([self._max_n,
                                     self._fft_freqs.shape[0]],
                                     dtype=complex)

        self._mode_amplitudes = np.zeros(self._max_n)
        self._mode_frequencies = np.zeros(self._max_n)
        self._mode_phases = np.zeros(self._max_n)

        self._mode = mode
        match self._mode:
            case CBFBModes.DIPOLAR:
                self._measure = _dipole_measure
            case CBFBModes.QUADRUPOLAR:
                self._measure = _quadrupole_measure

        if frequency_limits is None:
            self._frequency_limits = (0., self._fft_freqs[-1])
        else:
            self._frequency_limits = tuple(frequency_limits)

        self._fft_inds = np.where(
                            (self._fft_freqs>=self._frequency_limits[0])
                            *(self._fft_freqs<=self._frequency_limits[1]))[0]

    @property
    def mode_amplitudes(self):
        return self._mode_amplitudes.copy()

    @property
    def mode_frequencies(self):
        return self._mode_frequencies.copy()

    @property
    def mode_phases(self):
        return self._mode_phases.copy()

    def track(self):

        n_bunch = self._measure(self._profile, self._bunch_data)

        self._motion_fft(n_bunch)
        self._mode_analysis(n_bunch)

    def _motion_fft(self, n_bunch: int):

        for i in range(n_bunch):
            self._fft_matrix[i] = (npfft.rfft(self._bunch_data[i], self._n_fft)
                                * (2/self._n_samples))

    def _mode_analysis(self, n_bunch: int):

        for i_mode in range(self._max_n):

            samp_data_fft_tot = 0

            for j_bunch in range(n_bunch):
                phase_advance = -j_bunch * i_mode * (2*np.pi) / n_bunch
                samp_data_fft_tot += (self._fft_matrix[j_bunch]
                                    * np.exp(1j*phase_advance) / n_bunch)

            amp, freq, phase = calc_amp_freq_phase(self._fft_freqs,
                                                   samp_data_fft_tot,
                                                   self._fft_inds, 0,
                                                   0,
                                                   offset_no_interp = np.pi/2)

            self._mode_amplitudes[i_mode] = amp
            self._mode_frequencies[i_mode] = freq
            self._mode_phases[i_mode] = phase



def calc_amp_freq_phase(data_freq: NDArray, data_fft: NDArray,
                        inds_range: list[int], samp_time: float,
                        ref_time: float, offset_no_interp: float)\
                                                 -> tuple[float, float, float]:
    """Function to calculate the amplitude, frequency and phase from
       the FFT of the oscillations.

    Args:
        data_freq (NDArray): The frequencies of the oscillations
        data_fft (NDArray): The fft of the oscillations
        inds_range (list[int]): The indices of interest
        samp_time (float): The initial time of the acquisition
        ref_time (float): The offset time to apply
        offset_interp (float): The phase offset to use when interpolating.
        offset_no_interp (float): The phase offset to use when not
                                  interpolating.
        interp (bool, optional): Flag to set if interpolation is used.
                                 Defaults to True.

    Returns:
        tuple[float, float, float]: The amplitude, frequency and phase
                                    of the maximum oscillation
    """

    amp = np.max(np.abs(data_fft[inds_range]))

    fft_amp_pos = np.where(np.abs(data_fft[inds_range]) == amp)[0][0]

    data_freq, phase = freq_phase(data_freq, data_fft,
                                  inds_range, fft_amp_pos,
                                  samp_time, ref_time,
                                  offset_no_interp)

    return amp, data_freq, phase

def freq_phase(data_freq: NDArray, data_fft: NDArray, inds_range: list[int],
               init_pos: int, samp_time: float, ref_time: float = 0,
               offset: float = 0) -> tuple[float, float]:
    """Calculate the oscillation frequency and phase without interpolation.

    Args:
        data_freq (NDArray): The frequencies of the FFT
        data_fft (NDArray): The FFT of the oscillations
        inds_range (list[int]): The indexes of interest
        init_pos (int): The approximate peak location
        samp_time (float): The initial time of the acquisition
        ref_time (float): The offset time to apply
                          Defaults to 0.
        offset (float, optional): Phase offset to apply.
                                  Defaults to 0.

    Returns:
        tuple[float, float]: tuple of frequency and phase
    """

    freq = data_freq[inds_range][init_pos]
    phase = (np.angle(data_fft[inds_range][init_pos]
                     * np.exp(1j*2*np.pi * freq * (ref_time-samp_time)))
             + offset)

    return freq, phase

def _linear_correction(time: ArrayLike[float], data: ArrayLike[float]) -> NDArray:
    """Function to compute the linear slope correction.

    Args:
        time (NDArray): The time off the data
        data (NDArray): The bunch data

    Returns:
        NDArray: The correction to apply
    """

    mean = np.mean(data, axis=0)
    fit_lin = curve_fit(_linear, time, mean,
                        p0=[(mean[-1]-mean[0])/(time[-1]-time[0]), 0])[0]

    return _linear(time, *fit_lin)

def _linear(t: float | NDArray,*p: tuple[float, float]) -> float | NDArray:
    a, b = p
    return a*t+b

def _dipole_measure(profile: Profile, queue: NDArray) -> int:
    return _populate_queue(profile.bunchPosition, queue)

def _quadrupole_measure(profile: Profile, queue: NDArray) -> int:
    return _populate_queue(profile.bunchLength, queue)

def _populate_queue(measure: NDArray, queue: NDArray) -> int:

    queue[:, :-1] = queue[:, 1:]
    for i in range(len(measure)):
        queue[i, -1] = measure[i]

    return i