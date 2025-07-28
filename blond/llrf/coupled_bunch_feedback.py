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
                 n_fft: Optional[int] = None):

        self.mode_numbers = np.array(mode_numbers)
        self.phases = np.array(phases)
        self.gains = np.array(gains)
        self.profile = profile
        self.mode = mode
        self.n_samples = n_samples

        self._sample_turns = np.arange(n_samples)
        self._n_fft = n_samples if n_fft is None else n_fft
        self._max_n = (np.max(mode_numbers) if max_n_bunch is None
                                            else max_n_bunch)

        self._bunch_data = np.zeros([self._max_n, n_samples])

        self._fft_matrix = np.zeros([self._max_n, n_samples],
                                    dtype=complex)

        match mode:
            case CBFBModes.DIPOLAR:
                self._measure = _dipole_measure
            case CBFBModes.QUADRUPOLAR:
                self._measure = _quadrupole_measure


    def track(self):

        self._measure(self.profile, self._bunch_data)

        for i, b in enumerate(self._bunch_data):

            correction = _linear_correction(self.profile.bin_centers, b)

            self._fft_matrix[i] = (npfft.rfft(b-correction, self._n_fft)
                                   * 2/self.n_samples)


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


def _dipole_measure(profile: Profile, queue: NDArray):

    positions = profile.bunchPosition
    for i, q in enumerate(queue):
        q[:-1] = q[1:]
        q[-1] = positions[i]

def _quadrupole_measure(profile: Profile, queue: NDArray):

    lengths = profile.bunchLength
    for i, q in enumerate(queue):
        q[:-1] = q[1:]
        q[-1] = lengths[i]
