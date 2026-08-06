# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
IQ / polar conversion helpers for the cavity feedbacks.

Small pure-numpy utilities (``cartesian_to_polar``,
``polar_to_cartesian``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def cartesian_to_polar(
    IQ_vector: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """
    Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components.

    Returns
    -------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].
    """
    return np.absolute(IQ_vector), np.angle(IQ_vector)


def polar_to_cartesian(
    amplitude: float | NumpyArray,
    phase: float | NumpyArray,
) -> NumpyArray | complex:
    """
    Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components.
    """
    return amplitude * (np.cos(phase) + 1j * np.sin(phase))
