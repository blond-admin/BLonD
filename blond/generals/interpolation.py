# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Interpolation routines that resemble the `np.interp` arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def interp_linear(x: NumpyArray, xp: NumpyArray, fp: NumpyArray) -> NumpyArray:
    """
    Perform linear interpolation.

    This function uses NumPy's `interp` function to perform linear interpolation
    for a set of input data points.

    Parameters
    ----------
    x : NumpyArray
        The x-values at which to evaluate the interpolation.
    xp : NumpyArray
        The x-values of the known data points.
    fp : NumpyArray
        The y-values of the known data points.

    Returns
    -------
    NumpyArray
        The interpolated y-values corresponding to the input `x` values.
    """
    return np.interp(x, xp, fp)


def interp_makima(x: NumpyArray, xp: NumpyArray, fp: NumpyArray) -> NumpyArray:
    """
    Perform interpolation using the MAKIMA method (Modified Akima Interpolation).

    This function uses the Akima1DInterpolator from SciPy to perform interpolation
    using the MAKIMA method, which provides a smoother interpolation than linear.

    Parameters
    ----------
    x : NumpyArray
        The x-values at which to evaluate the interpolation.
    xp : NumpyArray
        The x-values of the known data points.
    fp : NumpyArray
        The y-values of the known data points.

    Returns
    -------
    NumpyArray
        The interpolated y-values corresponding to the input `x` values.
    """
    interpolator = Akima1DInterpolator(xp, fp, method="makima")
    return interpolator(x)


def interp_pchip(x: NumpyArray, xp: NumpyArray, fp: NumpyArray) -> NumpyArray:
    """
    Perform interpolation using the PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) method.

    This function uses the PchipInterpolator from SciPy to perform interpolation
    using the PCHIP method, which ensures monotonicity and reduces oscillations.

    Parameters
    ----------
    x : NumpyArray
        The x-values at which to evaluate the interpolation.
    xp : NumpyArray
        The x-values of the known data points.
    fp : NumpyArray
        The y-values of the known data points.

    Returns
    -------
    NumpyArray
        The interpolated y-values corresponding to the input `x` values.
    """
    interpolator = PchipInterpolator(xp, fp)
    return interpolator(x)
