# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Collection of implementations to do statistics.

References
----------
Paula Hickersberger
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def gauss_fit(hist_x: NumpyArray, hist_y: NumpyArray) -> NumpyArray:
    """
    Perform a gaussian fit on a profile with a single bunches.

    Returns the amplitude, the mean and the standard deviation
    of the fitted gaussian curve for each bunch.

    Parameters
    ----------
    hist_x
        X-axis of the histogram to perform the fitting on.
    hist_y
        Y-axis of the histogram to perform the fitting on.

    Returns
    -------
    params
        Amplitude, mean and standard deviation for each bunch
        Shape (n_bunches,).
    """
    return multi_gauss_fit(hist_x, hist_y, n_bunches=1)[0]


def multi_gauss_fit(
    hist_x: NumpyArray, hist_y: NumpyArray, n_bunches: int
) -> NumpyArray:
    """
    Perform a gaussian fit on a profile with multiple bunches.

    Returns the amplitude, the mean and the standard
    deviation of the fitted gaussian curve for each bunch.

    Parameters
    ----------
    hist_x
        X-axis of the histogram to perform the fitting on.
    hist_y
        Y-axis of the histogram to perform the fitting on.
    n_bunches
        Number of bunches in the profile.

    Returns
    -------
    params
        Amplitude, mean and standard deviation for each bunch.
        Shape (n_bunches, 3).
    """
    n_bins_per_bunch = int(len(hist_x) / n_bunches)
    params = np.zeros([n_bunches, 3], dtype=float)

    for bucket_i in range(n_bunches):
        selection = slice(
            bucket_i * n_bins_per_bunch, (bucket_i + 1) * n_bins_per_bunch
        )

        bucket_hist_x = hist_x[selection]
        bucket_hist_y = hist_y[selection]

        p = [
            bucket_hist_y.max(),
            bucket_hist_x[np.argmax(bucket_hist_y)],
            bucket_hist_x[int(2 * n_bins_per_bunch / 4)]
            - bucket_hist_x[int(n_bins_per_bunch / 4)],
        ]

        popt, _ = curve_fit(gauss, bucket_hist_x, bucket_hist_y, p)
        for i in range(3):
            params[bucket_i, i] = popt[i]

    return params


def gauss(
    x: NumpyArray, amplitude: int, center: int, sigma_x: int
) -> NumpyArray:
    r"""
    Calculate the Gauss function.

    .. math::

        A\, e^{\frac{(x - x_0)^2}{2\sigma_x^2}}.

    Parameters
    ----------
    x
        Input array at which points to calculate the gaussian.
    amplitude
        Amplitude of the function.
    center
        Mean.
    sigma_x
        Standard deviation.

    Returns
    -------
    gauss_y
        Values of the Gauss curve.
    """
    return amplitude * np.exp(-((x - center) ** 2) / 2.0 / sigma_x**2)
