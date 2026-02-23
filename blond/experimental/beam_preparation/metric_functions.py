# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def rms_emittance(
    density: NumpyArray,
    dt_grid: NumpyArray,
    dE_grid: NumpyArray,
    multiply_by_pi: bool = False,
):
    """Calculates the RMS emittance of a single bunch,
     based on its density in phase space

    Parameters
    ----------
    density
        2D phase space mass distribution.
    dt_grid
        Time coordinates of the distribution, in [s].
    dE_grid
        Energy coordinates of the distribution, in [eV].
    multiply_by_pi
        whether to multiply the rms emittance by a factor pi.
        This is how rms emittance is defined in some tomography codes


    Returns
    -------
    rms_emittance
        The calculated emittance
    """

    # calculate means and variances along axes
    xbar = np.sum(density * dt_grid)
    xms = np.sum(density * dt_grid**2.0)
    ybar = np.sum(density * dE_grid)
    yms = np.sum(density * dE_grid**2.0)
    xybar = np.sum(density * dt_grid * dE_grid)

    # combine into rms emittance and scale by dt and dE resolution
    rms_emittance = np.sqrt(
        (xms - xbar**2.0) * (yms - ybar**2.0) - (xybar - xbar * ybar) ** 2.0
    )

    if multiply_by_pi:
        rms_emittance *= np.pi

    return rms_emittance


def q_percent_emittance(
    density: NumpyArray,
    dt_grid: NumpyArray,
    dE_grid: NumpyArray,
    q: float = 0.9,
) -> float:
    """Calculates the phase space area occupied
     by a fraction q of a distribution

    Notes
    -----
    The basic calculation is the same as
    in the LongitudinalTomography/tomographyv3 library

    Parameters
    ----------
    density
        2D phase space mass distribution.
    dt_grid
        Time coordinates of the distribution, in [s].
    dE_grid
        Energy coordinates of the distribution, in [eV].
    q
        percentage of density to enclose


    Returns
    -------
    emittance
        The calculated emittance
    """

    # extract sampling of time and energy axes
    dt = dt_grid[1, 0] - dt_grid[0, 0]
    dE = dE_grid[0, 1] - dE_grid[0, 0]

    cumulative_array = np.cumsum(np.flip(np.sort(density.flatten())))
    n_bins = np.argmin(
        np.abs(cumulative_array - q)
    )  # figure out how many bins are necessary to contain a fraction q of distribution

    emittance = n_bins * dE * dt

    return emittance
