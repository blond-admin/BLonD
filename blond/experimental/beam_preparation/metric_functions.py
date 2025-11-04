from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def rms_emittance(
    density: NumpyArray | CupyArray,
    dt_grid: NumpyArray | CupyArray,
    dE_grid: NumpyArray | CupyArray,
):
    """Calculates the RMS emittance of a single bunch, based on its density in phase space
    Notes
    -----
    The basic calculation is the same as in the LongitudinalTomography/tomographyv3 library to ensure consistency

    Parameters
    ----------
    density
        2D phase space mass distribution.
    dt_grid
        Time coordinates of the distribution, in [s].
    dE_grid
        Energy coordinates of the distribution, in [eV].


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
    rms_emittance = np.pi * np.sqrt(
        (xms - xbar**2.0) * (yms - ybar**2.0) - (xybar - xbar * ybar) ** 2.0
    )

    return rms_emittance


def q_percent_emittance(
    density: NumpyArray | CupyArray,
    dt_grid: NumpyArray | CupyArray,
    dE_grid: NumpyArray | CupyArray,
    q: float = 0.9,
) -> float:
    """Calculates the phase space area occupied by a fraction q of a distribution
    Notes
    -----
    The basic calculation is the same as in the LongitudinalTomography/tomographyv3 library to ensure consistency

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
