from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

def rms_emittance(density: NumpyArray | CupyArray, dt_grid: NumpyArray | CupyArray, dE_grid: NumpyArray | CupyArray):
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
    #extract sampling of time and energy axes
    dt = dt_grid[1,0] - dt_grid[0,0]
    dE = dE_grid[0,1] - dE_grid[0,0]

    #calculate means and variances along axes
    y_matrix_tomo1, x_matrix_tomo1 = np.meshgrid(np.arange(density.shape[1]),
                                                 np.arange(density.shape[0]))
    xbar = np.sum(density * x_matrix_tomo1)
    xms = np.sum(density * x_matrix_tomo1**2.)
    ybar = np.sum(density * y_matrix_tomo1)
    yms = np.sum(density * y_matrix_tomo1**2.)
    xybar = np.sum(density * x_matrix_tomo1 * y_matrix_tomo1)

    #combine into rms emittance and scale by dt and dE resolution
    rms_emittance = np.pi*dt*dE*np.sqrt((xms - xbar**2.)
                                       * (yms - ybar**2.)
                                       - (xybar - xbar*ybar)**2.)

    return rms_emittance

def ninety_percent_emittance(density: NumpyArray | CupyArray, dt_grid: NumpyArray | CupyArray, dE_grid: NumpyArray | CupyArray) -> float:
    """ Calculates the 90% emittance of a single bunch, based on its density in phase space
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
        emittance
            The calculated emittance
    """

    #extract sampling of time and energy axes
    dt = dt_grid[1,0] - dt_grid[0,0]
    dE = dE_grid[0,1] - dE_grid[0,0]

    cumulative_array = np.cumsum(np.flip(np.sort(density.flatten())))
    n_bins_90 = np.argmin(np.abs(cumulative_array-0.9)) #figure out how many bins are necessary to contain 90% of distribution

    emittance = n_bins_90*dE*dt

    return emittance
