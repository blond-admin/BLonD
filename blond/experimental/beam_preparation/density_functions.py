from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def gaussian_density(
    hamilton: NumpyArray | CupyArray, sigma: float
) -> NumpyArray | CupyArray:
    """Turns a hamiltonian into a gaussian density distribution
     with some standard deviation sigma

    Parameters
    ----------
    hamilton
        2D hamiltonian, in [eV]
    sigma
        standard deviation of distribution, in [eV]


    Returns
    -------
    density
        2D density distribution
    """
    density = np.exp(-(hamilton**2) / (2 * sigma**2))
    density /= np.sum(density)  # Normalize density mass function
    return density


def binomial_density(
    hamilton: NumpyArray | CupyArray, bunch_length: float, exponent: float
) -> NumpyArray | CupyArray:
    """Turns a hamiltonian into a binomial density distribution
    with some bunch length and form factor

    Parameters
    ----------
    hamilton
        2D hamiltonian, in [eV]
    bunch_length
        length of distribution, in [eV]
    exponent
        exponent of the distribution


    Returns
    -------
    density
        2D density distribution
    """
    density = (1 - (2.0 * (hamilton) / bunch_length) ** 2) ** (exponent + 0.5)
    select = np.isnan(density)
    density[select] = (
        0.0  # binomial distribution is nan where there is no particles
    )
    density = density / np.sum(density)  # normalize
    return density
