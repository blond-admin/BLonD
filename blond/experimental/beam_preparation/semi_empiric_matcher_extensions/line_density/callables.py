# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables_numba import (
    _gen_density_numba,
    _gen_hist_numba,
    _gen_state_numba,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def state_vector_to_density(
    state_vector: NumpyArray, hamilton_2D: NumpyArray
) -> NumpyArray:
    """
    Transform a state vector into a histogram.

    Parameters
    ----------
    state_vector
        The vector defining the density of each orbit.
    hamilton_2D
        The Hamiltonian potential that defines the orbits.

    Returns
    -------
    density
        The density distribution according to the `hamilton_2D`.

    See Also
    --------
    state_vector_to_histogram: The equivalent to ``density.sum(axis=1)``.
    """
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    density = np.zeros(hamilton_2D.shape, float)
    _gen_density_numba(H_change, density, hamilton_2D, mid, state_vector)
    return density


def state_vector_to_histogram(
    state_vector: NumpyArray, hamilton_2D: NumpyArray
) -> NumpyArray:
    """
    Transform a state vector into a histogram.

    Following operations are implicitly done.
    1. transformed into a density distribution according to the `hamilton_2D`.
    2. a histogram of the density distribution is obtained.

    Parameters
    ----------
    state_vector
        The vector defining the density of each orbit.
    hamilton_2D
        The Hamiltonian potential that defines the orbits.

    Returns
    -------
    histogram
        The histogram of the underlying density distribution.

    See Also
    --------
    state_vector_to_density: Obtain the underlying density distribution.
    """
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    histogram = np.zeros(hamilton_2D.shape[0], float)
    _gen_hist_numba(H_change, hamilton_2D, histogram, mid, state_vector)
    return histogram
