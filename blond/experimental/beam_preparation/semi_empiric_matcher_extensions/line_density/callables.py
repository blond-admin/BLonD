# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables_numba import (
    _gen_density_numba,
    _gen_hist_numba,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def occupation_per_equipotential_to_density(
    occupation_per_equipotential: NumpyArray,
    potential_2D: NumpyArray,
    density_write: NumpyArray,
) -> None:
    """
    Transform a state vector into a density distribution.

    The state vector defines the density on each orbit of `potential_2D`.

    Parameters
    ----------
    occupation_per_equipotential
        The vector defining the density of each orbit.
    potential_2D
        The Hamiltonian potential that defines the orbits.
    density_write
        The density distribution will be written to this array.
        The density distribution according to the `potential_2D`.

    See Also
    --------
    occupation_per_equipotential_to_histogram: The equivalent to ``density.sum(axis=1)``.
    """

    mid = potential_2D.shape[1] // 2

    # Precompute gradient
    H_1d = potential_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    _gen_density_numba(
        H_change,
        density_write,
        potential_2D,
        mid,
        occupation_per_equipotential,
    )


def occupation_per_equipotential_to_histogram(
    occupation_per_equipotential: NumpyArray, potential_2D: NumpyArray
) -> NumpyArray:
    """
    Transform a state vector into a histogram.

    Following operations are implicitly done.
    1. The state vector is transformed into a density distribution according to the `potential_2D`.
    2. A histogram of the density distribution is obtained.

    Parameters
    ----------
    occupation_per_equipotential
        The vector defining the density of each orbit.
    potential_2D
        The Hamiltonian potential that defines the orbits.

    Returns
    -------
    histogram
        The histogram of the underlying density distribution.

    See Also
    --------
    occupation_per_equipotential_to_density: Obtain the underlying density distribution.
    """

    mid = potential_2D.shape[1] // 2

    # Precompute gradient
    H_1d = potential_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    histogram = np.zeros(potential_2D.shape[0], float)
    assert potential_2D.shape[0] == len(occupation_per_equipotential)
    _gen_hist_numba(
        H_change, potential_2D, histogram, mid, occupation_per_equipotential
    )
    return histogram
