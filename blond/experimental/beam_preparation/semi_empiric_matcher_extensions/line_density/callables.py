from __future__ import annotations

import time
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables_numba import (
    _gen_density_numba,
    _gen_hist_numba,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def state_vector_to_histogram(
    state_vector: NumpyArray, hamilton_2D: NumpyArray
) -> NumpyArray:
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    histogram = np.zeros(hamilton_2D.shape[0], float)
    _gen_hist_numba(H_change, hamilton_2D, histogram, mid, state_vector)
    return histogram


def state_vector_to_hammilton_coordinates(
    state_vector: NumpyArray, hamilton_2D: NumpyArray
) -> NumpyArray:
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    density = np.zeros(hamilton_2D.shape, float)
    _gen_density_numba(H_change, density, hamilton_2D, mid, state_vector)
    return density
