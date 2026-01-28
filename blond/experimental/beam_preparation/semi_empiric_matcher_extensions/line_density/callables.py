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


def histogram_to_state_vector(
    histogram: NumpyArray, hamilton_2D: NumpyArray
) -> NumpyArray:
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    state_vector = np.zeros(hamilton_2D.shape[0], float)
    print("_gen_state_numba")
    t0 = time.time()
    _gen_state_numba(H_change, hamilton_2D, histogram, mid, state_vector)
    print(time.time() - t0)
    return state_vector


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


def get_test_profile(res, analy=True):
    # Parameters
    mean = 2.5e-9 / 2  # Mean of the distribution
    std_dev = 2.5e-9 / 8  # Standard deviation
    size = 10000  # Number of data points

    if analy:
        hist_x = np.linspace(*(0, 2.5e-9), res)
        hist_y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -((hist_x - mean) ** 2) / (2 * std_dev**2)
        )
    else:
        # Generate random data from a Gaussian distribution
        data = np.random.normal(loc=mean, scale=std_dev, size=size)

        # Get the histogram (density=False for raw counts)
        hist_y, bin_edges = np.histogram(data, bins=512, density=False)
    return hist_y


if __name__ == "__main__":  # TODO debug
    hamilton_2D = np.load(
        "/home/slauber/PycharmProjects/deleteme/blonder/blond"
        "/experimental/beam_preparation/semi_empiric_matcher_extensions/line_density/ham2d_dev.npy"
    )
    histogram = get_test_profile(hamilton_2D.shape[0], analy=True)
    keep_id = None
    if keep_id is not None:
        histogram[keep_id] = 1

    state_vector = histogram_to_state_vector(histogram, hamilton_2D)
    plt.subplot(2, 1, 1)
    plt.plot(state_vector, label="state_vector", alpha=0.1)
    if keep_id is not None:
        keep = float(state_vector[keep_id])
        state_vector[:] = 0
        state_vector[keep_id] = keep
        plt.plot(state_vector, label="state_vector")
    histogram_rec = state_vector_to_histogram(state_vector, hamilton_2D)

    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(histogram, label="histogram", c="C0")
    plt.plot(histogram_rec, label="histogram_rec", c="C1")
    if keep_id is not None:
        plt.plot(keep_id, histogram[keep_id], "o", c="C0")
        plt.plot(keep_id, histogram_rec[keep_id], "x", c="C1")
    plt.legend()
    plt.show()
