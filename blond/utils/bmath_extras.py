from __future__ import annotations

from typing import TYPE_CHECKING

from blond.utils import bmath as bm

if TYPE_CHECKING:
    from typing import Tuple
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


def mean_and_std(
    array: NumpyArray | CupyArray, weights: NumpyArray | CupyArray
) -> Tuple[float, float]:
    """Calculate average and standard deviation, considering weights

    Parameters
    ----------
    array
        Array to calculate the statistic functions from
    weights
        Array holding the weight corresponding to each array entry.

    Returns
    -------
    weighted_mean, weighted_std
    """
    # Calculate the weighted mean using numpy.average
    weighted_mean = bm.average(array, weights=weights)

    # Calculate the weighted variance
    # .. generates memory. ideally this function
    #    is written as c++ and cuda kernel to prevent a bigger memory footprint
    diff = array - weighted_mean
    diff *= diff  # diff squared, but recycling the memory
    weighted_std = bm.sqrt(bm.average(diff, weights=weights))
    return float(weighted_mean), float(weighted_std)
