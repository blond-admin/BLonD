# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from numba import njit


@njit
def inplace_partition_bool(arr):
    """
    Move all True entries to the end of the array.

    Parameters
    ----------
    arr
        1D array to be sorted.

    Returns
    -------
    Index
        Index that divides the array into
        ``arr[:cutoff] == False`` and
        ``arr[cutoff:] == True``.
    """
    i = 0
    j = arr.size - 1

    while i <= j:
        if not arr[i]:
            i += 1
        else:
            # arr[i] is True, swap with arr[j]
            arr[i], arr[j] = arr[j], arr[i]
            j -= 1
    return j + 1

def test():
    import numpy as np

    # Example: 1% True values in a large array
    n = 100_000
    arr = np.random.rand(n) < 0.01  # ~1% True
    arr2 = arr
    cutoff = inplace_partition_bool(arr)

    assert np.all(arr[:cutoff] == False)
    assert np.all(arr[cutoff:] == True)
    arr.resize((cutoff,), refcheck=False)
    assert np.all(arr[:] == False)
    assert np.all(arr2[:] == False)
    print(arr.shape)

if __name__ == "__main__":
    test()
