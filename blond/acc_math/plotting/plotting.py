# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of plotting routines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def plot_separatrix_single_rf(
    phi_array: NumpyArray, separatrix_array: NumpyArray, **kwargs
) -> None:
    """
    Plot of the separatrix.

    Parameters
    ----------
    phi_array
        Array of phase coordinates at which the separatrix is sampled, in [rad].
    separatrix_array
        Array of values of the separatrix at each point in phi_array.
    **kwargs
        Keyword arguments for ``matplotlib.pyplot.plot``.
    """
    plt.plot(phi_array, separatrix_array, **kwargs)
