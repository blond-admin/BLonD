# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Functions to plot general and RF paramaters**

:Authors: **Helga Timko**
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray as NumpyArray


def plot_voltage_programme(
    time: NumpyArray,
    voltage: NumpyArray,
    sampling: int = 1,
    dirname: str | PathLike[str] = "fig",
    figno: int = 0,
):
    """
    Plot of the RF voltage as a function of time.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8, 6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(time[::sampling], voltage[::sampling])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"RF voltage [V]")

    # Save figure
    fign = dirname + "/RF_voltage_%d" % figno + ".png"
    plt.savefig(fign)
    plt.clf()
