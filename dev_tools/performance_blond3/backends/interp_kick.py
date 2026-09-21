# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `kick_interpolated`."""

import matplotlib.pyplot as plt
from scan_performance import (
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

from blond.core.backends.backend import backend

N_MACROPARTICLES = N_MACROPARTICLES_SCAN
N_BINS = 128


def make_kwargs(n_macroparticles: int) -> dict:
    """Build the `kick_interpolated` arguments for `n_macroparticles`."""
    bin_centers = backend.linspace(-4, 4, N_BINS, dtype=backend.float)
    return {
        "dt": backend.linspace(-5, 5, n_macroparticles, dtype=backend.float),
        "dE": backend.zeros(n_macroparticles, dtype=backend.float),
        "voltage": bin_centers**2,
        "bin_centers": bin_centers,
        "charge": 10.0,
        "acceleration_kick": 0.0,
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of `kick_interpolated`."""
    plot_performance(
        kernel_name="kick_interpolated",
        make_kwargs=make_kwargs,
        scan_values=N_MACROPARTICLES,
        n_warmup=3,
        xlabel="n_macroparticles",
        title=f"kick_interpolated, n_bins={N_BINS}",
        save_name="kick_interpolated",
    )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
