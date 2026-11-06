# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Measure the performance of the histogram function.

Notes
-----
Authors:
Leonard Thiele
"""

import matplotlib.pyplot as plt
import numpy as np
from scan_performance import (
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

from blond.core.backends.backend import backend

N_MACROPARTICLES = N_MACROPARTICLES_SCAN
#: Bin counts scanned by the benchmark, in factor-4 steps. The lower
#: end is a coarsely sliced single bunch, the upper end covers the
#: largest profile in use: the multi-turn FCC-ee example slices 1118
#: bunches with 2**10 bins each, i.e. about 1.1e6 bins in total.
N_BINS_ = [2**21, 2**6]


def main() -> None:  # pragma: no cover
    """Measure the performance of the histogram function."""
    for n_bins in N_BINS_:

        def make_kwargs(n_macroparticles: int, n_bins=n_bins) -> dict:
            """Build the `histogram` arguments for `n_macroparticles`."""
            rng = np.random.default_rng(42)
            input_array = (rng.random(n_macroparticles) - 0.5) * 20
            return {
                # casting to correct data type, outside the timed window
                "array_read": backend.array(input_array, dtype=backend.float),
                "array_write": backend.zeros(n_bins, dtype=backend.float),
                "start": -12.0,
                "stop": 8.0,
            }

        plot_performance(
            kernel_name="histogram",
            make_kwargs=make_kwargs,
            scan_values=N_MACROPARTICLES,
            n_warmup=1,
            xlabel="n_macroparticles",
            title=f"histogram, n_bins={n_bins}",
            save_name=f"histogram_{n_bins}bins",
        )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
