# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `beam_phase`."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blond.core.backends.backend import backend

# `dev_tools` is deliberately not a package (it would be picked up by
# setuptools' package discovery), so import the helpers by path.
sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))
from scan_performance import plot_performance  # noqa: E402

N_BINS = [2**exponent for exponent in range(8, 22, 2)]


def make_kwargs(n_bins: int) -> dict:
    """Build the `beam_phase` arguments for `n_bins`."""
    rng = np.random.default_rng(42)
    hist_x = np.linspace(0, 1, n_bins)
    hist_y = rng.standard_normal(n_bins)
    return {
        "hist_x": backend.array(hist_x, dtype=backend.float),
        "hist_y": backend.array(hist_y, dtype=backend.float),
        "alpha": 1.4,
        "omega_rf": 1.4,
        "phi_rf": 1.4,
        "bin_size": float(hist_x[1] - hist_x[0]),
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of `beam_phase`."""
    plot_performance(
        kernel_name="beam_phase",
        make_kwargs=make_kwargs,
        scan_values=N_BINS,
        n_warmup=3,
        n_runs=100,
        xlabel="n_bins",
        save_name="beam_phase",
    )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
