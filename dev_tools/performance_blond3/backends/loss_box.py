# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `loss_box`."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blond.core.backends.backend import backend

# `dev_tools` is deliberately not a package (it would be picked up by
# setuptools' package discovery), so import the helpers by path.
sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))
from scan_performance import (  # noqa: E402
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

N_MACROPARTICLES = N_MACROPARTICLES_SCAN


def make_kwargs(n_macroparticles: int) -> dict:
    """Build the `loss_box` arguments for `n_macroparticles`."""
    # The box keeps the inner 80 % of the particles in both planes. The
    # limits are backend floats: the cuda wrapper asserts that type.
    return {
        "e_max": backend.float(0.8),
        "e_min": backend.float(-0.8),
        "t_min": backend.float(-0.8),
        "t_max": backend.float(0.8),
        "dt": backend.linspace(-1, 1, n_macroparticles, dtype=backend.float),
        "dE": backend.linspace(1, -1, n_macroparticles, dtype=backend.float),
        "flags": backend.zeros(n_macroparticles, dtype=np.int32),
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of `loss_box`."""
    plot_performance(
        kernel_name="loss_box",
        make_kwargs=make_kwargs,
        scan_values=N_MACROPARTICLES,
        n_warmup=3,
        n_runs=20,
        xlabel="n_macroparticles",
        save_name="loss_box",
    )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
