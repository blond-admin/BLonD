# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `drift_simple` and `drift_exact`."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from blond.core.backends.backend import backend

# `dev_tools` is deliberately not a package (it would be picked up by
# setuptools' package discovery), so import the helpers by path.
sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))
from scan_performance import (  # noqa: E402
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

N_MACROPARTICLES = N_MACROPARTICLES_SCAN

# LHC-like parameters: revolution period in s, energies in eV.
REVOLUTION_PERIOD = 8.89e-5
ETA_0 = 3.18e-4
ALPHA_0 = 3.19e-4
HIGHER_ALPHA = [1e-6, 2e-8]
BETA = 0.99999786
ENERGY = 450e9


def _phase_space(n_macroparticles: int) -> dict:
    """Build `dt` [s] and `dE` [eV] of a bunch-like distribution."""
    return {
        "dt": backend.linspace(
            -1e-9, 1e-9, n_macroparticles, dtype=backend.float
        ),
        "dE": backend.linspace(
            -1e8, 1e8, n_macroparticles, dtype=backend.float
        ),
    }


def make_kwargs_simple(n_macroparticles: int) -> dict:
    """Build the `drift_simple` arguments for `n_macroparticles`."""
    return {
        **_phase_space(n_macroparticles),
        "T": REVOLUTION_PERIOD,
        "eta_0": ETA_0,
        "beta": BETA,
        "energy": ENERGY,
    }


def make_kwargs_exact(n_macroparticles: int) -> dict:
    """Build the `drift_exact` arguments for `n_macroparticles`."""
    return {
        **_phase_space(n_macroparticles),
        "T": REVOLUTION_PERIOD,
        "alpha_0": ALPHA_0,
        "higher_alpha": backend.array(HIGHER_ALPHA, dtype=backend.float),
        "beta": BETA,
        "energy": ENERGY,
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of the drifts."""
    plot_performance(
        kernel_name="drift_simple",
        make_kwargs=make_kwargs_simple,
        scan_values=N_MACROPARTICLES,
        n_warmup=3,
        n_runs=20,
        xlabel="n_macroparticles",
        save_name="drift_simple",
    )
    plot_performance(
        kernel_name="drift_exact",
        make_kwargs=make_kwargs_exact,
        scan_values=N_MACROPARTICLES,
        n_warmup=3,
        n_runs=20,
        xlabel="n_macroparticles",
        title=f"drift_exact, {len(HIGHER_ALPHA)} higher-order alphas",
        save_name="drift_exact",
    )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
