# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Testing the performance of `kick_multi_harmonic` and `kick_single_harmonic`."""

import matplotlib.pyplot as plt
from scan_performance import (
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

from blond.core.backends.backend import backend

N_MACROPARTICLES = N_MACROPARTICLES_SCAN
N_RF = 2


def make_kwargs(n_macroparticles: int) -> dict:
    """Build the `kick_multi_harmonic` arguments for `n_macroparticles`."""
    dt = backend.linspace(-5, 5, n_macroparticles, dtype=backend.float)
    return {
        "dt": dt,
        "dE": backend.zeros(n_macroparticles, dtype=backend.float),
        "voltage": backend.linspace(1, 5, N_RF, dtype=backend.float),
        "omega_rf": backend.linspace(1, 5, N_RF, dtype=backend.float),
        "phi_rf": backend.linspace(1, 5, N_RF, dtype=backend.float),
        "charge": 2.0,
        "n_rf": N_RF,
        "acceleration_kick": 0.0,
    }


def make_kwargs_single_harmonic(n_macroparticles: int) -> dict:
    """Build the `kick_single_harmonic` arguments for `n_macroparticles`."""
    return {
        "dt": backend.linspace(-5, 5, n_macroparticles, dtype=backend.float),
        "dE": backend.zeros(n_macroparticles, dtype=backend.float),
        "voltage": 3.0,
        "omega_rf": 2.0,
        "phi_rf": 0.5,
        "charge": 1.0,
        "acceleration_kick": 0.0,
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of the RF kicks."""
    plot_performance(
        kernel_name="kick_multi_harmonic",
        make_kwargs=make_kwargs,
        scan_values=N_MACROPARTICLES,
        n_warmup=10,
        xlabel="n_macroparticles",
        title=f"kick_multi_harmonic, n_rf={N_RF}",
        save_name="kick_multi_harmonic",
    )
    plot_performance(
        kernel_name="kick_single_harmonic",
        make_kwargs=make_kwargs_single_harmonic,
        scan_values=N_MACROPARTICLES,
        n_warmup=10,
        xlabel="n_macroparticles",
        save_name="kick_single_harmonic",
    )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
