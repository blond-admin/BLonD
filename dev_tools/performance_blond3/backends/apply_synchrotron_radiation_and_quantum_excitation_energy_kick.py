# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Testing the performance of `apply_synchrotron_radiation_and_quantum_excitation_energy_kick`.

Note: `dE` is initialised with zeros, so the damping term contributes nothing
to the floating-point work — this benchmark primarily stresses the
quantum-excitation noise generator and the memory bandwidth on `beam_dE`.
"""

from functools import partial

import matplotlib.pyplot as plt
from scan_performance import (
    N_MACROPARTICLES_SCAN,
    plot_performance,
)

from blond.core.backends.backend import backend

N_MACROPARTICLES = N_MACROPARTICLES_SCAN


def make_kwargs(
    n_macroparticles: int, disable_quantum_excitation: bool
) -> dict:
    """Build the kernel arguments for `n_macroparticles`."""
    return {
        "beam_dE": backend.zeros(n_macroparticles, dtype=backend.float),
        "energy_lost": 1.0e3,
        "longitudinal_damping_time": 100.0,
        "natural_energy_spread": 1.0e-3,
        "total_energy": 1.0e9,
        "disable_quantum_excitation": disable_quantum_excitation,
    }


def main() -> None:  # pragma: no cover
    """Testing the performance of `apply_synchrotron_radiation_and_quantum_excitation_energy_kick`."""
    for disable_quantum_excitation in (False, True):
        plot_performance(
            kernel_name="apply_synchrotron_radiation_and_quantum_excitation_energy_kick",
            make_kwargs=partial(
                make_kwargs,
                disable_quantum_excitation=disable_quantum_excitation,
            ),
            scan_values=N_MACROPARTICLES,
            n_warmup=3,
            xlabel="n_macroparticles",
            title="SR + QE energy kick, "
            f"disable_quantum_excitation={disable_quantum_excitation}",
            save_name="synchrotron_radiation_"
            + ("without" if disable_quantum_excitation else "with")
            + "_quantum_excitation",
        )
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
