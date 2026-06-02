# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""A minimum working example of how to start a simulation with BLonD."""

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.testing import pytest_active
from blond.utilities.separatrix.symbolic_separatrix import (
    SymbolicSeparatrixHelper,
)

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

n_turns = 10_000
n_macroparticles = 1e6


def main():
    ring = Ring(26658.883)  # general definition of ring
    rf_station_1 = SingleHarmonicRFStation(
        harmonic=35640, voltage=6e6, phi_rf=np.deg2rad(-10)
    )
    drift1 = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    ring.add_elements(
        [rf_station_1, drift1]
    )  # add elements that resemble one turn

    # Define the ramp
    n_turns = 10_000
    magnetic_cycle = MagneticCyclePerTurn.init_from_linspace(
        reference_particle=proton,
        values=np.linspace(
            450e9, 450e9 + rf_station_1.voltage / 10 * n_turns, n_turns + 1
        ),
    )

    # Define the general beam properties
    beam1 = Beam(intensity=1e9, particle_type=proton)

    # Assemble simulation, will trigger late-init processes that link the
    # objects together
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    sim.print_one_turn_execution_order()

    # As the physics case is defined in the simulation,
    # the beam can be populated with particles according to the separatrix.
    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=0.1e-9,
            n_macroparticles=n_macroparticles,
            reinsertion=True,
        ),
    )
    beam1.plot_hist2d()
    sep_helper = SymbolicSeparatrixHelper.from_simulation(
        simulation=sim,
    )
    sep_helper.plot_separatrix(beam=beam1, zorder=10)
    plt.show()

    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.title("Beam before simulation")
    beam1.plot_hist2d()

    # Artificially introduce offset to show filamentation
    dts = beam1.write_partial_dt()
    dts += 0.05e-9

    sim.run_simulation(
        beams=(beam1,),
        n_turns=n_turns,
    )
    plt.figure(0)
    plt.subplot(2, 1, 2)
    plt.title("Beam after simulation")
    beam1.plot_hist2d()
    plt.tight_layout()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
