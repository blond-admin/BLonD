# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    DriftObservation,
    DriftSimple,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SimulationObservation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.core.beam.beams import EmptyBeam
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 950e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    observe_simulation = SimulationObservation(each_turn_i=1)
    observe_rf = RFStationPhaseObservation(
        each_turn_i=1, rf_station=rf_station
    )
    observe_drift = DriftObservation(each_turn_i=1, drift=drift1)

    sim.run_simulation(
        EmptyBeam(particle_type=energy_cycle.reference_particle),
        observe=(observe_simulation, observe_rf, observe_drift),
    )
    plt.subplot(3, 1, 1)
    plt.plot(observe_rf.turns_array, observe_simulation.t_revs)
    plt.ylabel("t_revs")
    plt.subplot(3, 1, 2)
    plt.plot(observe_rf.turns_array, observe_rf.omegas)
    plt.ylabel("omegas")
    plt.subplot(3, 1, 3)
    plt.plot(observe_rf.turns_array, observe_drift.eta_0s)
    plt.ylabel("eta_0")
    plt.tight_layout()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
