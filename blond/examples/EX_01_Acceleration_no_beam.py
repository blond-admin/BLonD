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
    BeamObservationOncePerTurn,
    DriftSimple,
    EmptyBeam,
    MagneticCyclePerTurn,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)

logging.basicConfig(level=logging.INFO)

n_turns = int(1e3)


def main():
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation(voltage=6e6, phi_rf=0, harmonic=35640)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 7e12, n_turns),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.transition_gamma = 55.759505
    beam1 = EmptyBeam(
        intensity=1e9,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)

    beam1.reference.total_energy = sim.magnetic_cycle.get_total_energy_init(
        particle_type=beam1.particle_type
    )

    sim.run_simulation(
        beams=(beam1,),
        n_turns=n_turns,
        observe=(phase_observation, bunch_observation),
        # callback=custom_action,
    )
    plt.figure()
    plt.ylabel("reference_time (s)")
    plt.xlabel("turn")
    plt.plot(bunch_observation.reference_time)
    plt.figure()
    plt.ylabel("T_rev (s)")
    plt.xlabel("turn")
    plt.plot(np.diff(bunch_observation.reference_time))


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
