# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover
import logging
import sys
from pstats import SortKey

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    BeamObservationOncePerTurn,
    DriftSimple,
    RfStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    backend,
    proton,
)
from blond.core.beam.beams import EmptyBeam
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn

logging.basicConfig(level=logging.INFO)

PROFILING = False


def main():
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRfStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf = 0

    N_TURNS = int(1e6)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 7e12, N_TURNS),
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

    phase_observation = RfStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1, beam=beam1)

    beam1.reference.total_energy = sim.magnetic_cycle.get_total_energy_init(
        turn_i_init=0, t_init=0, particle_type=beam1.particle_type
    )

    with backend.temporary_specials_mode("python"):
        if PROFILING:
            sim.profiling(
                beams=(beam1,),
                turn_i_init=0,
                profile_n_turns=1e5,
                sortby=SortKey.TIME,
                # callback=custom_action,
            )
            sys.exit(0)
        sim.run_simulation(
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
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
