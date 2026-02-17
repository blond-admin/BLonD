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
from xpart.longitudinal.rfbucket_matching import (  # ThermalDistribution,; ParabolicDistribution,
    QGaussianDistribution,
)

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    DriftSimple,
    MagneticCyclePerTurn,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.interfaces.xsuite import (
    XsuiteRFBucketMatcher,
)

logging.basicConfig(level=logging.INFO)


def main():
    ring = Ring(26_658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf = 85  # 45*(np.pi/180)

    N_TURNS = int(1)
    energy_init = 450e9
    energy_cycle = MagneticCyclePerTurn(
        value_init=energy_init,
        values_after_turn=np.linspace(energy_init, energy_init, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.transition_gamma = 55.759505
    beam1 = Beam(
        intensity=1e6,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    zmax = ring.circumference / (2 * np.amin(rf_station.harmonic))

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=XsuiteRFBucketMatcher(
            distribution_type=QGaussianDistribution,
            sigma_z=zmax / 4,
            n_macroparticles=int(1e3),
            seed=42,
        ),
    )

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value % 10 != 0:
            return

        plt.scatter(
            beam.read_partial_dt(),
            beam.read_partial_dE(),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            n_turns=N_TURNS,
            observe=[phase_observation],
            beams=[beam1],
        )
    except AssertionError as exc:
        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS,
            observe=[phase_observation, bunch_observation],
        )

    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist(bunch_observation.dts[i, :], bins=20, density=True)
            plt.title(f"Turn {i}")
            plt.xlabel("Time deviation dt [s]")
            plt.ylabel("Number of macroparticles")
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
