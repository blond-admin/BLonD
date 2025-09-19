# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt
from xpart.longitudinal.rfbucket_matching import (
    ThermalDistribution,
)

from blond import (
    Beam,
    BunchObservation,
    CavityPhaseObservation,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.interefaces.xsuite.beam_preparation.rfbucket_matching import (
    XsuiteRFBucketMatcher,
)

logging.basicConfig(level=logging.INFO)


def main():
    ring = Ring(26658.883)

    cavity1 = SingleHarmonicCavity()
    cavity1.harmonic = 35640
    cavity1.voltage = 6e6
    cavity1.phi_rf = 0

    N_TURNS = int(10)
    energy_init = 10e9
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
        n_particles=1e6,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    zmax = ring.circumference / (2 * np.amin(cavity1.harmonic))

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=XsuiteRFBucketMatcher(
            distribution_type=ThermalDistribution,
            energy_init=energy_init,
            cavity=cavity1,
            sigma_z=zmax / 2,
            n_macroparticles=int(1e6),
        ),
    )

    phase_observation = CavityPhaseObservation(
        each_turn_i=1,
        cavity=cavity1,
    )
    bunch_observation = BunchObservation(each_turn_i=1)

    def custom_action(simulation: Simulation):
        if simulation.turn_i.value % 10 != 0:
            return

        plt.scatter(
            simulation.beams[0].read_partial_dt(),
            simulation.beams[0].read_partial_dE(),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=[phase_observation],
        )
    except FileNotFoundError as exc:
        sim.run_simulation(
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=[phase_observation, bunch_observation],
        )

    ANIMATE = True
    if ANIMATE:
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                bins=256,
                # range=[[0, 2.5e-9], [-4e8, 4e8]],
            )
            plt.draw()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
