# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt
from xpart.longitudinal.rfbucket_matching import (
QGaussianDistribution,
#ThermalDistribution,
#ParabolicDistribution,
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
    ring = Ring(26_658.883)

    cavity1 = SingleHarmonicCavity()
    cavity1.harmonic = 35640
    cavity1.voltage = 6e6
    cavity1.phi_rf = 85 #45*(np.pi/180)


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
        n_particles=1e6,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    zmax = ring.circumference / (2 * np.amin(cavity1.harmonic))

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=XsuiteRFBucketMatcher(
            distribution_type=QGaussianDistribution,
            sigma_z=zmax / 4,
            n_macroparticles=int(1e3),
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
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist(
                bunch_observation.dts[i, :],
                bins=20,
                density=True
            )
            plt.title(f"Turn {i}")
            plt.xlabel("Time deviation dt [s]")
            plt.ylabel("Number of macroparticles")
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
