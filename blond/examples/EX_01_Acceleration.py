# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    BunchObservation,
    CavityPhaseObservation,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.experimental.beam_preparation.empiric_matcher import EmpiricMatcher

logging.basicConfig(level=logging.INFO)


def main():
    ring = Ring(26658.883)

    cavity1 = SingleHarmonicCavity()
    cavity1.harmonic = 35640
    cavity1.voltage = 6e6
    cavity1.phi_rf = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.transition_gamma = 55.759505
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()
    BIGAUS = True
    if BIGAUS:
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=1e3,
            ),
        )
    else:  # pragma: no cover
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=EmpiricMatcher(
                grid_base_dt=np.linspace(0, 2.5e-9, 100),
                grid_base_dE=np.linspace(
                    -(777538700.0 * 2), 777538700.0 * 2, 100
                ),
                n_macroparticles=1e6,
                seed=0,
                maxiter_intensity_effects=0,
            ),
        )

    phase_observation = CavityPhaseObservation(
        each_turn_i=1,
        cavity=cavity1,
    )
    bunch_observation = BunchObservation(each_turn_i=1)

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
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
        print(f"Loaded {phase_observation.common_name}")
    except (FileNotFoundError, AssertionError) as exc:
        sim.run_simulation(
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
            # callback=custom_action,
        )
    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                bins=256,
                range=[[0, 2.5e-9], [-4e8, 4e8]],
            )
            plt.draw()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
