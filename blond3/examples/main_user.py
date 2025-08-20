# pragma: no cover
import numpy as np

from blond3 import (
    SingleHarmonicCavity,
    StaticProfile,
    DriftSimple,
    MagneticCyclePerTurn,
    Beam,
    proton,
    Ring,
    Simulation,
    WakeField,
)
from blond3.beam_preparation.emittance import EmittanceMatcher
from blond3.physics.cavities import MultiHarmonicCavity
from blond3.physics.drifts import DriftXSuite
from blond3.physics.feedbacks.base import LocalFeedback, GlobalFeedback
from blond3.physics.impedances.solvers import InductiveImpedanceSolver
from blond3.physics.impedances.sources import InductiveImpedance


class Main:
    @staticmethod
    def describe_accelerator():
        # Description of accelerator
        my_ring = Ring(circumference=20)

        cavity1 = SingleHarmonicCavity(harmonic=1)
        profile1 = StaticProfile(cut_left=0, cut_right=1, n_bins=128)
        one_turn_execution_order = (
            DriftSimple(orbit_length=0.4 * my_ring.circumference),
            cavity1,
            DriftSimple(orbit_length=0.5 * my_ring.circumference),
            MultiHarmonicCavity(n_harmonics=1, main_harmonic_idx=0),
            WakeField(
                sources=(InductiveImpedance(34.6669349520904 / 10e9),),
                solver=InductiveImpedanceSolver(),
            ),
            profile1,
            # LocalFeedback(cavity1, profile1),
            GlobalFeedback(profile1),
            DriftXSuite(orbit_length=0.1 * my_ring.circumference),
        )

        my_cycle = MagneticCyclePerTurn(
            reference_particle=proton,
            values_after_turn=np.linspace(1e9, 3e9, 50),
            value_init=1e9,
        )

        my_beam = Beam(
            n_particles=1e6,
            particle_type=proton,
        )

        my_ring.add_elements(one_turn_execution_order, reorder=False)

        return my_ring, my_cycle, my_beam

    @staticmethod
    def ready_simulation_and_beam(my_ring, my_cycle, my_beam):
        # Preparation of simulation
        # Here everything might be interconnected
        simulation = Simulation(ring=my_ring, magnetic_cycle=my_cycle)
        # Already minor simulation of single turn
        simulation.prepare_beam(
            preparation_routine=EmittanceMatcher(
                some_emittance=10,
                n_macroparticles=1e6,
            ),
            beam=my_beam,
        )
        return simulation, my_beam

    @staticmethod
    def run_simulation(simulation, my_beam):
        # Full simulation. everything here should be optimized
        results = simulation.run_simulation(
            turn_i_init=10, n_turns=100, beams=(my_beam,)
        )


def main():
    my_ring, my_cycle, my_beam = Main.describe_accelerator()
    simulation, my_beam = Main.ready_simulation_and_beam(
        my_ring=my_ring,
        my_cycle=my_cycle,
        my_beam=my_beam,
    )
    Main.run_simulation(
        simulation=simulation,
        my_beam=my_beam,
    )


if __name__ == "__main__":
    main()
