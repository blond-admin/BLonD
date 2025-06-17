import numpy as np

from classes import (
    Cavity,
    Drift,
    LumpedCavities,
    InductiveImpedance,
    LocalFeedback,
    GlobalFeedback,
    EnergyCycle,
    Beam,
    Ring,
    Simulation,
    EmittanceMatcher,
    DriftXSuite,
    Profile,
)
from iteration2.classes import proton


class Main:
    @staticmethod
    def describe_accelerator():
        # Description of accelerator
        cavity1 = Cavity(harmonic=1)
        profile1 = Profile()
        one_turn_execution_order = (
            Drift(share_of_circumference=0.4),
            cavity1,
            Drift(share_of_circumference=0.5),
            LumpedCavities(),
            InductiveImpedance(),
            profile1,
            LocalFeedback(cavity1, profile1),
            GlobalFeedback(profile1),
            DriftXSuite(share_of_circumference=0.1),
        )

        my_cycle = EnergyCycle(beam_energy_by_turn=np.linspace(1e9, 3e9, 50))

        my_beam = Beam(
            n_particles=1e6,
            n_macroparticles=1e6,
            particle_type=proton,
        )

        my_accelerator = Ring(circumference=20)
        my_accelerator.add_elements(one_turn_execution_order, reorder=False)
        my_accelerator.set_energy_cycle(my_cycle)
        my_accelerator.add_beam(beam=my_beam)
        return my_accelerator

    @staticmethod
    def ready_simulation_and_beam(my_accelerator):
        # Preparation of simulation
        # Here everything might be interconnected
        simulation = Simulation(ring=my_accelerator)
        # Already minor simulation of single turn
        simulation.prepare_beam(preparation_routine=EmittanceMatcher(some_emittance=10))
        return simulation

    @staticmethod
    def run_simulation(simulation):
        # Full simulation. everything here should be optimized
        results = simulation.run_simulation(turn_i_init=10, n_turns=100)


def main():
    my_accelerator = Main.describe_accelerator()
    simulation = Main.ready_simulation_and_beam(my_accelerator)
    Main.run_simulation(simulation)


if __name__ == "__main__":
    main()
