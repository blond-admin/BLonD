# pragma: no cover

import numpy as np

from blond import (
    BiGaussian,
    DriftSimple,
    Simulation,
    StaticProfile,
    WakeField,
    proton,
)
from blond._core.backends.backend import backend
from blond._core.beam.beams import Beam
from blond._core.ring.ring import Ring
from blond.cycles.magnetic_cycle import MagneticCycleBase, MagneticCyclePerTurn
from blond.physics.cavities import MultiHarmonicCavity
from blond.physics.drifts import DriftXSuite
from blond.physics.impedances.solvers import InductiveImpedanceSolver
from blond.physics.impedances.sources import InductiveImpedance


class Main:
    @staticmethod
    def describe_accelerator() -> tuple[Ring, MagneticCyclePerTurn, Beam]:
        # Description of accelerator
        my_ring = Ring(circumference=20)

        profile1 = StaticProfile(cut_left=0, cut_right=1, n_bins=128)
        cavity = MultiHarmonicCavity(
            n_harmonics=10,
            main_harmonic_idx=0,
        )
        cavity.voltage = 1e3 * backend.ones(10, dtype=backend.float)  # TODO
        # should
        # be
        # reasonable
        # value
        cavity.phi_rf = 0 * backend.ones(10, dtype=backend.float)  # TODO
        # should be
        # reasonable
        # value
        cavity.harmonic = backend.ones(10, dtype=backend.float)  # TODO
        # should be
        # reasonable
        # value
        one_turn_execution_order = (
            DriftSimple(
                orbit_length=0.4 * my_ring.circumference, transition_gamma=11
            ),
            cavity,
            WakeField(
                sources=(InductiveImpedance(34.6669349520904 / 10e9),),
                solver=InductiveImpedanceSolver(),
            ),
            profile1,
            # LocalFeedback(cavity1, profile1),
            # GlobalFeedback(profile1),
            # DriftXSuite(orbit_length=0.1 * my_ring.circumference),
        )

        my_cycle = MagneticCyclePerTurn(
            reference_particle=proton,
            values_after_turn=np.linspace(1e9, 3e9, 110),
            value_init=1e9,
        )

        my_beam = Beam(
            intensity=1e6,
            particle_type=proton,
        )

        my_ring.add_elements(one_turn_execution_order, reorder=False)

        return my_ring, my_cycle, my_beam

    @staticmethod
    def ready_simulation_and_beam(
        my_ring: Ring,
        my_cycle: MagneticCycleBase,
        my_beam: Beam,
    ) -> tuple:
        # Preparation of simulation
        # Here everything might be interconnected
        simulation = Simulation(ring=my_ring, magnetic_cycle=my_cycle)
        # Already minor simulation of single turn
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                n_macroparticles=100,
                sigma_dt=1e-9,
                sigma_dE=1e9,
            ),
            beam=my_beam,
        )
        return simulation, my_beam

    @staticmethod
    def run_simulation(
        simulation: Simulation,
        my_beam: Beam,
    ) -> None:
        # Full simulation. everything here should be optimized
        simulation.run_simulation(
            turn_i_init=10, n_turns=100, beams=(my_beam,)
        )


def main() -> None:
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


if __name__ == "__main__":  # pragma: no cover
    main()
