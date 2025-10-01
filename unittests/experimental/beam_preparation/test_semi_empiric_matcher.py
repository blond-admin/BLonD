import unittest
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.quiver import Quiver

from blond import Beam, DriftSimple, Simulation
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
    get_hamiltonian_semi_analytic,
)


class TestSemiEmpiricMatcher(unittest.TestCase):
    def test_something(self):
        from blond.testing.simulation import SimulationTwoRfStations

        sim = SimulationTwoRfStations()
        self._test_matching(sim)

    def test_something2(self):
        from blond.testing.simulation import SimulationTwoRfStationsWithWake

        sim = SimulationTwoRfStationsWithWake()
        self._test_matching(sim)

    def _test_matching(self, sim):
        simulation = sim.simulation
        beam = sim.beam1
        ts = (
            np.linspace(
                0,
                simulation.magnetic_cycle.get_t_rev_init(
                    simulation.ring.circumference,
                    turn_i_init=0,
                    t_init=0,
                    particle_type=beam.particle_type,
                ),
            )
            / 36540
        )
        sim.simulation.plot_potential_well_empiric(
            ts=ts,
            particle_type=beam.particle_type,
        )
        plt.show()
        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=SemiEmpiricMatcher(
                t_lim=(ts.min(), ts.max()),
                h_max=50,
                n_macroparticles=1e6,
                internal_grid_shape=(2048 - 1, 2048 - 1),
                density_modifier=5,
            ),
        )
        sim.beam1.plot_hist2d()

        def custom_action(
            simulation: Simulation, beam: Beam
        ):  # pragma: no cover
            plt.figure("bunch live")
            plt.clf()
            plt.subplot(2, 1, 1)
            beam.plot_hist2d()
            plt.subplot(2, 1, 2)
            beam.plot_hist(axis=0)
            plt.draw()
            plt.pause(0.1)

        sim.simulation.run_simulation(
            beams=(sim.beam1,), callback=custom_action, n_turns=None
        )


class TestCallables:
    def test_get_hamiltonian_semi_analytic(self):
        get_hamiltonian_semi_analytic()  # TODO


if __name__ == "__main__":
    unittest.main()
