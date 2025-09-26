import unittest

import matplotlib.pyplot as plt

from blond import Beam, DriftSimple, Simulation
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
    get_hamiltonian_semi_analytic,
)


class TestSemiEmpiricMatcher(unittest.TestCase):
    def test_something(self):
        from blond.testing.simulation import SimulationTwoRfStations

        sim = SimulationTwoRfStations()

        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=SemiEmpiricMatcher(
                n_macroparticles=1e6,
            ),
        )

        sim.beam1.plot_hist2d()

        def custom_action(
            simulation: Simulation, beam: Beam
        ):  # pragma: no cover
            plt.figure("bunch live")
            plt.clf()
            beam.plot_hist2d()
            plt.draw()
            plt.pause(0.1)

        sim.simulation.run_simulation(
            beams=(sim.beam1,), callback=custom_action
        )
        plt.show()


if __name__ == "__main__":
    unittest.main()
