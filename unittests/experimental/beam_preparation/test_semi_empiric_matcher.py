import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import Beam, Simulation, backend
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
    get_hamiltonian_semi_analytic,
)


class TestSemiEmpiricMatcher(unittest.TestCase):
    def test_roughly_correct_no_intensity(self):
        # check if the mean and the 10% and 90% percentiles are correct
        from blond.testing.simulation import SimulationTwoRfStations

        # pinned values
        expected_dt = {
            10: 1.0517069437554483e-09,
            50: 1.247751235666783e-09,
            90: 1.4438786832826622e-09,
        }
        expected_dE = {
            10: -131473272.0,
            50: -44513.6328125,
            90: 131371752.0,
        }
        sim = SimulationTwoRfStations()
        self._test_matching(sim)

        DEV_PLOT = False
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                plt.clf()
                beam.plot_hist2d()
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,), callback=my_callback
            )
        for percentile in (10, 50, 90):
            percentile_dt = float(np.percentile(sim.beam1._dt, percentile))
            percentile_dE = float(np.percentile(sim.beam1._dE, percentile))
            # print(percentile, ":", percentile_dt,",")
            # print(percentile, ":", percentile_dE,",")
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-5 if backend.float == np.float32 else 1e-12,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-5 if backend.float == np.float32 else 1e-12,
            )

    def test_roughly_correct_intensity(self):
        from blond.testing.simulation import SimulationTwoRfStationsWithWake

        sim = SimulationTwoRfStationsWithWake()
        self._test_matching(sim)
        raise Exception()  # FIXME

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

        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(ts.min(), ts.max()),
                hamilton_max=50,
                n_macroparticles=1e6,
                internal_grid_shape=(512 - 1, 512 - 1),
                density_modifier=5,
                increment_intensity_effects_until_iteration_i=40,
                maxiter_intensity_effects=50,
                animate=False,
            ),
        )


class TestCallables:
    def test_get_hamiltonian_semi_analytic(self):
        # Define simple test inputs
        ts = np.linspace(0, 1, 100)  # Time or spatial grid
        eta = 0.1  # Some parameter (perhaps perturbation strength or scaling factor)
        shape = (100, 100)  # Shape of Hamiltonian matrix

        # Define a simple potential well — e.g., harmonic potential
        def gen_potential_well(x):
            return 0.5 * x**2

        # Use the same ts as x-values
        potential_values = gen_potential_well(ts)

        # Reference energy for testing (hypothetical or computed elsewhere)
        reference_total_energy = 1.0  # Placeholder

        # Call the function
        deltaE_grid, time_grid, hamilton_2D = get_hamiltonian_semi_analytic(
            ts=ts,
            potential_well=potential_values,
            reference_total_energy=reference_total_energy,
            eta=eta,
            shape=shape,
        )

        hamilton_2D_expected = (
            0.5 * eta / reference_total_energy * deltaE_grid.T * deltaE_grid.T
            + potential_values
        ).T  # [eV]
        DEV_PLOT = False
        if DEV_PLOT:
            plt.figure()
            plt.imshow(hamilton_2D_expected)
            plt.title("hamilton_2D_expected")
            plt.figure()
            plt.imshow(hamilton_2D)
            plt.title("hamilton_2D")
            plt.show()
        np.testing.assert_allclose(
            hamilton_2D_expected,
            hamilton_2D,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
