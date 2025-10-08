import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    MultiHarmonicCavity,
    Simulation,
    SingleHarmonicCavity,
    WakeField,
    backend,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
    get_hamiltonian_semi_analytic,
)
from blond.physics.profiles import ProfileBaseClass


class TestSemiEmpiricMatcher(unittest.TestCase):
    def test_roughly_correct_no_intensity(self):
        # check if the mean and the 10% and 90% percentiles are correct
        from blond.testing.simulation import SimulationTwoRfStations

        backend.set_specials("fortran")

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

        DEV_PLOT = True
        if DEV_PLOT:
            idx = np.argmax(sim.beam1._dt)
            data = np.ones((1000, 2))
            data[:, :] = np.nan

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 1 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                data[simulation.turn_i.value % data.shape[0], 0] = (
                    sim.beam1._dt[idx]
                )
                data[simulation.turn_i.value % data.shape[0], 1] = (
                    sim.beam1._dE[idx]
                )
                plt.plot(data[:, 0], data[:, 1], ".")
                plt.axhline(beam._dE.mean())
                plt.axvline(beam._dt.mean())
                plt.subplot(2, 1, 2)
                if simulation.turn_i.value == 0:
                    plt.cla()
                plt.hist(beam._dt, bins=256, histtype="step", density=True)
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,),
                callback=my_callback,
                n_turns=1e6,
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
        DEV_PLOT = True
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 10 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                plt.axhline(beam._dE.mean())
                plt.axvline(beam._dt.mean())
                plt.subplot(2, 1, 2)
                plt.hist(beam._dt, bins=256, histtype="step")
                plt.draw()
                plt.draw()
                plt.pause(0.1)

            sim.simulation.turn_i.value = 0
            my_callback(simulation=sim.simulation, beam=sim.beam1)
            sim.simulation.run_simulation(
                beams=(sim.beam1,), callback=my_callback
            )
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
        print(ts.min(), ts.max())
        # actively change the harmonic off the revolution time.
        # matching should still work
        # cav = sim.simulation.ring.elements.get_element(MultiHarmonicCavity)
        # cav.harmonic = 10*33000 * np.ones(len(cav.harmonic), backend.float)
        # cav = sim.simulation.ring.elements.get_element(SingleHarmonicCavity)
        # cav.harmonic = 10*33000

        sim.simulation.prepare_beam(
            beam=sim.beam1,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(ts.min(), ts.max()),
                hamilton_max=27e3 * 10,
                n_macroparticles=1e6,
                internal_grid_shape=(1024 - 1, 1024 - 1),
                density_modifier=4,
                increment_intensity_effects_until_iteration_i=10,
                maxiter_intensity_effects=1000,
                tolerance=0.000001,
                animate=True,
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
