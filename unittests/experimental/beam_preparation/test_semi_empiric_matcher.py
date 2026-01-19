import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    Simulation,
    backend,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
    get_hamilton_semi_analytic,
)


class TestSemiEmpiricMatcher(unittest.TestCase):
    def test_roughly_correct_no_intensity_above_transition(self):
        # check if the mean and the 10% and 90% percentiles are correct
        from blond.testing.simulation import SimulationTwoRfStations

        # pinned values
        expected_dt = {
            10: 9.37316899945945e-10,
            50: 1.2509709354375757e-09,
            90: 1.5621352417176764e-09,
        }
        expected_dE = {
            10: -203033118.014676,
            50: -497190.7807013786,
            90: 200759056.35187533,
        }
        sim = SimulationTwoRfStations()
        self._test_matching(sim)

        DEV_PLOT = False
        if DEV_PLOT:
            idx = np.argmax(sim.beam1.read_partial_dt())
            data = np.ones((1000, 2))
            data[:, :] = np.nan

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 1 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                data[simulation.turn_i.value % data.shape[0], 0] = (
                    sim.beam1.read_partial_dt()[idx]
                )
                data[simulation.turn_i.value % data.shape[0], 1] = (
                    sim.beam1.read_partial_dE()[idx]
                )
                plt.plot(data[:, 0], data[:, 1], ".")
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                if simulation.turn_i.value == 0:
                    plt.cla()
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,),
                callbacks=my_callback,
                n_turns=1e6,
            )
        for percentile in (10, 50, 90):
            percentile_dt = float(
                np.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                np.percentile(sim.beam1.read_partial_dE(), percentile)
            )
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def test_roughly_correct_intensity_above_transition(self):
        from blond.testing.simulation import SimulationTwoRfStationsWithWake

        sim = SimulationTwoRfStationsWithWake()
        self._test_matching(sim)
        DEV_PLOT = False
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 10 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d(range=((0.7e-9, 1.8e-9), (-3.5e8, 3.5e8)))
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.draw()
                plt.pause(0.1)

            sim.simulation.turn_i.value = 0
            my_callback(simulation=sim.simulation, beam=sim.beam1)
            sim.simulation.run_simulation(
                beams=(sim.beam1,), callbacks=my_callback
            )
        # pinned values
        expected_dt = {
            10: 8.945807526908758e-10,
            50: 1.1946294403398606e-09,
            90: 1.4906477092137164e-09,
        }
        expected_dE = {
            10: -202934616.99385634,
            50: -491867.9813599617,
            90: 200521467.98119268,
        }
        for percentile in (10, 50, 90):
            percentile_dt = float(
                np.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                np.percentile(sim.beam1.read_partial_dE(), percentile)
            )

            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def test_roughly_correct_no_intensity_below_transition(self):
        # check if the mean and the 10% and 90% percentiles are correct
        from blond.testing.simulation import SimulationTwoRfStations

        # pinned values
        expected_dt = {
            10: 2.184644423501519e-09,
            50: 2.498358817146089e-09,
            90: 2.809566454479882e-09,
        }
        expected_dE = {
            10: -2094325465.9539328,
            50: -5128618.048338078,
            90: 2070868084.7240956,
        }
        sim = SimulationTwoRfStations(below_transition_crossing=True)
        self._test_matching(sim, below_transition_crossing=True)

        DEV_PLOT = False
        if DEV_PLOT:
            idx = np.argmax(sim.beam1.read_partial_dt())
            data = np.ones((1000, 2))
            data[:, :] = np.nan

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 1 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d()
                data[simulation.turn_i.value % data.shape[0], 0] = (
                    sim.beam1.read_partial_dt()[idx]
                )
                data[simulation.turn_i.value % data.shape[0], 1] = (
                    sim.beam1.read_partial_dE()[idx]
                )
                plt.plot(data[:, 0], data[:, 1], ".")
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                if simulation.turn_i.value == 0:
                    plt.cla()
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.pause(0.1)

            sim.simulation.run_simulation(
                beams=(sim.beam1,),
                callbacks=my_callback,
                n_turns=1e6,
            )
        for percentile in (10, 50, 90):
            percentile_dt = float(
                np.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                np.percentile(sim.beam1.read_partial_dE(), percentile)
            )

            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )

            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def test_roughly_correct_intensity_below_transition(self):
        from blond.testing.simulation import SimulationTwoRfStationsWithWake

        sim = SimulationTwoRfStationsWithWake(below_transition_crossing=True)
        self._test_matching(sim, below_transition_crossing=True)
        DEV_PLOT = False
        if DEV_PLOT:

            def my_callback(simulation: Simulation, beam: Beam):
                if simulation.turn_i.value % 10 != 0:
                    return
                plt.subplot(2, 1, 1)
                plt.cla()
                beam.plot_hist2d()
                plt.axhline(beam.read_partial_dE().mean())
                plt.axvline(beam.read_partial_dt().mean())
                plt.subplot(2, 1, 2)
                plt.hist(
                    beam.read_partial_dt(),
                    bins=256,
                    histtype="step",
                    density=True,
                )
                plt.draw()
                plt.draw()
                plt.pause(0.1)

            sim.simulation.turn_i.value = 0
            my_callback(simulation=sim.simulation, beam=sim.beam1)
            sim.simulation.run_simulation(
                beams=(sim.beam1,), callbacks=my_callback
            )
        # pinned values
        expected_dt = {
            10: 2.198671357631324e-09,
            50: 2.5062625899860837e-09,
            90: 2.8134437780368726e-09,
        }
        expected_dE = {
            10: -2092017166.3373477,
            50: -5219386.432623548,
            90: 2067526837.520195,
        }
        for percentile in (10, 50, 90):
            percentile_dt = float(
                np.percentile(sim.beam1.read_partial_dt(), percentile)
            )
            percentile_dE = float(
                np.percentile(sim.beam1.read_partial_dE(), percentile)
            )
            np.testing.assert_allclose(
                expected_dt[percentile],
                percentile_dt,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                expected_dE[percentile],
                percentile_dE,
                rtol=1e-4,
            )

    def _test_matching(self, sim, below_transition_crossing=False):
        simulation = sim.simulation
        beam = sim.beam1
        t_rev = simulation.magnetic_cycle.get_t_rev_init(
            simulation.ring.circumference,
            particle_type=beam.particle_type,
        )
        ts = (
            np.linspace(
                0 + (t_rev / 2 if below_transition_crossing else 0),
                t_rev + (t_rev / 2 if below_transition_crossing else 0),
            )
            / 36540
        )
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
                hamilton_to_density_kwargs=dict(
                    hamilton_max=100,
                    density_modifier=4,
                ),
                n_macroparticles=1e5,
                internal_grid_shape=(512 - 1, 512 - 1),
                increment_intensity_effects_until_iteration_i=10,
                maxiter_intensity_effects=1000,
                tolerance=0.000001,
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
        deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=potential_values,
            reference_total_energy=reference_total_energy,
            eta=eta,
            beta=1,
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
