import unittest
from unittest.mock import Mock, create_autospec

import numpy as np

from blond3 import (
    Simulation,
    Ring,
    Beam,
    SingleHarmonicCavity,
    RfStationParams,
    DriftSimple,
    proton,
)
from blond3.cycles.energy_cycle import EnergyCyclePerTurn
from blond3.handle_results.observables import Observables, BunchObservation


class TestSimulation(unittest.TestCase):
    def setUp(self):
        ring = Ring(circumference=26658.883,
        )

        cavity1 = SingleHarmonicCavity(
        )
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        N_TURNS = int(1e3)
        energy_cycle = EnergyCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS)
        )

        drift1 = DriftSimple(
            share_of_circumference=1.0,
        )
        drift1.transition_gamma = 55.759505

        beam1 = Beam(n_particles=1e9, particle_type=proton)
        beam1.setup_beam(dt=np.linspace(1, 10, 10), dE=np.linspace(11, 20, 10))
        self.simulation = Simulation.from_locals(locals())


    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__exec_on_init_simulation(self):
        self.simulation._exec_on_init_simulation()

    def test__exec_on_run_simulation(self):
        self.simulation._exec_on_run_simulation(n_turns=10, turn_i_init=1)

    @unittest.skip
    def test__run_simulation_counterrotating_beam(self):
        # TODO: implement test for `_run_simulation_counterrotating_beam`
        self.simulation._run_simulation_counterrotating_beam(
            n_turns=None, turn_i_init=None, observe=None, show_progressbar=None
        )

    def test__run_simulation_single_beam(self):
        observe = Mock(spec=Observables)

        def my_callback(simulation: Simulation) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=True)
        self.simulation._run_simulation_single_beam(
            n_turns=10,
            turn_i_init=0,
            observe=(observe,),
            show_progressbar=True,
            callback=mock_func,
        )
        observe.update.assert_called()
        mock_func.assert_called()

    def test_beams(self):
        self.assertNotEqual(None, self.simulation.beams)

    def test_energy_cycle(self):
        self.assertNotEqual(None, self.simulation.energy_cycle)

    @unittest.skip("Testet in setUp")
    def test_from_locals(self):
        #self.simulation.from_locals(locals=None)
        pass

    @unittest.skip
    def test_get_hash(self):
        # TODO: implement test for `get_hash`
        self.simulation.get_hash()

    @unittest.skip
    def test_get_legacy_map(self):
        # TODO: implement test for `get_legacy_map`
        self.simulation.get_legacy_map()

    @unittest.skip
    def test_get_potential_well(self):
        # TODO: implement test for `get_potential_well`
        self.simulation.get_potential_well()

    @unittest.skip
    def test_get_potential_well_analytic(self):
        # TODO: implement test for `get_potential_well_analytic`
        self.simulation.get_potential_well_analytic()

    @unittest.skip
    def test_get_potential_well_empiric(self):
        # TODO: implement test for `get_potential_well_empiric`
        self.simulation.get_potential_well_empiric()

    @unittest.skip
    def test_get_separatrix(self):
        # TODO: implement test for `get_separatrix`
        self.simulation.get_separatrix()

    def test_invalidate_cache(self):
        self.simulation.invalidate_cache()

    @unittest.skip
    def test_load_results(self):
        # TODO: implement test for `load_results`
        self.simulation.load_results(
            n_turns=None, turn_i_init=None, observe=None, callback=None
        )

    def test_on_init_simulation(self):
        self.simulation.on_init_simulation(simulation=self.simulation)

    @unittest.skip
    def test_on_prepare_beam(self):
        # TODO: implement test for `on_prepare_beam`
        self.simulation.on_prepare_beam(preparation_routine=None, turn_i=None)

    def test_on_run_simulation(self):
        self.simulation.on_run_simulation(
            simulation=self.simulation, n_turns=10, turn_i_init=0
        )

    def test_print_one_turn_execution_order(self):
        self.simulation.print_one_turn_execution_order()

    def test_profiling(self):
        self.simulation.profiling(
            turn_i_init=0, profile_start_turn_i=10, profile_n_turns=20
        )

    def test_ring(self):
        self.assertIsInstance(self.simulation.ring, Ring)

    def test_run_simulation(self):
        observe = BunchObservation(each_turn_i=10)

        def my_callback(simulation: Simulation) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=False)

        self.simulation.run_simulation(
            n_turns=10,
            turn_i_init=0,
            observe=(observe,),
            show_progressbar=True,
            callback=mock_func,
        )
        mock_func.assert_called()


if __name__ == "__main__":
    unittest.main()
