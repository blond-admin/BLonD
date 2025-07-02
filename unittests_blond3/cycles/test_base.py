import unittest
from unittest.mock import Mock

from blond3 import Simulation
from blond3.cycles.base import ProgrammedCycle, RfParameterCycle
from blond3.physics.cavities import CavityBaseClass


class ProgrammedCycleHelper(ProgrammedCycle):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class TestProgrammedCycle(unittest.TestCase):
    def setUp(self):
        self.programmed_cycle = ProgrammedCycleHelper()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class TestRfParameterCycle(unittest.TestCase):
    def setUp(self):
        self.rf_parameter_cycle = RfParameterCycle()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        simulation = Mock(spec=Simulation)
        cavity = Mock(spec=CavityBaseClass)
        self.rf_parameter_cycle.set_owner(cavity)
        self.rf_parameter_cycle.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(spec=Simulation)

        self.rf_parameter_cycle.on_run_simulation(
            simulation=simulation, n_turns=132, turn_i_init=3
        )

    def test_set_owner(self):
        cavity = Mock(spec=CavityBaseClass)

        self.rf_parameter_cycle.set_owner(cavity=cavity)


if __name__ == "__main__":
    unittest.main()
