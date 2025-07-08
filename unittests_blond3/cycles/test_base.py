import unittest

from blond3 import Simulation
from blond3.cycles.base import ProgrammedCycle


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


if __name__ == "__main__":
    unittest.main()
