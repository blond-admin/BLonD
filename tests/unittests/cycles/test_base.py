import unittest

from blond import Simulation
from blond.cycles.base import ProgrammedCycle
from blond.testing.backend_testing import BLonDTestCase


class ProgrammedCycleHelper(ProgrammedCycle):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
    ) -> None:
        pass


class TestProgrammedCycle(BLonDTestCase):
    def setUp(self):
        self.programmed_cycle = ProgrammedCycleHelper()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp


if __name__ == "__main__":
    unittest.main()
