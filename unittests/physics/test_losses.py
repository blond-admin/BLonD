import unittest

from blond import Simulation
from blond._core.beam.base import BeamBaseClass
from blond.physics.losses import LossesBaseClass


class LossesBaseClassHelper(LossesBaseClass):
    def track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        pass


class TestLossesBaseClass(unittest.TestCase):
    def test_init(self):
        LossesBaseClassHelper()
