import unittest
from sys import flags

import numpy as np

from blond import Beam, Simulation, uranium_29
from blond._core.beam.base import BeamBaseClass, BeamFlags
from blond.physics.losses import LossesBaseClass


class LossesBaseClassHelper(LossesBaseClass):
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

    def test_track(self):
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        flags = np.ones(10)
        flags[:5] = BeamFlags.LOST.value
        beam.setup_beam(dt=np.arange(10), dE=np.ones(10), flags=flags)
        LossesBaseClassHelper().track(beam=beam)
        self.assertEqual(beam.common_array_size, 5)
        np.testing.assert_almost_equal(
            np.sort(beam.read_partial_dt()),
            np.sort(np.arange(10)[5:]),
        )
