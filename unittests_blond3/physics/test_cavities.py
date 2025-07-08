import unittest
from unittest.mock import Mock

from blond3 import WakeField, Simulation
from blond3.physics.cavities import (
    CavityBaseClass,
    MultiHarmonicCavity,
    SingleHarmonicCavity,
)
from blond3.physics.feedbacks.base import LocalFeedback


class TestCavityBaseClass(unittest.TestCase):
    def setUp(self):
        self.cavity_base_class = CavityBaseClass(
            n_rf=10,
            section_index=0,
            local_wakefield=Mock(WakeField),
            cavity_feedback=Mock(LocalFeedback),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_n_rf(self):
        self.assertEqual(10, self.cavity_base_class.n_rf)

    def test_on_init_simulation(self):
        simulation = Mock(Simulation)
        simulation.turn_i = Mock()
        self.cavity_base_class.on_init_simulation(
            simulation=simulation,
        )

    def test_on_run_simulation(self):
        simulation = Mock(Simulation)
        self.cavity_base_class.on_run_simulation(
            simulation=simulation,
            n_turns=10,
            turn_i_init=1,
        )



    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.cavity_base_class.track(beam=None)


class TestMultiHarmonicCavity(unittest.TestCase):
    def setUp(self):
        self.multi_harmonic_cavity = MultiHarmonicCavity(
            n_harmonics=15,
            section_index=0,
            local_wakefield=Mock(WakeField),
            cavity_feedback=Mock(LocalFeedback),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.multi_harmonic_cavity.track(beam=None)


class TestSingleHarmonicCavity(unittest.TestCase):
    def setUp(self):
        self.single_harmonic_cavity = SingleHarmonicCavity(
            section_index=0,
            local_wakefield=Mock(WakeField),
            cavity_feedback=Mock(LocalFeedback),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.single_harmonic_cavity.track(beam=None)


if __name__ == "__main__":
    unittest.main()
