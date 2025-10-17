import unittest
import numpy as np

from blond.handle_results.observables_as_elements import BeamLoggerElement
from blond._core.simulation.simulation import Simulation
from blond._core.beam.beams import Beam
from blond._core.beam.particle_types import proton


n_particles = 10
beam = Beam(intensity=1e11, particle_type=proton)  # or appropriate params
dt = np.linspace(-1e-9, 1e-9, n_particles)
dE = np.linspace(-1e6, 1e6, n_particles)
beam.setup_beam(dt=dt, dE=dE)


class DummySimulation(Simulation):
    pass


class TestBeamLoggerElement(unittest.TestCase):
    def setUp(self):
        self.beam = beam
        self.n_turns = 3
        self.logger = BeamLoggerElement(beam=self.beam, n_turns=self.n_turns)

    def test_track_logs_data_correctly(self):
        for turn in range(self.n_turns):
            # Modify beam data slightly each turn
            self.beam._dE = np.full(self.beam.common_array_size, turn)
            self.beam._dt = np.full(self.beam.common_array_size, turn + 0.5)
            self.logger.track(self.beam)

            self.assertEqual(self.logger._de_log.shape, (self.n_turns, self.beam.common_array_size))
            self.assertEqual(self.logger._dt_log.shape, (self.n_turns, self.beam.common_array_size))

        # Check that data was logged correctly
        for turn in range(self.n_turns):
            np.testing.assert_array_equal(self.logger._de_log[turn], np.full(self.beam.common_array_size, turn))
            np.testing.assert_array_equal(self.logger._dt_log[turn], np.full(self.beam.common_array_size, turn + 0.5))

        self.assertEqual(self.logger._active_index, self.n_turns)

    def test_track_stops_after_n_turns(self):
        for _ in range(self.n_turns):
            self.logger.track(self.beam)

        # Save old logs for comparison
        old_de_log = self.logger._de_log.copy()
        old_dt_log = self.logger._dt_log.copy()
        old_active_index = self.logger._active_index


        self.logger.track(self.beam)

        # Logs and index unchanged
        np.testing.assert_array_equal(self.logger._de_log, old_de_log)
        np.testing.assert_array_equal(self.logger._dt_log, old_dt_log)
        self.assertEqual(self.logger._active_index, old_active_index)

    def test_get_logged_data_returns_correct_dict(self):
        self.logger.track(self.beam)
        data = self.logger.get_logged_data()

        self.assertIn("de", data)
        self.assertIn("dt", data)
        self.assertIsInstance(data["de"], np.ndarray)
        self.assertIsInstance(data["dt"], np.ndarray)

    def test_get_turn_data_returns_correct_arrays(self):
        self.logger.track(self.beam)
        turn_data = self.logger.get_turn_data(0)

        np.testing.assert_array_equal(turn_data["de"], self.logger._de_log[0])
        np.testing.assert_array_equal(turn_data["dt"], self.logger._dt_log[0])


if __name__ == "__main__":
    unittest.main()

