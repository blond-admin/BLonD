import unittest
from unittest.mock import Mock

import numpy as np

from blond import Simulation
from blond._core.base import DynamicParameter
from blond._core.beam.base import BeamBaseClass
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)

simulation = Mock(Simulation)
simulation.ring.n_cavities = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.section_i = DynamicParameter(None)
simulation.section_i.current_group = 0
simulation.turn_i = DynamicParameter(None)
simulation.turn_i.value = 0

beam = Mock(BeamBaseClass)
beam.common_array_size = 4
beam.reference_time = 0.8
beam.reference_total_energy = 11.0
beam.read_partial_dE.return_value = np.arange(4, dtype=float)
beam.read_partial_dt.return_value = np.arange(4, dtype=float) + 0.1
beam.read_partial_flags.return_value = np.ones(4, dtype=int)


class TestBeamObservationInRingElement(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = BeamObservationInRingElement(
            each_turn_i=1,
            section_index=0,
            n_turns=5,
            folder=callers_relative_path("results/", stacklevel=1),
            name="test_obs",
        )

    def test___init__(self) -> None:
        obs = BeamObservationInRingElement(
            each_turn_i=2,
            n_turns=10,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        self.assertEqual(obs.each_turn_i, 2)
        self.assertEqual(obs.n_turns, 10)

    def test_on_run_simulation(self) -> None:
        """Ensure that recorders are created correctly."""
        self.observation.common_name = "test"
        self.observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=10,
            turn_i_init=0,
        )

        for recorder_name in [
            "_dEs",
            "_dts",
            "_flags",
            "_reference_time",
            "_reference_total_energy",
        ]:
            recorder = getattr(self.observation, recorder_name)
            self.assertIsInstance(recorder, DenseArrayRecorder)

    def test_track_records_beam_data(self) -> None:
        """Ensure that track() writes correct data to recorders."""
        self.observation._dEs = Mock(DenseArrayRecorder)
        self.observation._dts = Mock(DenseArrayRecorder)
        self.observation._flags = Mock(DenseArrayRecorder)
        self.observation._reference_time = Mock(DenseArrayRecorder)
        self.observation._reference_total_energy = Mock(DenseArrayRecorder)

        self.observation.track(beam)

        self.observation._dEs.write.assert_called_once_with(
            beam.read_partial_dE.return_value
        )
        self.observation._dts.write.assert_called_once_with(
            beam.read_partial_dt.return_value
        )
        self.observation._flags.write.assert_called_once_with(
            beam.read_partial_flags.return_value
        )
        self.observation._reference_time.write.assert_called_once_with(
            beam.reference_time
        )
        self.observation._reference_total_energy.write.assert_called_once_with(
            beam.reference_total_energy
        )
if __name__ == "__main__":
    unittest.main()
