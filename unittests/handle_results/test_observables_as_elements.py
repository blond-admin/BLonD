import unittest
from copy import deepcopy
from unittest.mock import Mock

import numpy as np

from blond import (
    Beam,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
)
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import ProbeBeam
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.beam_physics_relevant_elements import (
    BeamPhysicsRelevantElements,
)
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
    BunchObservationMetaParams,
    InducedVoltageObservationCR,
)

simulation = Mock(Simulation)
simulation.ring.n_rf_stations = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.section_i = DynamicParameter(None)
simulation.section_i.current_group = 0
simulation.turn_i = DynamicParameter(None)
simulation.turn_i.value = 0

beam = Mock(BeamBaseClass)
beam.reference = Mock(ReferenceCoordinates)
beam.common_array_size = 4
beam.reference.time = 0.8
beam.reference.total_energy = 11.0
beam.read_partial_dE.return_value = np.arange(4, dtype=float)
beam.read_partial_dt.return_value = np.arange(4, dtype=float) + 0.1
beam.read_partial_flags.return_value = np.ones(4, dtype=int)


class TestBeamObservationInRingElement(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = BeamObservationInRingElement(
            each_turn_i=1,
            section_index=0,
            n_turns=3,
            folder=callers_relative_path("results/", stacklevel=1),
            name="test_obs",
        )
        self.observation.common_filepath = "test"
        self.observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=3,
        )

    def test_recorders_are_initialized(self):
        """Ensure recorders exist and are DenseArrayRecorder instances."""
        for rec_name in [
            "_dEs",
            "_dts",
            "_flags",
            "_reference_time",
            "_reference_total_energy",
        ]:
            self.assertTrue(hasattr(self.observation, rec_name))
            rec = getattr(self.observation, rec_name)
            self.assertEqual(rec._memory.shape[0], 3 // 1 + 2)

    def test_track_and_retrieve_data(self):
        """Ensure that calling track() stores data and public properties return it."""
        for _ in range(3):
            self.observation.track(beam)

        np.testing.assert_array_equal(
            self.observation.dEs,
            np.tile(beam.read_partial_dE.return_value, (3, 1)),
            err_msg="ΔE values not recorded correctly",
        )

        np.testing.assert_array_equal(
            self.observation.dts,
            np.tile(beam.read_partial_dt.return_value, (3, 1)),
            err_msg="Δt values not recorded correctly",
        )

        np.testing.assert_array_equal(
            self.observation.flags,
            np.tile(beam.read_partial_flags.return_value, (3, 1)),
            err_msg="Flags not recorded correctly",
        )

        np.testing.assert_array_equal(
            self.observation.reference_time,
            np.full(3, beam.reference.time),
            err_msg="Reference time not recorded correctly",
        )

        np.testing.assert_array_equal(
            self.observation.reference_total_energy,
            np.full(3, beam.reference.total_energy),
            err_msg="Reference total energy not recorded correctly",
        )

    def test_ignores_probe_beam(self):
        observation = BunchObservationMetaParams(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        observation.common_filepath = "test"

        simulation.ring = Mock(Ring)
        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.elements = [
            observation,
        ]

        observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=3,
        )

        probe_beam = Mock(spec=ProbeBeam)
        probe_beam.reference = Mock(ReferenceCoordinates)
        probe_beam.common_array_size = 4
        probe_beam.reference.time = 0.8
        probe_beam.reference.total_energy = 11.0
        probe_beam.read_partial_dE.return_value = np.arange(4, dtype=float)
        probe_beam.read_partial_dt.return_value = (
            np.arange(4, dtype=float) + 0.1
        )
        probe_beam.read_partial_flags.return_value = np.ones(4, dtype=int)

        for _ in range(3):
            observation.track(probe_beam)

        assert len(observation.sigma_dt) == 0
        assert len(observation.sigma_dE) == 0
        assert len(observation.mean_dt) == 0
        assert len(observation.mean_dE) == 0
        assert len(observation.rms_emittance) == 0


class TestInducedVoltage(unittest.TestCase):
    def setUp(self) -> None:
        sim = Mock(Simulation)
        sim.turn_i = 0
        shc = Mock(SingleHarmonicRFStation)
        shc._local_wakefield = Mock(WakeField)
        shc._local_wakefield._profile = Mock(StaticProfile)
        shc._local_wakefield._profile.hist_x = np.array([0, 1])
        shc._local_wakefield.induced_voltage = np.zeros(5)
        shc.name = "mock"
        shc._turn_i = Mock(DynamicParameter)
        shc._turn_i.value = 0
        beam = Mock(Beam)
        beam.is_counter_rotating = False

        obs = InducedVoltageObservationCR(rf_station=shc, each_turn_i=1)

        with self.assertWarnsRegex(
            Warning, "no induced voltage calculated yet "
        ):
            obs._track(beam)
        with self.assertRaisesRegex(
            AttributeError,
            "'NoneType' object has no attribute 'get_valid_entries'",
        ):
            _ = obs.induced_voltage

    def test___init__(self):
        pass


if __name__ == "__main__":
    unittest.main()
