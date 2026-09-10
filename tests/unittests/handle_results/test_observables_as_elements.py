import unittest
from unittest.mock import Mock

import numpy as np

from blond import (
    Resonators,
    Ring,
    Simulation,
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
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)

simulation = Mock(Simulation)
simulation.ring = Mock(Ring)
simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
simulation.ring.n_rf_stations = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.turn_counter = DynamicParameter(None)
simulation.turn_counter.value = 0

beam = Mock(BeamBaseClass)
beam.reference = Mock(ReferenceCoordinates)
beam.common_array_size = 4
beam.reference.time = 0.8
beam.reference.total_energy = 11.0
beam.read_partial_dE.return_value = np.arange(4, dtype=float)
beam.read_partial_dt.return_value = np.arange(4, dtype=float) + 0.1
beam.read_partial_flags.return_value = np.ones(4, dtype=int)
beam._is_counter_rotating = True


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

        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.get_elements.return_value = [self.observation]

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
            self.assertEqual(rec._memory.shape[0], 5)

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
        observation = BeamObservationInRingElement(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        observation.common_filepath = "test"

        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.elements = [observation]
        simulation.ring.elements.get_elements.return_value = [observation]

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

        self.assertEqual(len(observation._dEs.get_valid_entries()), 0)
        self.assertEqual(len(observation._dts.get_valid_entries()), 0)
        self.assertEqual(
            len(observation._reference_time.get_valid_entries()), 0
        )
        self.assertEqual(
            len(observation._reference_total_energy.get_valid_entries()), 0
        )
        self.assertEqual(len(observation._flags.get_valid_entries()), 0)


class TestBunchObservationMetaParams(unittest.TestCase):
    def test_ignores_probe_beam(self):
        observation = BunchObservationMetaParams(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        observation.common_filepath = "test"

        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.elements = [observation]

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

        self.assertEqual(len(observation.sigma_dt), 0)
        self.assertEqual(len(observation.sigma_dE), 0)
        self.assertEqual(len(observation.mean_dt), 0)
        self.assertEqual(len(observation.mean_dE), 0)
        self.assertEqual(len(observation.rms_emittance), 0)


class TestBunchObservationMetaParamsPlacement(unittest.TestCase):
    """Turn, energy and intensity records plus the placement bookkeeping."""

    def _observation(self, **kwargs) -> BunchObservationMetaParams:
        observation = BunchObservationMetaParams(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            **kwargs,
        )
        observation.common_filepath = "test"
        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.elements = [observation]
        observation.on_run_simulation(
            simulation=simulation, beam=beam, n_turns=3
        )
        return observation

    def _real_beam(self):
        real = Mock(BeamBaseClass)
        real.reference = Mock(ReferenceCoordinates)
        real.reference.time = 0.0
        real.reference.total_energy = 63.0e9
        real.intensity = 2.7e12
        real._dt = np.array([0.0, 1.0, 2.0, 3.0]) * 1e-12
        real._dE = np.array([-1.0, 0.0, 1.0, 2.0]) * 1e6
        real.rms_emittance = 1.0e-6
        return real

    def test_records_turn_energy_and_intensity(self):
        observation = self._observation(turn_fraction=0.25)
        real = self._real_beam()
        for turn in range(3):
            simulation.turn_counter.value = turn
            real.intensity = 2.7e12 * (1.0 - 0.1 * turn)
            observation.track(real)
        simulation.turn_counter.value = 0
        np.testing.assert_array_equal(observation.turns, [0.25, 1.25, 2.25])
        np.testing.assert_array_equal(observation.total_energy, [63.0e9] * 3)
        np.testing.assert_allclose(
            observation.intensity, 2.7e12 * np.array([1.0, 0.9, 0.8])
        )
        self.assertEqual(len(observation.sigma_dt), 3)

    def test_beam_filter_selects_one_beam(self):
        wanted = self._real_beam()
        other = self._real_beam()
        observation = self._observation(beam=wanted)
        observation.track(other)
        observation.track(wanted)
        observation.track(other)
        self.assertEqual(len(observation.turns), 1)
        self.assertEqual(len(observation.intensity), 1)

    def test_placement_bookkeeping(self):
        observation = BunchObservationMetaParams(
            each_turn_i=1,
            folder="",
            section_index=3,
            name="stats_s3",
            label="co_rotating",
            turn_fraction=0.5,
        )
        self.assertEqual(observation.section_index, 3)
        self.assertEqual(observation.name, "stats_s3")
        self.assertEqual(observation.label, "co_rotating")
        self.assertEqual(observation.turn_fraction, 0.5)
        # The label defaults to the name.
        unlabeled = BunchObservationMetaParams(
            each_turn_i=1, folder="", name="just_a_name"
        )
        self.assertEqual(unlabeled.label, "just_a_name")


class TestInducedVoltageObservationCR(unittest.TestCase):
    def test_no_induced_voltage(self):
        wakefield = WakeField(
            solver=SingleTurnResonatorConvolutionSolver(),
            sources=(
                Resonators(
                    center_frequencies=1, shunt_impedances=1, quality_factors=1
                ),
            ),
        )
        wakefield._profile = Mock(StaticProfile)
        wakefield._profile.hist_x = np.arange(3)

        observation = InducedVoltageObservationCR(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            wake_field=wakefield,
        )
        observation.common_filepath = "test"

        simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
        simulation.ring.elements.elements = [observation]

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
        probe_beam._is_counter_rotating = True

        for _ in range(3):
            observation.track(probe_beam)

        # no observation due to attribute error
        self.assertEqual(
            len(observation._induced_voltage.get_valid_entries()), 0
        )
        self.assertEqual(
            len(observation._beam_reference_time.get_valid_entries()), 0
        )
        self.assertEqual(len(observation._beam_profile.get_valid_entries()), 0)


if __name__ == "__main__":
    unittest.main()
