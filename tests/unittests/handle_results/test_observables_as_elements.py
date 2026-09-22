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
from blond.testing.backend_testing import BLonDTestCase

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


def _simulation_with(elements: list) -> Mock:
    """Local simulation mock, so turn counter state does not leak."""
    local_simulation = Mock(Simulation)
    local_simulation.ring = Mock(Ring)
    local_simulation.ring.elements = Mock(BeamPhysicsRelevantElements)
    local_simulation.ring.elements.elements = elements
    local_simulation.turn_counter = DynamicParameter(0)
    return local_simulation


class TestBeamObservationInRingElement(BLonDTestCase):
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
            self.assertEqual(rec._memory.shape[0], 3)

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

    def test_each_turn_i_with_repeated_element(self):
        """The same element placed 5 times in the ring, observed on turns
        0 and 2 of 3 with ``each_turn_i=2``; all ten recordings must
        fit."""
        observation = BeamObservationInRingElement(
            each_turn_i=2,
            section_index=0,
            n_turns=3,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        local_simulation = _simulation_with([observation] * 5)
        local_simulation.ring.elements.get_elements.return_value = [
            observation
        ] * 5
        observation.on_run_simulation(
            simulation=local_simulation,
            beam=beam,
            n_turns=3,
        )

        for turn_i in range(3):
            if not observation.is_active_this_turn(turn_i=turn_i):
                continue
            for _ in range(5):
                observation.track(beam)

        self.assertEqual(len(observation.dEs), 10)


class TestBunchObservationMetaParams(BLonDTestCase):
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

    def test_each_turn_i_not_dividing_n_turns(self):
        """Turns 0, 2, 4 of 5 are observed with ``each_turn_i=2``; all
        three observations must fit into the recorders."""
        observation = BunchObservationMetaParams(
            each_turn_i=2,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        observation.on_run_simulation(
            simulation=_simulation_with([observation]),
            beam=beam,
            n_turns=5,
        )
        observed_beam = Mock(BeamBaseClass)
        observed_beam._dt = np.arange(4, dtype=float)
        observed_beam._dE = np.arange(4, dtype=float)
        observed_beam.rms_emittance = 1.0

        for turn_i in range(5):
            if observation.is_active_this_turn(turn_i=turn_i):
                observation.track(observed_beam)

        self.assertEqual(len(observation.mean_dt), 3)


class TestInducedVoltageObservationCR(BLonDTestCase):
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

    def test_each_turn_i_not_dividing_n_turns(self):
        """Turns 0, 2, 4 of 5 are observed with ``each_turn_i=2``, each
        recording both beams; all six observations must fit."""
        wake_field = Mock(WakeField)
        wake_field._profile = Mock(StaticProfile)
        wake_field._profile.hist_x = np.arange(3)
        wake_field.profile = wake_field._profile
        wake_field.profile.hist_y = np.ones(3)
        wake_field.induced_voltage = np.ones(3)

        observation = InducedVoltageObservationCR(
            each_turn_i=2,
            folder=callers_relative_path("results/", stacklevel=1),
            wake_field=wake_field,
        )
        local_simulation = _simulation_with([observation])
        observation.on_run_simulation(
            simulation=local_simulation,
            beam=beam,
            n_turns=5,
        )
        observed_beam = Mock(BeamBaseClass)
        observed_beam.reference = Mock(ReferenceCoordinates)
        observed_beam.reference.time = 0.8

        for turn_i in range(5):
            local_simulation.turn_counter.value = turn_i
            if not observation.is_active_this_turn(turn_i=turn_i):
                continue
            # OBS CAV OBS per beam: only the second passage records
            for is_counter_rotating in (False, True):
                observed_beam._is_counter_rotating = is_counter_rotating
                observation.track(observed_beam)
                observation.track(observed_beam)

        self.assertEqual(len(observation.beam_reference_time), 6)


if __name__ == "__main__":
    unittest.main()
