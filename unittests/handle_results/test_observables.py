import unittest
from copy import deepcopy
from unittest.mock import Mock

import numpy as np

from blond import Simulation, SingleHarmonicCavity, StaticProfile, WakeField
from blond._core.base import DynamicParameter
from blond._core.beam.base import BeamBaseClass
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BeamObservationEndOfTurn,
    CavityPhaseObservation,
    DynamicProfileConstNBinsObservation,
    ObservablesEndOfTurnBase,
    StaticMultiProfileObservation,
    StaticProfileObservation,
    WakeFieldObservation,
)
from blond.physics.profiles import DynamicProfileConstNBins

simulation = Mock(
    Simulation,
)
simulation.ring.n_cavities = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.section_i = DynamicParameter(None)
simulation.section_i.current_group = 0
simulation.turn_i = DynamicParameter(None)
simulation.turn_i.value = 0
beam = Mock(BeamBaseClass)
beam.common_array_size = 128
beam.reference_time = 0.8
beam.reference_beta = 0.9
beam.reference_total_energy = 11
beam._dt = np.ones(beam.common_array_size, dtype=float)
beam._dE = np.ones(beam.common_array_size, dtype=float)
beam._flags = np.ones(beam.common_array_size, dtype=int)


class ObservablesHelper(ObservablesEndOfTurnBase):
    def update(self, simulation: Simulation) -> None:
        pass

    def to_disk(self) -> None:
        pass

    def from_disk(self) -> None:
        pass


class TestObservables(unittest.TestCase):
    def setUp(self) -> None:
        self.observables = ObservablesHelper(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.observables = ObservablesHelper(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test_from_disk(self) -> None:
        self.observables.on_init_simulation(
            simulation=simulation,
        )
        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.observables.update(
            simulation=simulation,
        )
        self.observables.to_disk()

        self.observables.from_disk()

    def test_on_run_simulation(self) -> None:
        self.observables.on_init_simulation(
            simulation=simulation,
        )
        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
            obs_per_turn=1,
        )

        assert len(self.observables._turns_array) == self.observables._n_turns
        assert np.all(np.where(np.diff(self.observables._turns_array) <= 0) == np.array([])) # monotonic increase
        assert np.mean(np.diff(self.observables._turns_array)) == 1
        assert np.all(self.observables._section_indices_to_observe == np.array([0]))  # only first one is selected

        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
            obs_per_turn=2,
        )

        assert len(self.observables._turns_array) == self.observables._n_turns * 2
        assert np.all(np.where(np.diff(self.observables._turns_array) <= 0) == np.array([]))  # monotonic increase
        assert np.isclose(np.mean(np.diff(self.observables._turns_array)), 0.5)
        assert np.all(self.observables._section_indices_to_observe == np.array([0, 1]))  # only first one is selected

        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=50,
            n_turns=100,
            obs_per_turn=2,
        )

        assert len(self.observables._turns_array) == self.observables._n_turns * 2
        assert np.all(np.where(np.diff(self.observables._turns_array) <= 0) == np.array([]))  # monotonic increase
        assert np.isclose(np.mean(np.diff(self.observables._turns_array)), 0.5)
        assert np.all(self.observables._section_indices_to_observe == np.array([0, 1]))  # only first one is selected

    def test_on_run_simulation_warnings(self):
        self.observables.on_init_simulation(
            simulation=simulation,
        )

        with self.assertWarnsRegex(UserWarning, "obs_per_turn must be greater"):
            self.observables.on_run_simulation(
                simulation=simulation,
                beam=beam,
                turn_i_init=0,
                n_turns=100,
                obs_per_turn=-1,
            )
        with self.assertWarnsRegex(UserWarning, "obs_per_turn must be smaller"):
            self.observables.on_run_simulation(
                simulation=simulation,
                beam=beam,
                turn_i_init=0,
                n_turns=100,
                obs_per_turn=3,
            )


class TestBunchObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.bunch_observation = BeamObservationEndOfTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam,
        )

    def test___init__(self) -> None:
        self.bunch_observation = BeamObservationEndOfTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam,
        )

    def test_from_disk(self) -> None:
        self.bunch_observation.on_init_simulation(
            simulation=simulation,
        )
        self.bunch_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.bunch_observation.update(
            simulation=simulation,
        )
        self.bunch_observation.to_disk()
        self.bunch_observation.from_disk()


class TestCavityPhaseObservation(unittest.TestCase):
    def setUp(self) -> None:
        cavity = Mock(
            SingleHarmonicCavity,
        )
        cavity.n_rf = 12
        cavity.phi_rf = 1
        cavity.delta_phi_rf = 1
        cavity._omega_rf = 1
        cavity.delta_omega_rf = 1
        cavity.voltage = 1
        self.cavity_phase_observation = CavityPhaseObservation(
            each_turn_i=1,
            cavity=cavity,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.cavity_phase_observation = CavityPhaseObservation(
            each_turn_i=1,
            cavity=Mock(
                SingleHarmonicCavity,
                folder=callers_relative_path("results/", stacklevel=1),
            ),
        )

    def test_from_disk(self) -> None:
        self.cavity_phase_observation.on_init_simulation(
            simulation=simulation,
        )
        self.cavity_phase_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.cavity_phase_observation.update(
            simulation=simulation,
        )
        self.cavity_phase_observation.to_disk()

        self.cavity_phase_observation.from_disk()


class TestStaticProfileObservation(unittest.TestCase):
    def setUp(self) -> None:
        profile = Mock(StaticProfile)
        profile.n_bins = 12
        profile._hist_y = np.ones(profile.n_bins, dtype=float)

        self.static_profile_observation = StaticProfileObservation(
            each_turn_i=1,
            profile=profile,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.static_profile_observation = StaticProfileObservation(
            each_turn_i=1,
            profile=Mock(StaticProfile),
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test_from_disk(self) -> None:
        self.static_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.static_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.static_profile_observation.update(
            simulation=simulation,
        )
        self.static_profile_observation.to_disk()

        self.static_profile_observation.from_disk()

    def test_update(self):
        self.static_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.static_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.static_profile_observation._section_indices_to_observe = np.array([0])
        self.static_profile_observation.update(simulation=simulation)

        prof = deepcopy(self.static_profile_observation)
        before_len = len(prof.hist_y)
        prof._profile._hist_y += 1
        prof.update(simulation=simulation)

        assert len(prof.hist_y) == before_len  # no update since we already had this turn


class TestWakeFieldObservation(unittest.TestCase):
    def setUp(self) -> None:
        wakefield = Mock(WakeField)
        wakefield._profile = Mock(StaticProfile)
        wakefield._profile.n_bins = 12
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=wakefield,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        wakefield.induced_voltage = np.ones(
            wakefield._profile.n_bins, dtype=float
        )

    def test___init__(self) -> None:
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=Mock(WakeField),
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test_from_disk(self) -> None:
        self.wake_field_observation.on_init_simulation(
            simulation=simulation,
        )
        self.wake_field_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.wake_field_observation.update(
            simulation=simulation,
        )
        self.wake_field_observation.to_disk()
        self.wake_field_observation.from_disk()

class TestDynamicProfileConstNBinsObservation(unittest.TestCase):
    def setUp(self) -> None:
        profile = Mock(DynamicProfileConstNBins)
        profile.n_bins = 12
        profile._hist_y = np.ones(profile.n_bins, dtype=float)
        profile._hist_x = np.arange(profile.n_bins, dtype=float)

        self.dynamic_profile_observation = DynamicProfileConstNBinsObservation(
            each_turn_i=1,
            profile=profile,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.dynamic_profile_observation = DynamicProfileConstNBinsObservation(
            each_turn_i=1,
            profile=Mock(DynamicProfileConstNBins),
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test_from_disk(self) -> None:
        self.dynamic_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.dynamic_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.dynamic_profile_observation.update(
            simulation=simulation,
        )
        self.dynamic_profile_observation.to_disk()

        self.dynamic_profile_observation.from_disk()


class TestStaticMultiProfileObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Mock(StaticProfile)
        self.profile.n_bins = 12
        self.profile._hist_y = np.ones(self.profile.n_bins, dtype=float)

        self.profile_2 = Mock(StaticProfile)
        self.profile_2.n_bins = 12
        self.profile_2._hist_y = np.ones(self.profile_2.n_bins, dtype=float)

        self.static_multi_profile_observation = StaticMultiProfileObservation(
            each_turn_i=1,
            profiles=[self.profile, self.profile_2],
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.static_multi_profile_observation = StaticMultiProfileObservation(
            each_turn_i=1,
            profiles=[self.profile, self.profile_2],
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test__error_throwing_wrong_length(self) -> None:
        wrong_profile = deepcopy(self.profile_2)
        wrong_profile.n_bins += 1
        with self.assertRaisesRegex(AssertionError, "n_bins"):
            self.static_multi_profile_observation = StaticMultiProfileObservation(
                each_turn_i=1,
                profiles=[self.profile, wrong_profile],
                folder=callers_relative_path("results/", stacklevel=1),
            )

    def test_from_disk(self) -> None:
        self.static_multi_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.static_multi_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            obs_per_turn=2,
            n_turns=100,
        )
        self.static_multi_profile_observation.update(
            simulation=simulation,
        )
        self.static_multi_profile_observation.to_disk()

        self.static_multi_profile_observation.from_disk()


if __name__ == "__main__":
    unittest.main()
