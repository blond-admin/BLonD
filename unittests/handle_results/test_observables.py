import unittest
from unittest.mock import Mock

import numpy as np

from blond import Simulation, SingleHarmonicCavity, StaticProfile, WakeField
from blond._core.beam.base import BeamBaseClass
from blond.handle_results.observables import (
    BunchObservation,
    CavityPhaseObservation,
    Observables,
    StaticProfileObservation,
    WakeFieldObservation,
)

simulation = Mock(
    Simulation,
)
beam = Mock(BeamBaseClass)
beam.common_array_size = 128
beam.reference_time = 0.8
beam.reference_beta = 0.9
beam.reference_total_energy = 11
beam._dt = np.ones(beam.common_array_size, dtype=float)
beam._dE = np.ones(beam.common_array_size, dtype=float)
beam._flags = np.ones(beam.common_array_size, dtype=int)


class ObservablesHelper(Observables):
    def update(self, simulation: Simulation, beam: BeamBaseClass) -> None:
        pass

    def to_disk(self) -> None:
        pass

    def from_disk(self) -> None:
        pass


class TestObservables(unittest.TestCase):
    def setUp(self) -> None:
        self.observables = ObservablesHelper(
            each_turn_i=1,
        )

    def test___init__(self) -> None:
        self.observables = ObservablesHelper(
            each_turn_i=1,
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
            beam=beam,
        )
        self.observables.to_disk()

        self.observables.from_disk()


class TestBunchObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.bunch_observation = BunchObservation(
            each_turn_i=1,
        )

    def test___init__(self) -> None:
        self.bunch_observation = BunchObservation(
            each_turn_i=1,
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
            beam=beam,
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
        )

    def test___init__(self) -> None:
        self.cavity_phase_observation = CavityPhaseObservation(
            each_turn_i=1,
            cavity=Mock(
                SingleHarmonicCavity,
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
            beam=beam,
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
        )

    def test___init__(self) -> None:
        self.static_profile_observation = StaticProfileObservation(
            each_turn_i=1,
            profile=Mock(StaticProfile),
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
            beam=beam,
        )
        self.static_profile_observation.to_disk()

        self.static_profile_observation.from_disk()


class TestWakeFieldObservation(unittest.TestCase):
    def setUp(self) -> None:
        wakefield = Mock(WakeField)
        wakefield._profile = Mock(StaticProfile)
        wakefield._profile.n_bins = 12
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=wakefield,
        )
        wakefield.induced_voltage = np.ones(
            wakefield._profile.n_bins, dtype=float
        )

    def test___init__(self) -> None:
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=Mock(WakeField),
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
            beam=beam,
        )
        self.wake_field_observation.to_disk()
        self.wake_field_observation.from_disk()


if __name__ == "__main__":
    unittest.main()
