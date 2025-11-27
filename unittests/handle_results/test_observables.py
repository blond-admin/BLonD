import unittest
from copy import deepcopy
from unittest.mock import Mock, PropertyMock

import numpy as np

from blond import Simulation, SingleHarmonicRfStation, StaticProfile, WakeField
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BeamObservationOncePerTurn,
    CavityPhaseObservation,
    DynamicProfileConstNBinsObservation,
    MultiBunchObservationMetaParams,
    ObservablesOncePerTurnBase,
    StaticMultiProfileObservation,
    StaticProfileObservation,
    WakeFieldObservation,
)
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.physics.profiles import DynamicProfileConstNBins

simulation = Mock(
    Simulation,
)
simulation.ring.n_cavities = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.section_i = DynamicParameter(None)
simulation.section_i.value = 0
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


class ObservablesHelper(ObservablesOncePerTurnBase):
    def update(self, simulation: Simulation) -> None:
        pass

    def to_disk(self) -> None:
        pass

    def from_disk(self) -> None:
        pass


class TestDenseArrayRecorder(unittest.TestCase):
    def test___init__(self):
        DenseArrayRecorder(
            filepath=callers_relative_path("not_exists.txt", stacklevel=1),
            shape=(
                1,
                1,
            ),
            dtype=float,
            overwrite=True,
        )

    def test___init___warns(self):
        with self.assertWarns(UserWarning):
            DenseArrayRecorder(
                filepath=callers_relative_path(
                    "resources/exists", stacklevel=1
                ),
                shape=(
                    1,
                    1,
                ),
                dtype=float,
                overwrite=False,
            )


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
        )

        assert len(self.observables._turns_array) == self.observables._n_turns + 2
        assert np.all(
            np.where(np.diff(self.observables._turns_array) <= 0)
            == np.array([])
        )  # monotonic increase
        assert np.mean(np.diff(self.observables._turns_array[1:])) == 1

        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )

    def test_rename(self) -> None:
        self.observables = ObservablesHelper(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )

        self.observables.on_init_simulation(
            simulation=simulation,
        )
        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )

        # without recorders, should run through
        orig_name = self.observables.common_filepath
        self.observables.rename(orig_name + "_1")
        assert self.observables.common_filepath == orig_name + "_1"

        self.observables._example_densr_arr_rec = DenseArrayRecorder(
            f"{orig_name}_1_example_densr_arr_rec", (1, 1)
        )
        recorders = self.observables.get_recorders()
        assert len(recorders) == 1

        recorders[0][1].filepath = "Notsame"
        with self.assertRaises(NameError):
            self.observables.rename("")
        recorders[0][1].filepath = orig_name + "_1"
        self.observables.rename(orig_name + "_2")
        assert self.observables.common_filepath == orig_name + "_2"

    def test_assert_lateinit_fail(self) -> None:
        obs_helper = ObservablesHelper(each_turn_i=0)

        obs_helper.dummy_value = None

        with self.assertRaises(AssertionError):
            obs_helper.get_recorders()
        with self.assertRaises(AssertionError):
            obs_helper.assert_lateinit()


class TestBunchObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.bunch_observation = BeamObservationOncePerTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam,
        )

    def test___init__(self) -> None:
        self.bunch_observation = BeamObservationOncePerTurn(
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


class TestMultiBunchObservationMetaParams(unittest.TestCase):
    def setUp(self) -> None:
        self.multi_bunch_observation_meta_params = MultiBunchObservationMetaParams(
            t_rf=2,
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam,
        )

    def test___init__(self) -> None:
        self.multi_bunch_observation_meta_params = MultiBunchObservationMetaParams(
            t_rf=2,
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam,
        )

    def test_from_disk_single_bunch(self) -> None:
        self.multi_bunch_observation_meta_params.on_init_simulation(
            simulation=simulation,
        )
        self.multi_bunch_observation_meta_params.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.multi_bunch_observation_meta_params.update(
            simulation=simulation,
        )
        self.multi_bunch_observation_meta_params.to_disk()
        self.multi_bunch_observation_meta_params.from_disk()

        emittance = np.sqrt(np.mean(beam._dE ** 2)
                              * np.mean(beam._dt ** 2)
                              - np.mean(beam._dE * beam._dt) ** 2)
        assert np.isclose(self.multi_bunch_observation_meta_params.rms_emittance,
                          emittance)
        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dE,
                          np.std(beam._dE))
        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dt,
                          np.std(beam._dt))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dE,
                          np.mean(beam._dE))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dt,
                          np.mean(beam._dt))

    def test_from_disk_double_bunch(self) -> None:
        # bucket length is 2, dt is at one, push one bucket to
        orig_bunch_length = beam.common_array_size
        beam_local = deepcopy(beam)
        beam_local._dE = np.append(beam_local._dE, beam_local._dE + 2)
        beam_local._dt = np.append(beam_local._dt, beam_local._dt + 2)
        beam_local.common_array_size = len(beam_local._dE)
        self.multi_bunch_observation_meta_params = MultiBunchObservationMetaParams(
            n_bunches=2,
            t_rf=2,
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
            beam=beam_local,
        )

        self.multi_bunch_observation_meta_params.recompute_mask = True
        self.multi_bunch_observation_meta_params.on_init_simulation(
            simulation=simulation,
        )
        self.multi_bunch_observation_meta_params.on_run_simulation(
            simulation=simulation,
            beam=beam,
            turn_i_init=0,
            n_turns=100,
        )
        self.multi_bunch_observation_meta_params.update(
            simulation=simulation,
        )
        self.multi_bunch_observation_meta_params.to_disk()
        self.multi_bunch_observation_meta_params.from_disk()

        emittance_1 = np.sqrt(np.mean(beam_local._dE[0:orig_bunch_length] ** 2)
                              * np.mean(beam_local._dt[0:orig_bunch_length] ** 2)
                              - np.mean(beam_local._dE[0:orig_bunch_length] * beam_local._dt[0:orig_bunch_length]) ** 2)
        assert np.isclose(self.multi_bunch_observation_meta_params.rms_emittance[0, 0],
                          emittance_1)
        emittance_2 = np.sqrt(np.mean(beam_local._dE[orig_bunch_length:] ** 2)
                              * np.mean(beam_local._dt[orig_bunch_length:] ** 2)
                              - np.mean(beam_local._dE[orig_bunch_length:] * beam_local._dt[orig_bunch_length:]) ** 2)
        assert np.isclose(self.multi_bunch_observation_meta_params.rms_emittance[1, 0],
                          emittance_2)

        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dE[0, 0],
                          np.std(beam_local._dE[0:orig_bunch_length]))
        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dt[0, 0],
                          np.std(beam_local._dt[0:orig_bunch_length]))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dE[0, 0],
                          np.mean(beam_local._dE[0:orig_bunch_length]))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dt[0, 0],
                          np.mean(beam_local._dt[0:orig_bunch_length]))

        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dE[1, 0],
                          np.std(beam_local._dE[orig_bunch_length:]))
        assert np.isclose(self.multi_bunch_observation_meta_params.sigma_dt[1, 0],
                          np.std(beam_local._dt[orig_bunch_length:]))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dE[1, 0],
                          np.mean(beam_local._dE[orig_bunch_length:]))
        assert np.isclose(self.multi_bunch_observation_meta_params.mean_dt[1, 0],
                          np.mean(beam_local._dt[orig_bunch_length:]) -
                          self.multi_bunch_observation_meta_params.t_rf)


class TestCavityPhaseObservation(unittest.TestCase):
    def setUp(self) -> None:
        cavity = Mock(
            SingleHarmonicRfStation,
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
                SingleHarmonicRfStation,
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
        type(profile).hist_y = PropertyMock(
            return_value=np.ones(profile.n_bins, dtype=float)
        )
        # no changeback required, as this is changed on mocked object
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
        simulation.section_i.value = 0
        simulation.turn_i.value = 0
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
        self.static_profile_observation._section_indices_to_observe = np.array(
            [0]
        )
        simulation.section_i.value = 0
        self.static_profile_observation.update(simulation=simulation)

        prof_obs = deepcopy(self.static_profile_observation)
        before_len = len(prof_obs.hist_y)
        prof_obs.update(simulation=simulation)

        assert (
            len(prof_obs.hist_y) == before_len
        )  # no update since we already had this turn


class TestWakeFieldObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.wakefield = Mock(WakeField)
        self.wakefield._profile = Mock(StaticProfile)
        self.wakefield._profile.n_bins = 12
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=self.wakefield,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        type(self.wakefield).induced_voltage = PropertyMock(
            return_value=np.ones(self.wakefield._profile.n_bins, dtype=float)
        )
        # no changeback required, as this is changed on mocked object

    def test___init__(self) -> None:
        self.wake_field_observation = WakeFieldObservation(
            each_turn_i=1,
            wakefield=Mock(WakeField),
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test_error_in_aquisition(self) -> None:
        prof = StaticProfile.from_cutoff(0, 1e-9, 3e9)
        wf = WakeField(
            section_index=0,
            profile=prof,
            sources=Mock(Resonators),
            solver=Mock(SingleTurnResonatorConvolutionSolver),
        )
        wf_obs = WakeFieldObservation(
            wakefield=wf,
            folder=callers_relative_path("results/", stacklevel=1),
            each_turn_i=1,
        )

        wf_obs.on_init_simulation(simulation=simulation)
        wf_obs.on_run_simulation(
            simulation=simulation, beam=beam, turn_i_init=0, n_turns=100
        )

        orig_save = type(wf).induced_voltage
        type(wf).induced_voltage = PropertyMock(
            side_effect=AttributeError("ind_volt_calc_failed")
        )

        simulation.section_i.value = 0
        wf_obs.update(simulation=simulation)

        with self.assertRaises(AttributeError):
            _ = wf.induced_voltage

        type(
            wf
        ).induced_voltage = orig_save  # important, otherwise the type is changed for the entire runtime

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
        simulation.section_i.value = 0
        self.wake_field_observation.update(
            simulation=simulation,
        )
        self.wake_field_observation.to_disk()
        self.wake_field_observation.from_disk()

        assert (
            len(self.wake_field_observation.induced_voltage[0])
            == self.wakefield.induced_voltage.shape[0]
        )
        np.testing.assert_allclose(
            self.wake_field_observation.induced_voltage[0],
            self.wakefield.induced_voltage,
        )


class TestDynamicProfileConstNBinsObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Mock(DynamicProfileConstNBins)
        self.profile.n_bins = 12
        type(self.profile).hist_y = PropertyMock(
            return_value=np.ones(self.profile.n_bins, dtype=float)
        )
        type(self.profile).hist_x = PropertyMock(
            return_value=np.arange(self.profile.n_bins, dtype=float)
        )
        # no changeback required, as this is changed on mocked object

        self.dynamic_profile_observation = DynamicProfileConstNBinsObservation(
            each_turn_i=1,
            profile=self.profile,
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

        assert len(self.dynamic_profile_observation.hist_y) == 1
        assert len(self.dynamic_profile_observation.hist_x) == 1
        assert (
            len(self.dynamic_profile_observation.hist_y[0])
            == self.profile.n_bins
        )
        assert (
            len(self.dynamic_profile_observation.hist_x[0])
            == self.profile.n_bins
        )
        np.testing.assert_allclose(
            self.profile.hist_y, self.dynamic_profile_observation.hist_y[0]
        )
        np.testing.assert_allclose(
            self.profile.hist_x, self.dynamic_profile_observation.hist_x[0]
        )


class TestStaticMultiProfileObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Mock(StaticProfile)
        self.profile.n_bins = 12
        type(self.profile).hist_y = PropertyMock(
            return_value=np.ones(self.profile.n_bins, dtype=float)
        )
        # no changeback required, as this is changed on mocked object
        self.profile.section_index = 0

        self.profile_2 = Mock(StaticProfile)
        self.profile_2.n_bins = 12
        type(self.profile_2).hist_y = PropertyMock(
            return_value=np.ones(self.profile_2.n_bins, dtype=float) * 2
        )
        # no changeback required, as this is changed on mocked object
        self.profile_2.section_index = 1

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
            self.static_multi_profile_observation = (
                StaticMultiProfileObservation(
                    each_turn_i=1,
                    profiles=[self.profile, wrong_profile],
                    folder=callers_relative_path("results/", stacklevel=1),
                )
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
        simulation.section_i.value = 0
        simulation.turn_i.value = 0
        self.static_multi_profile_observation.update(
            simulation=simulation,
        )

        self.static_multi_profile_observation.to_disk()

        self.static_multi_profile_observation.from_disk()

        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[0],
            self.profile.hist_y,
        )
        assert len(self.static_multi_profile_observation.hist_y) == 1

        simulation.section_i.value = 1
        self.static_multi_profile_observation.update(
            simulation=simulation,
        )
        assert len(self.static_multi_profile_observation.hist_y) == 2
        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[1],
            self.profile_2.hist_y,
        )

        # no update if we repeat
        self.static_multi_profile_observation.update(
            simulation=simulation,
        )
        assert len(self.static_multi_profile_observation.hist_y) == 2
        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[1],
            self.profile_2.hist_y,
        )


if __name__ == "__main__":
    unittest.main()
