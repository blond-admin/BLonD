import unittest
from copy import deepcopy
from unittest.mock import Mock, PropertyMock

import numpy as np

from blond import (
    Beam,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    electron,
)
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BeamObservationOncePerTurn,
    BeamStatisticsOncePerTurn,
    DynamicProfileConstNBinsObservation,
    ObservablesOncePerTurnBase,
    RFStationPhaseObservation,
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
simulation.ring.n_rf_stations = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.section_i = DynamicParameter(None)
simulation.section_i.current_group = 0
simulation.turn_i = DynamicParameter(None)
simulation.turn_i.value = 0
beam = Mock(BeamBaseClass)
beam._dE = Mock(DistributedArray)
beam._dt = Mock(DistributedArray)
beam._flags = Mock(DistributedArray)
beam.common_array_size = 128
beam.reference = Mock(ReferenceCoordinates)
beam.reference.time = 0.8
beam.reference.beta = 0.9
beam.reference.total_energy = 11
beam._dt.array_local = np.ones(beam.common_array_size, dtype=float)
beam._dE.array_local = np.ones(beam.common_array_size, dtype=float)
beam._flags.array_local = np.ones(beam.common_array_size, dtype=int)
beam.read_partial_dt.return_value = beam._dt.array_local
beam.read_partial_dE.return_value = beam._dE.array_local
beam.read_partial_flags.return_value = beam._flags.array_local


class ObservablesHelper(ObservablesOncePerTurnBase):
    def _update(self) -> None:
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
        self.assertEqual(self.observables.each_turn_i, 1)
        self.assertEqual(
            self.observables.common_filepath,
            callers_relative_path("results/", stacklevel=1) + "last",
        )

    def test_from_disk(self) -> None:
        self.observables.on_init_simulation(
            simulation=simulation,
        )
        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        self.observables.update()
        self.observables.to_disk()

        self.observables.from_disk()

    def test_on_run_simulation(self) -> None:
        self.observables.on_init_simulation(
            simulation=simulation,
        )
        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )

        assert len(self.observables._turns_array) == (
            self.observables._n_turns + 1
        )
        assert np.all(
            np.where(np.diff(self.observables._turns_array) <= 0)
            == np.array([])
        )  # monotonic increase
        assert np.mean(np.diff(self.observables._turns_array[:])), 1

        self.observables.on_run_simulation(
            simulation=simulation,
            beam=beam,
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
        )

        self.beam = Beam(
            intensity=100,
            particle_type=electron,
        )
        common_array_size = 128
        self.beam.reference.time = 0.8
        # self.beam.reference.beta = 0.9
        self.beam.reference.total_energy = 11
        self.beam._dt = np.ones(common_array_size, dtype=float)
        self.beam._dE = np.ones(common_array_size, dtype=float)
        self.beam._flags = np.ones(common_array_size, dtype=int)

        self.beam.setup_beam(
            dE=np.ones(common_array_size, dtype=float),
            dt=np.ones(common_array_size, dtype=float),
            reference_time=0.8,
            reference_total_energy=11,
        )

    def test___init__(self) -> None:
        self.assertEqual(self.bunch_observation.each_turn_i, 1)
        self.assertEqual(
            self.bunch_observation.common_filepath,
            callers_relative_path("results/", stacklevel=1) + "last",
        )

    def test_from_disk(self) -> None:
        self.bunch_observation.on_init_simulation(
            simulation=simulation,
        )
        self.bunch_observation.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        self.bunch_observation.update()

        # test properties
        np.testing.assert_almost_equal(
            self.bunch_observation.dts,
            self.bunch_observation._dts.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_observation.dEs,
            self.bunch_observation._dEs.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_observation.flags,
            self.bunch_observation._flags.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_observation.reference_time,
            self.bunch_observation._reference_time.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_observation.reference_total_energy,
            self.bunch_observation._reference_total_energy.get_valid_entries(),
        )

        self.bunch_observation.to_disk()

        to_compare = BeamObservationOncePerTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        to_compare.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        to_compare.from_disk()

        np.testing.assert_almost_equal(
            to_compare.dts, self.bunch_observation.dts
        )
        np.testing.assert_almost_equal(
            to_compare.dEs, self.bunch_observation.dEs
        )
        np.testing.assert_almost_equal(
            to_compare.flags, self.bunch_observation.flags
        )
        np.testing.assert_almost_equal(
            to_compare.reference_time, self.bunch_observation.reference_time
        )
        np.testing.assert_almost_equal(
            to_compare.reference_total_energy,
            self.bunch_observation.reference_total_energy,
        )


class TestBunchStatistics(unittest.TestCase):
    def setUp(self) -> None:
        self.beam = Beam(
            intensity=100,
            particle_type=electron,
        )
        common_array_size = 128
        self.beam.reference_time = 0.8
        self.beam.reference_beta = 0.9
        self.beam.reference_total_energy = 11
        self.beam._dt = np.ones(common_array_size, dtype=float)
        self.beam._dE = np.ones(common_array_size, dtype=float)
        self.beam._flags = np.ones(common_array_size, dtype=int)
        self.beam.setup_beam(
            dE=np.ones(common_array_size, dtype=float),
            dt=np.ones(common_array_size, dtype=float),
            reference_time=0.8,
            reference_total_energy=11,
        )

        self.bunch_statistics = BeamStatisticsOncePerTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.assertEqual(self.bunch_statistics.each_turn_i, 1)
        self.assertEqual(
            self.bunch_statistics.common_filepath,
            callers_relative_path("results/", stacklevel=1) + "last",
        )

    def test_from_disk(self) -> None:
        self.bunch_statistics.on_init_simulation(
            simulation=simulation,
        )
        self.bunch_statistics.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        self.bunch_statistics.update()

        # test properties
        np.testing.assert_almost_equal(
            self.bunch_statistics.bunch_position,
            self.bunch_statistics._bunch_position.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_statistics.energy_spread,
            self.bunch_statistics._energy_spread.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_statistics.bunch_length,
            self.bunch_statistics._bunch_length.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_statistics.reference_time,
            self.bunch_statistics._reference_time.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.bunch_statistics.reference_total_energy,
            self.bunch_statistics._reference_total_energy.get_valid_entries(),
        )

        self.bunch_statistics.to_disk()

        to_compare = BeamStatisticsOncePerTurn(
            each_turn_i=1,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        to_compare.on_run_simulation(
            simulation=simulation,
            beam=self.beam,
            n_turns=100,
        )
        to_compare.from_disk()

        np.testing.assert_almost_equal(
            to_compare.bunch_position, self.bunch_statistics.bunch_position
        )
        np.testing.assert_almost_equal(
            to_compare.energy_spread, self.bunch_statistics.energy_spread
        )
        np.testing.assert_almost_equal(
            to_compare.bunch_length, self.bunch_statistics.bunch_length
        )
        np.testing.assert_almost_equal(
            to_compare.reference_time, self.bunch_statistics.reference_time
        )
        np.testing.assert_almost_equal(
            to_compare.reference_total_energy,
            self.bunch_statistics.reference_total_energy,
        )


class TestRFStationPhaseObservation(unittest.TestCase):
    def setUp(self) -> None:
        rf_station = Mock(
            SingleHarmonicRFStation,
        )
        rf_station.n_rf = 12
        rf_station.phi_rf = 1
        rf_station.delta_phi_rf = 1
        rf_station._omega_rf = 1
        rf_station.delta_omega_rf = 1
        rf_station.voltage = 1
        self.rf_station_phase_observation = RFStationPhaseObservation(
            each_turn_i=1,
            rf_station=rf_station,
            folder=callers_relative_path("results/", stacklevel=1),
        )

    def test___init__(self) -> None:
        self.rf_station_phase_observation = RFStationPhaseObservation(
            each_turn_i=1,
            rf_station=Mock(
                SingleHarmonicRFStation,
                folder=callers_relative_path("results/", stacklevel=1),
            ),
        )

    def test_from_disk(self) -> None:
        self.rf_station_phase_observation.on_init_simulation(
            simulation=simulation,
        )
        self.rf_station_phase_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        self.rf_station_phase_observation.update()
        self.rf_station_phase_observation.to_disk()

        # test properties
        np.testing.assert_almost_equal(
            self.rf_station_phase_observation.phases,
            self.rf_station_phase_observation._phases.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.rf_station_phase_observation.omegas,
            self.rf_station_phase_observation._omegas.get_valid_entries(),
        )
        np.testing.assert_almost_equal(
            self.rf_station_phase_observation.voltages,
            self.rf_station_phase_observation._voltages.get_valid_entries(),
        )

        self.rf_station_phase_observation.to_disk()

        to_compare = RFStationPhaseObservation(
            each_turn_i=1,
            rf_station=self.rf_station_phase_observation._rf_station,
            folder=callers_relative_path("results/", stacklevel=1),
        )
        to_compare.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        to_compare.from_disk()

        np.testing.assert_almost_equal(
            to_compare.phases, self.rf_station_phase_observation.phases
        )
        np.testing.assert_almost_equal(
            to_compare.omegas, self.rf_station_phase_observation.omegas
        )
        np.testing.assert_almost_equal(
            to_compare.voltages, self.rf_station_phase_observation.voltages
        )


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
            n_turns=100,
        )
        simulation.section_i.value = 0
        simulation.turn_i.value = 0
        self.static_profile_observation.update()
        self.static_profile_observation.to_disk()

        self.static_profile_observation.from_disk()

    def test_update(self):
        self.static_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.static_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        self.static_profile_observation._section_indices_to_observe = np.array(
            [0]
        )
        simulation.section_i.value = 0
        self.static_profile_observation.update()
        with self.assertRaisesRegex(
            RuntimeError,
            "already called update in this turn for turn",
        ):
            self.static_profile_observation.update()

        assert len(self.static_profile_observation.hist_y) == 1


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
        wf_obs.on_run_simulation(simulation=simulation, beam=beam, n_turns=100)

        orig_save = type(wf).induced_voltage
        type(wf).induced_voltage = PropertyMock(
            side_effect=AttributeError("ind_volt_calc_failed")
        )

        simulation.section_i.value = 0
        wf_obs.update()

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
            n_turns=100,
        )
        simulation.section_i.value = 0
        self.wake_field_observation.update()
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
            n_turns=100,
        )
        self.dynamic_profile_observation.update()
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
            n_turns=100,
        )
        simulation.section_i.value = 0
        simulation.turn_i.value = 0
        self.static_multi_profile_observation.update()

        self.static_multi_profile_observation.to_disk()

        self.static_multi_profile_observation.from_disk()

        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[0][0],
            self.profile.hist_y,
        )
        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[0][1],
            self.profile_2.hist_y,
        )
        assert len(self.static_multi_profile_observation.hist_y) == 1
        assert (
            len(self.static_multi_profile_observation.hist_y[0]) == 2
        )  # two profiles per turn

        simulation.turn_i.value = 1
        self.static_multi_profile_observation.update()
        assert len(self.static_multi_profile_observation.hist_y) == 2

        # no update if we repeat
        with self.assertRaisesRegex(
            RuntimeError,
            "already called update in this turn for turn",
        ):
            self.static_multi_profile_observation.update()
        assert len(self.static_multi_profile_observation.hist_y) == 2
        np.testing.assert_allclose(
            self.static_multi_profile_observation.hist_y[1][1],
            self.profile_2.hist_y,
        )


if __name__ == "__main__":
    unittest.main()
