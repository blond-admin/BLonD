import unittest
from copy import deepcopy
from unittest.mock import Mock, PropertyMock

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    BoxLosses,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    electron,
    momentum_compaction_factor,
    proton,
)
from blond.core.base import DynamicParameter
from blond.core.beam.base import BeamBaseClass
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BeamHist2dOncePerTurn,
    BeamObservationOncePerTurn,
    BeamStatisticsOncePerTurn,
    DriftObservation,
    DynamicProfileConstNBinsObservation,
    IQCavityFeedbackObservation,
    ObservablesOncePerTurnBase,
    RFStationPhaseObservation,
    SimulationObservation,
    StaticMultiProfileObservation,
    StaticProfileObservation,
    WakeFieldObservation,
)
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.physics.profiles import DynamicProfileConstNBins
from blond.utilities.separatrix.symbolic_separatrix import (
    SymbolicSeparatrixHelper,
)

simulation = Mock(
    Simulation,
)
simulation.ring.n_rf_stations = 2
simulation.ring.section_lengths = [250, 250]
simulation.ring.circumference = 500
simulation.turn_counter = DynamicParameter(None)
simulation.turn_counter.value = 0
simulation.current_t_rev = 123
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
beam.dt_min = 1
beam.dt_max = 2
sep_helper = Mock(SymbolicSeparatrixHelper)
dE_sep = np.ones(256)
sep_helper.get_separatrix.return_value = np.stack([dE_sep, -dE_sep])
simulation._get_separatrix_helper.return_value = sep_helper


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
            self.observables._n_turns
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


class TestBeamObservation(unittest.TestCase):
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

    def test_lossy_simulation(self):
        DEV_DRAW = False
        for intensity in (0, 1e9):
            ring = Ring(26658.883)

            rf_station = SingleHarmonicRFStation()
            rf_station.harmonic = 35640
            rf_station.voltage = 6e6
            rf_station.phi_rf_design = 0

            N_TURNS = int(1e3)

            energy_cycle = MagneticCyclePerTurn(
                value_init=450e9,
                values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
                reference_particle=proton,
            )

            drift1 = DriftSimple(
                orbit_length=26658.883,
            )
            drift1.momentum_compaction_factor = momentum_compaction_factor(
                transition_gamma=55.759505
            )
            loss_box = BoxLosses(  # This is required to test the observable with losses
                purge_flagged_macroparticles=True,
                t_min=0,
                t_max=2.5e-9,
            )

            beam1 = Beam(
                intensity=intensity,
                particle_type=proton,
            )

            sim = Simulation.from_locals(locals())
            sim.print_one_turn_execution_order()
            sim.prepare_beam(
                beam=beam1,
                preparation_routine=BiGaussian(
                    sigma_dt=0.4e-9 / 4,
                    sigma_dE=1e9 / 4,
                    reinsertion=False,
                    seed=1,
                    n_macroparticles=1e3,
                ),
            )

            phase_observation = RFStationPhaseObservation(
                each_turn_i=1,
                rf_station=rf_station,
            )
            bunch_observation = BeamObservationOncePerTurn(each_turn_i=2)
            obs_beam_hist2d = BeamHist2dOncePerTurn(
                each_turn_i=2, bins=128 if intensity == 0 else (128, 64)
            )
            beam1._is_distributed = True
            with self.assertRaisesRegex(
                NotImplementedError, "This needs to be implement"
            ):
                sim.run_simulation(
                    beams=(beam1,),
                    n_turns=N_TURNS // 50,
                    observe=(obs_beam_hist2d,),
                )
            with self.assertRaisesRegex(
                NotImplementedError, "This needs to be implement"
            ):
                sim.run_simulation(
                    beams=(beam1,),
                    n_turns=N_TURNS // 50,
                    observe=(bunch_observation,),
                )
            beam1._is_distributed = False
            sim.run_simulation(
                beams=(beam1,),
                n_turns=N_TURNS,
                observe=(
                    phase_observation,
                    bunch_observation,
                    obs_beam_hist2d,
                ),
            )
            if DEV_DRAW:
                plt.plot(phase_observation.phases)
                plt.figure()
                for i in range(bunch_observation.dts.shape[0]):
                    plt.clf()
                    sel = ~np.isnan(bunch_observation.dts[i, :])
                    plt.hist2d(
                        bunch_observation.dts[i, sel],
                        bunch_observation.dEs[i, sel],
                        bins=256,
                        # range=[[0, 2.5e-9], [-4e8, 4e8]],
                    )
                obs_beam_hist2d.plot(result_idx=-1)
                obs_beam_hist2d.plot_fancy(result_idx=-1)


class TestBunchStatistics(unittest.TestCase):
    def setUp(self) -> None:
        self.beam = Beam(
            intensity=100,
            particle_type=electron,
        )
        common_array_size = 128
        self.beam.reference._time = 0.8
        self.beam.reference._beta = 0.9
        self.beam.reference._total_energy = 11
        self.beam._dt = DistributedArray(
            np.ones(common_array_size, dtype=float)
        )
        self.beam._dE = DistributedArray(
            np.ones(common_array_size, dtype=float)
        )
        self.beam._flags = DistributedArray(
            np.ones(common_array_size, dtype=int)
        )
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
        rf_station.omega_rf = 1
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
        type(profile).hist_x = PropertyMock(
            return_value=np.arange(profile.n_bins, dtype=float)
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
        simulation.turn_counter.value = 0
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
        self.static_profile_observation.update()
        with self.assertRaisesRegex(
            RuntimeError,
            "already called update in this turn for turn",
        ):
            self.static_profile_observation.update()

        assert len(self.static_profile_observation.hist_y) == 1

    def test_plot_waterfall(self) -> None:
        self.static_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.static_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        self.static_profile_observation.update()

        from matplotlib.collections import QuadMesh

        mesh = self.static_profile_observation.plot_waterfall()
        assert isinstance(mesh, QuadMesh)
        np.testing.assert_allclose(
            mesh.get_array().data.ravel(),
            self.static_profile_observation.hist_y.ravel(),
        )
        plt.close(mesh.axes.figure)


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

    def test_plot_waterfall(self) -> None:
        self.dynamic_profile_observation.on_init_simulation(
            simulation=simulation
        )
        self.dynamic_profile_observation.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )
        self.dynamic_profile_observation.update()

        from matplotlib.collections import QuadMesh

        mesh = self.dynamic_profile_observation.plot_waterfall()
        assert isinstance(mesh, QuadMesh)
        np.testing.assert_allclose(
            mesh.get_array().data.ravel(),
            self.dynamic_profile_observation.hist_y.ravel(),
        )
        plt.close(mesh.axes.figure)


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

    def test_init_sort_profiles_by_section_false(self) -> None:
        obs = StaticMultiProfileObservation(
            each_turn_i=1,
            profiles=[self.profile_2, self.profile],
            folder=callers_relative_path("results/", stacklevel=1),
            sort_profiles_by_section=False,
        )
        self.assertIs(obs._profiles[0], self.profile_2)
        self.assertIs(obs._profiles[1], self.profile)

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
        simulation.turn_counter.value = 0
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

        simulation.turn_counter.value = 1
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


class TestSimulationObservation(unittest.TestCase):
    def setUp(self):
        self.obs = SimulationObservation(each_turn_i=2)

    def test___init__(self):
        pass  # done by setup

    def test_on_run(self):
        self.obs.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )

    def test_update(self):
        self.obs.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=10,
        )
        self.obs._update()
        self.obs._update()
        self.assertEqual(self.obs.t_revs[0], 123)
        self.assertEqual(len(self.obs.t_revs), 2)  # two updates before


class TestIQCavityFeedbackObservation(unittest.TestCase):
    def test_coarse_recorder_filepaths_are_pairwise_distinct(self):
        # Regression: _v_ant_coarse, _i_beam_coarse and _i_gen_coarse were
        # all constructed with the "_v_ant_coarse" suffix, so the three
        # recorders silently overwrote each other's file on disk.
        feedback = Mock()
        feedback.profile.n_bins = 16
        feedback.harmonic = 8.0
        feedback.n_rf_periods_per_coarse_grid = 1
        feedback.n_rf_stations_in_ring = 1

        sim = Mock(Simulation)
        sim.ring.elements.get_elements.return_value = [Mock()]

        observation = IQCavityFeedbackObservation(
            each_turn_i=1, feedback=feedback
        )
        observation.on_run_simulation(simulation=sim, beam=Mock(), n_turns=4)

        filepaths = [
            observation._v_ant_coarse.filepath,
            observation._i_beam_coarse.filepath,
            observation._i_gen_coarse.filepath,
        ]
        self.assertEqual(len(set(filepaths)), len(filepaths), filepaths)

    def test_coarse_width_uses_the_feedbacks_station_count(self):
        """The overshoot divisor comes from the feedback, not a ring lookup.

        The coarse buffer is one turn plus one section of overshoot, so the
        divisor is the number of RF stations in the ring. Re-deriving it here
        by filtering for ``SingleHarmonicRFStation`` divided by zero on a ring
        whose stations are all ``MultiHarmonicRFStation`` (the two are
        siblings, not parent and child) -- a supported configuration, since a
        feedback may regulate one harmonic slot of a multi-harmonic station.
        The feedback already counts stations against ``RFStationBaseClass``,
        so the observation asks it instead of guessing.
        """
        feedback = Mock()
        feedback.profile.n_bins = 16
        feedback.harmonic = 8.0
        feedback.n_rf_periods_per_coarse_grid = 1
        feedback.n_rf_stations_in_ring = 2

        sim = Mock(Simulation)
        # A ring whose RF stations are all multi-harmonic: the old
        # SingleHarmonicRFStation filter yields an empty list here.
        sim.ring.elements.get_elements.return_value = []

        observation = IQCavityFeedbackObservation(
            each_turn_i=1, feedback=feedback
        )
        observation.on_run_simulation(simulation=sim, beam=Mock(), n_turns=4)

        # Analytic one-turn-plus-overshoot estimate with the feedback's
        # station count (1 + 1/2), plus the safety margin of one cell per
        # possible segment (n_stations + 1 segments).
        self.assertEqual(
            observation.len_coarse_max,
            int(np.ceil(1.5 * 8.0)) + 2 + 1,
        )

    @staticmethod
    def _feedback_stub(
        rf_centers_lengths: tuple[int, ...],
        harmonic: float = 8.0,
        n_stations: int = 1,
        i_beam_forward: np.ndarray | None = None,
        v_ant: np.ndarray | None = None,
    ) -> Mock:
        """Stub feedback exposing exactly what ``_update`` consumes."""
        n_bins = 4
        total = int(sum(rf_centers_lengths))
        feedback = Mock()
        feedback.profile.n_bins = n_bins
        feedback.harmonic = harmonic
        feedback.n_rf_periods_per_coarse_grid = 1
        feedback.n_rf_stations_in_ring = n_stations
        feedback._rf_centers = np.zeros(total)
        feedback._rf_centers_lengths = np.asarray(
            rf_centers_lengths, dtype=int
        )
        if v_ant is None:
            v_ant = np.ones(total, dtype=complex)
        feedback.antenna_voltage_coarse_grid = v_ant
        feedback.generator_current_coarse_grid = np.ones(total, dtype=complex)
        if i_beam_forward is None:
            i_beam_forward = np.zeros(
                int(rf_centers_lengths[-1]), dtype=complex
            )
        feedback.beam_current_forward_coarse_grid = i_beam_forward
        feedback.antenna_voltage_fine_grid = np.zeros(n_bins, dtype=complex)
        feedback.beam_current_fine_grid = np.zeros(n_bins, dtype=complex)
        feedback.generator_current_fine_grid = np.zeros(n_bins, dtype=complex)
        feedback.relative_voltage_correction = np.zeros(n_bins)
        feedback.phase_correction = np.zeros(n_bins)
        gap = feedback._parent_rf_station.calc_gap_voltage_with_feedbacks
        gap.return_value = np.zeros(n_bins)
        return feedback

    @staticmethod
    def _observation_for(feedback: Mock) -> IQCavityFeedbackObservation:
        sim = Mock(Simulation)
        sim.turn_counter = DynamicParameter(None)
        sim.turn_counter.value = 3
        observation = IQCavityFeedbackObservation(
            each_turn_i=1, feedback=feedback
        )
        observation.on_run_simulation(simulation=sim, beam=Mock(), n_turns=4)
        return observation

    def test_grid_slightly_beyond_analytic_prediction_is_recorded(self):
        """A grid one cell over the analytic prediction must not crash.

        The per-turn coarse grid is produced by ``np.arange`` walks; each
        of the (up to ``n_stations + 1``) segments can yield one cell more
        than the analytic ``(1 + 1/n_stations) * harmonic / n_periods``
        prediction (measured: 51801 cells vs 51800 predicted, 2 sections,
        sub-stepping, station at the section end). The allocation must
        carry a per-segment margin instead of crashing with a raw numpy
        shape error mid-run.
        """
        # harmonic=8, 1 station: analytic prediction = ceil(2 * 8) = 16,
        # actual grid 17 cells -- inside the +-1-per-segment margin.
        feedback = self._feedback_stub(rf_centers_lengths=(13, 4))
        observation = self._observation_for(feedback)
        observation._update()  # must not raise
        row = observation.v_ant_coarse[0]
        self.assertEqual(int(np.count_nonzero(~np.isnan(row))), 17)

    def test_overflowing_grid_raises_descriptive_error(self):
        """A grid beyond allocation + margin raises a descriptive error.

        Not a raw numpy broadcast ``ValueError``: the message names the
        observable, the turn, and the actual vs allocated lengths.
        """
        # harmonic=8, 1 station: allocation is prediction (16) + margin;
        # 40 cells overflow any justified margin.
        feedback = self._feedback_stub(rf_centers_lengths=(36, 4))
        observation = self._observation_for(feedback)
        with self.assertRaisesRegex(
            RuntimeError,
            r"IQCavityFeedbackObservation.*turn 3.*40.*"
            rf"{observation.len_coarse_max}",
        ):
            observation._update()

    def test_beam_current_columns_align_with_voltage_columns(self):
        """``i_beam_coarse`` columns mean the same cell as ``v_ant_coarse``.

        ``beam_current_forward_coarse_grid`` is forward-segment-local while
        the voltage/current spans the whole per-turn grid; the recorder
        must translate by the forward offset
        ``len(_rf_centers) - _rf_centers_lengths[-1]`` like every physics
        consumer does.
        """
        i_beam = np.array([0.0, 5.0, 0.0, 0.0], dtype=complex)
        v_ant = np.ones(12, dtype=complex)
        v_ant[9] = 0.5  # sag where the bunch sits: global column 9
        feedback = self._feedback_stub(
            rf_centers_lengths=(8, 4), i_beam_forward=i_beam, v_ant=v_ant
        )
        observation = self._observation_for(feedback)
        observation._update()
        i_row = observation.i_beam_coarse[0]
        v_row = observation.v_ant_coarse[0]
        # Columns outside the forward segment [8, 12) are masked.
        self.assertTrue(np.all(np.isnan(i_row[:8])))
        # The bunch cell lands in the same column in both matrices.
        self.assertEqual(int(np.nanargmax(np.abs(i_row))), 9)
        self.assertEqual(int(np.nanargmin(np.abs(v_row))), 9)


class TestIQCavityFeedbackObservationTracked(unittest.TestCase):
    """Coarse-recorder column alignment in a real tracked simulation."""

    def test_beam_and_voltage_columns_align_in_multi_section_run(self):
        """The recorded bunch column matches the voltage-sag column.

        Drives the real ``IQCavityFeedbackObservation._update`` path: a
        matched bunch with strong beam loading is tracked through a
        two-section ring with PI-regulated feedbacks, observing station 0
        (whose per-turn grid starts with a reverse segment, so the
        forward offset is non-zero).
        """
        from blond import ConstantMagneticCycle, mu_plus
        from blond.cycles.magnetic_cycle import (
            MagneticCyclePerTurnAllRFStations,
        )
        from blond.physics.feedbacks.cavity_feedback import (
            IQCavityFeedbackTimingClass,
        )
        from blond.physics.feedbacks.generator_current_controller import (
            GeneratorCurrentPIController,
        )

        n_sections = 2
        n_turns = 4
        energy = 4.0e9
        delta_e_turn = 20.0e6
        harmonic = 512
        circumference = 5990.0
        alpha_p = 10.395e-4
        r_over_q = 518.0
        q_l = 1.29e6
        v_design = 30.0e6
        intensity = 5.0e13
        i_gen_bias = v_design / (2.0 * r_over_q * q_l)
        gain_p = 0.1 / (r_over_q * 2.0 * np.pi)

        cycle_probe = ConstantMagneticCycle(
            reference_particle=mu_plus,
            value=energy,
            in_unit="total energy",
        )
        t_rev = cycle_probe.get_t_rev_init(
            circumference, particle_type=mu_plus
        )
        t_rf = t_rev / harmonic

        ring = Ring(circumference=circumference, check_section_indices=False)
        half_drift = circumference / n_sections / 2
        feedbacks = []
        elements = []
        for section_index in range(n_sections):
            profile = StaticProfile.from_rad(
                np.pi * 1.5,
                np.pi * 4.5,
                128,
                t_rf,
                section_index=section_index,
            )
            controller = GeneratorCurrentPIController(
                gain_proportional=gain_p,
                gain_integral=gain_p / (30.0 * t_rf),
                generator_current_bias=i_gen_bias + 0.0j,
                n_delay=2,
            )
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=r_over_q,
                Q_L=q_l,
                generator_current_bias=i_gen_bias + 0.0j,
                n_cavities=1,
                initial_voltage=v_design,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
                controller=controller,
                voltage_setpoint=v_design + 0.0j,
            )
            station = SingleHarmonicRFStation(
                voltage=v_design,
                phi_rf=0.0,
                harmonic=harmonic,
                cavity_feedback=feedback,
                profile=profile,
                section_index=section_index,
            )
            feedbacks.append(feedback)
            elements += [
                DriftSimple(
                    orbit_length=half_drift,
                    momentum_compaction_factor=alpha_p,
                    section_index=section_index,
                ),
                station,
                DriftSimple(
                    orbit_length=half_drift,
                    momentum_compaction_factor=alpha_p,
                    section_index=section_index,
                ),
            ]
        ring.add_elements(elements, reorder=False)

        delta_e_section = delta_e_turn / n_sections
        values = (
            energy + delta_e_section * np.arange(1, n_sections * n_turns + 1)
        ).reshape(n_sections, n_turns, order="F")
        cycle = MagneticCyclePerTurnAllRFStations(
            reference_particle=mu_plus,
            value_init=energy,
            values_after_rf_station_per_turn=values,
            in_unit="total energy",
        )
        sim = Simulation(ring=ring, magnetic_cycle=cycle)

        tracked_beam = Beam(intensity=intensity, particle_type=mu_plus)
        tracked_beam.reference.total_energy = energy
        sim.prepare_beam(
            beam=tracked_beam,
            preparation_routine=BiGaussian(
                n_macroparticles=3000,
                sigma_dt=0.06 * t_rf,
                sigma_dE=None,
                seed=7,
                reinsertion=True,
            ),
        )
        # Shift the bunch one RF period into the profile window (the
        # window starts at 0.75 t_rf; the matched bunch sits around 0).
        tracked_beam._dt.array_local += t_rf

        observation = IQCavityFeedbackObservation(
            each_turn_i=1, feedback=feedbacks[0]
        )
        sim.run_simulation(
            (tracked_beam,),
            n_turns=n_turns,
            observe=(observation,),
            show_progressbar=False,
        )

        i_row = observation.i_beam_coarse[-1]
        v_row = observation.v_ant_coarse[-1]
        valid_i = np.flatnonzero(~np.isnan(i_row))
        valid_v = np.flatnonzero(~np.isnan(v_row))
        n_grid = len(valid_v)

        # The forward segment is the LAST rf_centers_lengths[-1] cells of
        # the per-turn grid: the beam-current columns must be a contiguous
        # block ending at the last grid column, NOT starting at column 0.
        self.assertTrue(np.all(np.diff(valid_i) == 1))
        self.assertEqual(int(valid_i[-1]), n_grid - 1)
        self.assertGreater(int(valid_i[0]), 0)

        # F2 argmax relation: the bunch's beam-current column and the
        # beam-loading sag of the antenna voltage name the same cell.
        beam_col = int(np.nanargmax(np.abs(i_row)))
        sag_col = int(np.nanargmin(np.abs(v_row)))
        self.assertLessEqual(abs(beam_col - sag_col), 2)


class TestDriftObservation(unittest.TestCase):
    def setUp(self):
        drift = Mock(DriftSimple)
        drift._last_eta_0 = 222
        self.obs = DriftObservation(each_turn_i=2, drift=drift)

    def test___init__(self):
        pass  # done by setup

    def test_on_run(self):
        self.obs.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=100,
        )

    def test_update(self):
        self.obs.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=10,
        )
        self.obs._update()
        self.obs._update()
        self.assertEqual(self.obs.eta_0s[0], 222)
        self.assertEqual(len(self.obs.eta_0s), 2)  # two updates before


if __name__ == "__main__":
    unittest.main()
