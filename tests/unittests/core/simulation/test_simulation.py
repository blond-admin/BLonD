from __future__ import annotations

import unittest
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import Mock, create_autospec

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    Beam,
    Cupy64Bit,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    backend,
    momentum_compaction_factor,
    mu_minus,
    mu_plus,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, NumpyBackend
from blond.core.beam.base import BeamBaseClass
from blond.core.ring.beam_physics_relevant_elements import (
    BeamPhysicsRelevantElements,
)
from blond.core.simulation.execution_models.conterrotating_beams import (
    MainloopCounterRotatingBeams,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.warnings_ import PerformanceWarning
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BeamObservationOncePerTurn,
    ObservablesOncePerTurnBase,
)
from blond.handle_results.observables_as_elements import (
    BunchObservationMetaParams,
)
from blond.testing.mocks import beam_mock

if TYPE_CHECKING:  # pragma: no cover
    pass  # type: ignore


class TestSimulation(unittest.TestCase):
    def setUp(self):
        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicRFStation()

        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf_design = 0

        N_TURNS = int(1e3)
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 460e9, N_TURNS),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        beam1 = Beam(intensity=1e9, particle_type=proton)
        beam1.setup_beam(
            dt=np.linspace(1, 10, 10),
            dE=np.linspace(11, 20, 10),
            reference_time=0,
            reference_total_energy=450e9,
        )
        self.simulation = Simulation.from_locals(locals())
        self.simulation._beams = (beam1,)
        self.beam = beam1

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__exec_on_init_simulation(self):
        self.simulation._exec_on_init_simulation()

    def test__exec_on_run_simulation(self):
        self.simulation._exec_on_run_simulation(
            n_turns=10,
            beam=self.beam,
        )

    def test_error_throwing(self):
        with self.assertRaises(NotImplementedError):
            self.simulation.run_simulation(
                beams=(self.beam, self.beam, self.beam)
            )

    def test_running_until_section_index(self):
        # self.simulation.finalize((self.beam,))
        with self.assertWarnsRegex(Warning, "n_turns is ignored since "):
            self.simulation.run_simulation(
                beams=(self.beam,),
                n_turns=2,
                until_section_index=0,
            )

        CR_beam = deepcopy(self.beam)
        CR_beam._is_counter_rotating = True
        self.simulation.execution_model = MainloopCounterRotatingBeams()
        with self.assertWarnsRegex(Warning, "n_turns is ignored since "):
            self.simulation.run_simulation(
                beams=(self.beam, CR_beam),
                n_turns=2,
                until_section_index=0,
            )

    def test__run_simulation_counterrotating_beam_no_int_effects(self):
        beam = Beam(
            intensity=1e9, particle_type=mu_plus, is_counter_rotating=False
        )
        beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-10e9, 10e8, 100),
            reference_time=0,
            reference_total_energy=63e9,
        )
        beam_CR = Beam(
            intensity=1e9, particle_type=mu_minus, is_counter_rotating=True
        )
        beam_CR.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-10e9, 10e8, 100),
            reference_time=0,
            reference_total_energy=63e9,
        )
        n_cavities = 2

        circumference = 5990
        ring = Ring(circumference=circumference)

        en_gain_per_turn = 2e9
        n_turns = 3
        total_voltage = en_gain_per_turn / np.sin(135 * np.pi / 180)
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=63e9,
            values_after_turn=np.linspace(
                63e9, 63e9 + en_gain_per_turn * n_turns, n_turns
            ),
            in_unit="kinetic energy",
            reference_particle=mu_plus,
        )
        harmonic = 25900
        momentum_compaction_factor_ = 11.4e-4
        bunch_observation = BunchObservationMetaParams(
            each_turn_i=1, beam=beam
        )
        bunch_observation_CR = BunchObservationMetaParams(
            each_turn_i=1, beam=beam_CR
        )

        one_turn_model = []
        for cavity_i in range(n_cavities):
            one_turn_model.extend(
                [
                    DriftSimple(  # for symmetry's sake for the CR bunch, we need to inject in the middle of a drift
                        momentum_compaction_factor=momentum_compaction_factor_,
                        orbit_length=circumference / n_cavities / 2,
                        section_index=cavity_i,
                    ),
                    bunch_observation_CR,
                    SingleHarmonicRFStation(
                        voltage=total_voltage / n_cavities,
                        phi_rf=0,
                        harmonic=harmonic,
                        section_index=cavity_i,
                    ),
                    bunch_observation,
                    DriftSimple(
                        momentum_compaction_factor=momentum_compaction_factor_,
                        orbit_length=circumference / n_cavities / 2,
                        section_index=cavity_i,
                    ),
                ]
            )
        ring.add_elements(one_turn_model, reorder=False)
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

        sim.run_simulation(
            beams=(beam, beam_CR),
            n_turns=n_turns,
        )
        assert len(bunch_observation.mean_dE) == n_turns * n_cavities
        assert len(bunch_observation.mean_dt) == n_turns * n_cavities
        assert len(bunch_observation.sigma_dE) == n_turns * n_cavities
        assert len(bunch_observation.sigma_dt) == n_turns * n_cavities
        assert len(bunch_observation.rms_emittance) == n_turns * n_cavities
        assert len(bunch_observation_CR.mean_dE) == n_turns * n_cavities
        assert len(bunch_observation_CR.mean_dt) == n_turns * n_cavities
        assert len(bunch_observation_CR.sigma_dE) == n_turns * n_cavities
        assert len(bunch_observation_CR.sigma_dt) == n_turns * n_cavities
        assert len(bunch_observation_CR.rms_emittance) == n_turns * n_cavities
        for member in ["mean_dE", "mean_dt", "sigma_dE", "sigma_dt"]:
            assert np.allclose(
                getattr(bunch_observation, member),
                getattr(bunch_observation_CR, member),
            )

    def test__run_simulation_single_beam(self):
        observe = Mock(spec=ObservablesOncePerTurnBase)

        def my_callback(simulation: Simulation, beam: Beam) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=True)
        self.simulation.turn_counter.value = 0
        self.simulation.finalize(
            beams=self.beam,
            n_turns=10,
            observe=(observe,),
        )
        self.simulation.mainloop(
            beams=self.beam,
            n_turns=10,
            observe=(observe,),
            show_progressbar=True,
            callbacks=mock_func,
        )
        observe.update.assert_called()
        mock_func.assert_called()

    def test__run_simulation_single_beam_many_callbacks(self):
        observe = Mock(spec=ObservablesOncePerTurnBase)

        def my_callback1(simulation: Simulation, beam: Beam) -> None:
            return

        def my_callback2(simulation: Simulation, beam: Beam) -> None:
            return

        mock_func1 = create_autospec(my_callback1, return_value=True)
        mock_func2 = create_autospec(my_callback2, return_value=True)
        self.simulation.turn_counter.value = 0
        self.simulation.finalize(
            beams=self.beam, n_turns=10, observe=(observe,)
        )
        self.simulation.mainloop(
            beams=self.beam,
            n_turns=10,
            observe=(observe,),
            show_progressbar=True,
            callbacks=(mock_func1, mock_func2),
        )
        observe.update.assert_called()
        mock_func1.assert_called()
        mock_func2.assert_called()

    def test_magnetic_cycle(self):
        self.assertNotEqual(None, self.simulation.magnetic_cycle)

    def test_from_locals(self):
        from blond.testing.mocks import (  # NOQA required for locals()
            cycle_const_mock,
            drift_simple_mock,
            single_harmonic_rf_station_mock,
            static_profile_mock,
            wakefield_profile_mock,
        )

        assert not hasattr(drift_simple_mock, "skip_find_instances_attributes")
        drift_simple_mock.section_index = 0
        drift_simple_mock.info_string.return_value = "drift_simple_mock"
        static_profile_mock.section_index = 0
        static_profile_mock.info_string.return_value = "static_profile_mock"
        wakefield_profile_mock.section_index = 0
        wakefield_profile_mock.info_string.return_value = (
            "wakefield_profile_mock"
        )
        single_harmonic_rf_station_mock.section_index = 0
        single_harmonic_rf_station_mock.info_string.return_value = (
            "single_harmonic_rf_station_mock"
        )
        ring = Ring(circumference=12)
        self.simulation.from_locals(locals=locals(), verbose=True)

    def test_get_potential_well_empiric(self):
        from blond.testing.simulation import SimulationTwoRFStations

        sim = SimulationTwoRFStations()
        ts = np.linspace(-2e-9, 2e-9, 100)

        potential_well, factor, tilt_dt_per_dE = (
            sim.simulation.get_potential_well_empiric(
                dt=ts,
                particle_type=proton,
                subtract_min=False,  # for tescase and repeated execution
            )
        )

        potential_well, factor, tilt_dt_per_dE = (
            sim.simulation.get_potential_well_empiric(
                dt=ts,
                particle_type=proton,
            )
        )
        potential_well += 4 * potential_well.mean()
        from blond import backend

        SAVE_PINNED = False

        if backend.float == np.float32:
            raise TypeError("32 Bit backends have been removed")
        elif backend.float == np.float64:
            bits = "64"
        else:
            raise Exception()

        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    f"resources/potential_well_{bits}.csv", stacklevel=1
                ),
                potential_well,
            )
        potential_well_pinned = np.loadtxt(
            callers_relative_path(
                f"resources/potential_well_{bits}.csv", stacklevel=1
            )
        )

        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(potential_well_pinned, label="potential_well_pinned")
            plt.plot(potential_well, "--", label="potential_well")
            plt.subplot(2, 1, 2)
            plt.plot(potential_well - potential_well_pinned)
            plt.legend()
            plt.show()

        from blond import backend

        if backend.float != np.float64:
            raise TypeError("32 Bit backends have been removed")

        np.testing.assert_allclose(
            potential_well_pinned,
            copy_to_cpu(potential_well),
            1e-12,
        )

    def test_plot_potential_well_empiric(self):
        self.simulation.plot_potential_well_empiric(
            dt=backend.linspace(0, 1e-9),
            particle_type=proton,
        )

    def test_load_results(self):
        observation = BeamObservationOncePerTurn(
            each_turn_i=10,
        )
        kwargs = dict(
            beams=(self.beam,),
            n_turns=10,
            observe=(observation,),
        )
        self.simulation.run_simulation(**kwargs)
        de_before_save = observation.dEs.copy()
        self.simulation.save_results(
            observe=(observation,),
            common_name="newname",
        )
        self.simulation.load_results(
            **kwargs,
            common_name="newname",
        )
        de_from_disk = observation.dEs.copy()
        np.testing.assert_almost_equal(de_before_save, de_from_disk)

        for name, rec in observation.get_recorders():
            rec.purge_from_disk()

    def test_on_init_simulation(self):
        self.simulation.on_init_simulation(simulation=self.simulation)

    @unittest.skip
    def test_prepare_beam(self):
        # TODO: implement test for `prepare_beam`
        beam = Mock(spec=BeamBaseClass)

        self.simulation.prepare_beam(
            preparation_routine=None, turn_i=None, beam=beam
        )

    def test_on_run_simulation(self):
        beam = Mock(spec=BeamBaseClass)

        self.simulation.on_run_simulation(
            simulation=self.simulation, n_turns=10, beam=beam
        )

    def test_print_one_turn_execution_order(self):
        self.simulation.print_one_turn_execution_order()

    def test_profiling(self):
        self.simulation.profiling(
            start_turn_i=10,
            n_turns=20,
            beams=(self.beam,),
            stats_lines=20,
        )
        self.simulation.profiling(
            start_turn_i=10,
            n_turns=20,
            beams=(self.beam,),
            stats_lines=None,
        )

    def test_profiling2(self):
        self.simulation.profiling(
            start_turn_i=10,
            n_turns=20,
            beams=(self.beam,),
            stats_lines=None,
        )

    def test_ring(self):
        self.assertIsInstance(self.simulation.ring, Ring)

    def test_run_simulation(self):
        observe = BeamObservationOncePerTurn(each_turn_i=10)

        def my_callback(simulation: Simulation, beam: BeamBaseClass) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=False)

        self.simulation.run_simulation(
            n_turns=10,
            observe=(observe,),
            show_progressbar=True,
            callbacks=mock_func,
            beams=(self.beam,),
        )
        mock_func.assert_called()

    def test_get_t_rev_init_returns_a_plain_float(self):
        """
        The initial revolution period is a scalar float, never an array.

        Consumers rely on this: ``ContinuousMultiTurnTimeDomainSolver``
        compares the profile window against it directly, and an array
        would make that comparison ambiguous instead of failing loudly.
        The method casts with ``float()`` on its only return path, so
        this pins the contract that lets callers skip a type check.
        """
        t_rev_init = self.simulation.get_t_rev_init()

        self.assertIsInstance(t_rev_init, float)
        self.assertGreater(t_rev_init, 0.0)

    def test_get_potential_well_empiric_shape(self):
        cavity = self.simulation.ring.elements.get_element(
            SingleHarmonicRFStation, recursive=False
        )
        particle_type = proton

        ts = np.linspace(
            0,
            self.simulation.magnetic_cycle.get_t_rev_init(
                circumference=self.simulation.ring.circumference,
                particle_type=particle_type,
            )
            / cavity.harmonic,
            20000,
        )
        phis = ts * cavity.calc_omega_rf_design(
            beam_beta=self.beam.reference.beta,
            ring_circumference=self.simulation.ring.circumference,
        )
        potential_well, factor, tilt_dt_per_dE = (
            self.simulation.get_potential_well_empiric(
                ts, particle_type=particle_type
            )
        )
        DEV_PLOT = False
        phi_s = np.pi

        potential_well_analytic = (
            particle_type.charge
            * cavity.voltage
            / (2 * np.pi)
            * (np.cos(phis) - np.cos(phi_s) + (phis - phi_s) * np.sin(phi_s))
        )
        if DEV_PLOT:
            plt.plot(
                potential_well,
                label="potential_well",
            )

            plt.plot(
                potential_well_analytic,
                "--",
                label="potential_well_analytic",
            )
            plt.legend()
            plt.show()
        np.testing.assert_allclose(
            copy_to_cpu(
                potential_well_analytic / potential_well_analytic.max() + 1
            ),
            copy_to_cpu(potential_well / potential_well.max() + 1),
            rtol=1e-4,
        )

    def test_get_potential_well_empiric_charge(self):
        cavity = self.simulation.ring.elements.get_element(
            SingleHarmonicRFStation, recursive=False
        )
        from blond.core.beam.particle_types import ParticleType, c, e, m_p

        noton = ParticleType(
            mass=m_p * c**2 / e,
            charge=2,
        )
        potential_wells = {proton: None, noton: None}
        ts = np.linspace(
            0,
            self.simulation.magnetic_cycle.get_t_rev_init(
                circumference=self.simulation.ring.circumference,
            )
            / cavity.harmonic,
            20000,
        )
        for particle_type in (proton, noton):
            potential_well, factor, tilt_dt_per_dE = (
                self.simulation.get_potential_well_empiric(
                    ts, particle_type=particle_type
                )
            )
            potential_wells[particle_type] = potential_well
        np.testing.assert_allclose(
            copy_to_cpu(potential_wells[proton]) + 1e6,
            copy_to_cpu(potential_wells[noton]) / 2 + 1e6,
            rtol=1e-5,
        )

    def test_get_potential_well_empiric_shape_acceleration(self):
        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicRFStation()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf_design = 0

        N_TURNS = int((20 * 60) * 11e3)
        energies = np.linspace(450e9, 7e12, N_TURNS)
        step = energies[1] - energies[0]
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=energies[0] - step,
            values_after_turn=energies,
            reference_particle=proton,
            in_unit="total energy",
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        beam1 = Beam(intensity=1e9, particle_type=proton)
        beam1.setup_beam(
            dt=np.linspace(1, 10, 10),
            dE=np.linspace(11, 20, 10),
            reference_time=0,
            reference_total_energy=energies[0] - step,
        )
        simulation = Simulation.from_locals(locals())
        beam = beam1

        cavity = simulation.ring.elements.get_element(
            SingleHarmonicRFStation, recursive=False
        )
        particle_type = proton

        ts = np.linspace(
            0,
            simulation.magnetic_cycle.get_t_rev_init(
                circumference=simulation.ring.circumference,
                particle_type=particle_type,
            )
            / cavity.harmonic,
            20000,
        )
        phis = ts * cavity.calc_omega_rf_design(
            beam_beta=beam.reference.beta,
            ring_circumference=simulation.ring.circumference,
        )
        potential_well, factor, tilt_dt_per_dE = (
            simulation.get_potential_well_empiric(
                ts, particle_type=particle_type
            )
        )
        DEV_PLOT = False
        simulation.turn_counter.value = 0
        phi_s = float(cavity.calc_phi_s_main_harmonic(beam=beam1))

        potential_well_analytic = (
            particle_type.charge
            * cavity.voltage
            / (2 * np.pi)
            * (np.cos(phis) - np.cos(phi_s) + (phis - phi_s) * np.sin(phi_s))
        )
        if DEV_PLOT:
            plt.plot(
                potential_well,
                label="potential_well",
            )

            plt.plot(
                potential_well_analytic,
                "--",
                label="potential_well_analytic",
            )
            plt.legend()
            plt.show()
        np.testing.assert_allclose(
            copy_to_cpu(
                potential_well_analytic / potential_well_analytic.max() + 1
            ),
            copy_to_cpu(potential_well / potential_well.max() + 1),
            rtol=1e-4,
        )

    def test_get_drift_term_empiric(self):
        from blond.testing.simulation import SimulationTwoRFStations

        sim = SimulationTwoRFStations()
        simulation = sim.simulation
        de = backend.linspace(-1e9, 1e9)
        beam = sim.beam1
        beam.reference.total_energy = 450e9
        drift_term = simulation.get_drift_term_empiric(
            dE=de,
            particle_type=proton,
        )
        E0 = beam.reference.total_energy
        beta = beam.reference.beta

        eta = float(simulation.ring.calc_average_eta_0(beam.reference.gamma))
        drift_term_analytic = (
            0.5 * eta / (np.square(beta) * E0) * de**2
        )  # [1/eV]
        DEV_DRAW = False
        if DEV_DRAW:
            plt.figure()

            print(drift_term - drift_term_analytic)
            plt.plot(drift_term)
            plt.plot(drift_term_analytic, "--")
            plt.show()
        np.testing.assert_allclose(
            copy_to_cpu(drift_term_analytic + 1),
            copy_to_cpu(drift_term + 1),
            atol=0.15,
        )

    def test_finalize_raises(self) -> None:
        self.simulation.magnetic_cycle._n_turns_max = None
        with self.assertRaises(ValueError):
            self.simulation.finalize(
                beams=beam_mock,
                n_turns=None,
                observe=(),
            )

    @pytest.mark.backend_mutation
    def test_finalize_warns(self) -> None:
        from blond import backend

        if not isinstance(backend, NumpyBackend):
            self.skipTest("Only on CPU")

        beam_mock.common_array_size = int(1e32)
        special_mode_org = backend.specials_mode
        backend.set_specials(mode="python")
        self.simulation.ring._elements = BeamPhysicsRelevantElements(
            check_section_indices=False
        )
        self.simulation.ring._elements.add_element(
            DriftSimple(
                momentum_compaction_factor=1,
                orbit_length=self.simulation.ring.circumference,
            )
        )
        with self.assertWarns(PerformanceWarning):
            self.simulation.finalize(
                beams=(beam_mock,),
                n_turns=None,
                observe=(),
            )
        backend.set_specials(mode=special_mode_org)

    def test__sanitize_callbacks(self):
        from blond import Simulation
        from blond.testing.mocks import simulation_mock

        def callback(sim, beam):
            return

        cases = (
            None,
            callback,
            (callback, callback),
            [callback, callback],
            [callback for i in range(2)],
        )
        for callbacks in cases:
            Simulation._sanitize_callbacks(simulation_mock, callbacks)

        with self.assertRaisesRegex(TypeError, "Unexpected callback type"):
            Simulation._sanitize_callbacks(simulation_mock, 1.0)
        with self.assertRaisesRegex(TypeError, "Unexpected callback type"):
            Simulation._sanitize_callbacks(
                simulation_mock, (callback for i in range(2))
            )

    @pytest.mark.backend_mutation
    @pytest.mark.cupy
    def test_compare_cpu_gpu(self):
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            # skip test if GPU is not available
            self.skipTest(str(exc))
        DEV_DEBUG = False
        results = []
        for i, backend_type in enumerate((Cupy64Bit, Numpy64Bit)):
            backend.change_backend(backend_type)
            from blond.testing.simulation import (
                SimulationTwoRFStationsWithWake,
            )

            sim = SimulationTwoRFStationsWithWake()

            hist_y_override = np.loadtxt(
                callers_relative_path("hist_y_override.txt", stacklevel=1),
            )
            wakefield = sim.simulation.ring.elements.get_element(WakeField)
            wakefield.profile._hist_y = backend.array(
                hist_y_override, dtype=wakefield.profile._hist_y.dtype
            )
            wakefield.profile.hist_y_to_density_factor = 1e-05
            sim.simulation.intensity_effect_manager.set_profiles(False)
            wakefield.calc_induced_voltage(sim.beam1)
            potential, factor, tilt = (
                sim.simulation.get_potential_well_empiric(
                    dt=np.linspace(0, 3e-9),
                    particle_type=sim.beam1.particle_type,
                    intensity=sim.beam1.intensity,
                )
            )
            if DEV_DEBUG:
                plt.figure("debug+potential")
                plt.plot(copy_to_cpu(potential), ("-", "--")[i])
            results.append(copy_to_cpu(potential))
        if DEV_DEBUG:
            plt.show()

        if backend.float == np.float32:
            raise TypeError("32 Bit backends have been removed")

        np.testing.assert_allclose(
            results[0],
            results[1],
            1e-12,
        )

    def test_current_t_rev(self):
        buffer = np.zeros(2)
        t_rev_effective = np.empty(10)
        t_rev_sim = np.empty(10)
        DEV_PLOT = False

        def callback(sim: Simulation, beam: Beam):
            buffer[0] = buffer[1]
            buffer[1] = beam.reference.time
            i = sim.turn_counter.value
            t_rev_effective[i] = buffer[1] - buffer[0]
            t_rev_sim[i] = sim.current_t_rev
            if DEV_PLOT:
                plt.plot(i, buffer[1] - buffer[0], "o")
                plt.plot(i, sim.current_t_rev, "x")

        self.simulation.run_simulation(
            self.beam,
            n_turns=10,
            callbacks=callback,
        )
        np.testing.assert_allclose(t_rev_effective, t_rev_sim)
        if DEV_PLOT:
            plt.show()

    def test_current_turn_dE_tot(self):
        # set initial value of energy
        buffer = (
            np.ones(2) * self.simulation.magnetic_cycle.get_total_energy_init()
        )
        dE_rev_effective = np.empty(10)
        dE_rev_sim = np.empty(10)
        DEV_PLOT = False

        def callback(sim: Simulation, beam: Beam):
            buffer[0] = buffer[1]
            buffer[1] = beam.reference.total_energy
            i = sim.turn_counter.value
            dE_rev_effective[i] = buffer[1] - buffer[0]
            dE_rev_sim[i] = sim.current_turn_dE_tot
            if DEV_PLOT:
                plt.plot(i, buffer[1] - buffer[0], "o")
                plt.plot(i, sim.current_turn_dE_tot, "x")

        with self.assertRaisesRegex(
            ValueError,
            "only available during the simulation",
        ):
            self.simulation.current_turn_dE_tot
        with self.assertRaisesRegex(
            ValueError,
            "only available during the simulation",
        ):
            self.simulation.current_t_rev
        self.simulation.run_simulation(
            self.beam,
            n_turns=10,
            callbacks=callback,
        )
        np.testing.assert_allclose(dE_rev_sim, dE_rev_effective)
        if DEV_PLOT:
            plt.show()


class TestTwoBeamProfilePlacementCheck(unittest.TestCase):
    """
    Live-profile placement validation of the counter-rotating mainloop.

    A live (``active=True``) profile consumed by an element must be tracked
    as a ring element on both sides of the consumer, so each of the two
    counter-rotating beams re-histograms it immediately before the consumer
    in its own traversal order; frozen profiles are exempt. The checker is
    exercised directly on lightweight stand-in objects.
    """

    @staticmethod
    def _simulation_with(elements):
        """
        Wrap an element list in the minimal simulation stand-in.

        Parameters
        ----------
        elements
            The one-turn ring element list.

        Returns
        -------
        SimpleNamespace
            Object exposing ``._ring.elements.elements`` like a Simulation.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            _ring=SimpleNamespace(elements=SimpleNamespace(elements=elements))
        )

    @staticmethod
    def _profile(active):
        """
        A minimal profile stand-in.

        Parameters
        ----------
        active
            Whether the profile histogram is live.

        Returns
        -------
        SimpleNamespace
            Object exposing ``.active`` like a profile.
        """
        from types import SimpleNamespace

        return SimpleNamespace(active=active)

    @staticmethod
    def _consumer(profile):
        """
        A minimal profile-consuming element stand-in.

        Parameters
        ----------
        profile
            The profile the element consumes.

        Returns
        -------
        SimpleNamespace
            Object exposing ``.profile`` like an RF station / wakefield.
        """
        from types import SimpleNamespace

        return SimpleNamespace(profile=profile)

    @staticmethod
    def _drift():
        """
        A minimal non-consuming, non-histogramming element stand-in.

        Returns
        -------
        SimpleNamespace
            Object with no ``.profile`` (a drift/kick spacer element).
        """
        from types import SimpleNamespace

        return SimpleNamespace()

    def _check(self, elements):
        """
        Run the placement check on an element list.

        Parameters
        ----------
        elements
            The one-turn ring element list.
        """
        from blond.core.simulation.execution_models.conterrotating_beams import (
            MainloopCounterRotatingBeams,
        )

        MainloopCounterRotatingBeams._check_two_beam_profile_placement(
            self._simulation_with(elements)
        )

    def test_minimal_sandwich_rejected(self):
        """
        The minimal ``(profile, consumer, profile)`` sandwich is rejected.

        This previously asserted the sandwich was *accepted*, encoding the
        old (incorrect) belief that placing the profile on both sides of
        the consumer is sufficient. Replaying the counter-rotating
        interleave shows it is not: for ``[profile, consumer, profile]``
        the counter beam re-histograms the profile between the co-rotating
        beam's own write and its read, so the co-rotating beam reads the
        wrong line density. A correct guard rejects it.
        """
        profile = self._profile(active=True)
        with self.assertRaises(ValueError):
            self._check([profile, self._consumer(profile), profile])

    def test_frozen_profile_needs_no_tracking(self):
        """A frozen (``active=False``) profile is exempt from the check."""
        profile = self._profile(active=False)
        self._check([self._consumer(profile)])

    def test_untracked_live_profile_raises(self):
        """A live profile that is never tracked is rejected."""
        profile = self._profile(active=True)
        with self.assertRaises(ValueError) as ctx:
            self._check([self._consumer(profile)])
        message = str(ctx.exception)
        self.assertIn("never tracked", message)
        # The rejection reason must stay accurate: real consumers
        # (wakefield / feedback) self-histogram their profile in place,
        # so the guard must NOT claim it is "never histogrammed". The
        # true reason is that this conservative, element-list-only check
        # cannot verify the shared live profile is read by the right beam.
        self.assertNotIn("never histogrammed", message)
        self.assertIn("conservative guard", message)

    def test_one_sided_live_profile_raises(self):
        """
        A live profile tracked on only one side is rejected.

        ``[profile, consumer]`` histograms the profile with the forward
        beam, but the counter-rotating beam reads it first (in reverse
        order) before re-histogramming it, so it reads the wrong beam.
        """
        profile = self._profile(active=True)
        with self.assertRaises(ValueError) as ctx:
            self._check([profile, self._consumer(profile)])
        self.assertIn("counter-rotating", str(ctx.exception))

    def test_attached_wakefield_profile_is_checked(self):
        """A profile attached via ``_local_wakefield`` is validated too."""
        from types import SimpleNamespace

        profile = self._profile(active=True)
        station = SimpleNamespace(
            profile=None,
            _local_wakefield=SimpleNamespace(profile=profile),
        )
        with self.assertRaises(ValueError):
            self._check([profile, station])
        # verified-safe padded live layout: accepted
        self._check([profile, station, profile, self._drift(), self._drift()])

    def test_attached_feedback_profile_is_checked(self):
        """A profile attached via ``cavity_feedback_list`` is validated too."""
        from types import SimpleNamespace

        profile = self._profile(active=True)
        station = SimpleNamespace(
            profile=None,
            cavity_feedback_list=[SimpleNamespace(profile=profile)],
        )
        with self.assertRaises(ValueError):
            self._check([station, profile])
        # verified-safe padded live layout: accepted
        self._check([profile, station, profile, self._drift(), self._drift()])

    def test_clobbering_layout_is_rejected(self):
        """
        ``[profile, consumer, profile, drift]`` corrupts and is rejected.

        The old one-sided guard green-lit this layout (the profile appears
        both before and after the consumer), yet tracing the mainloop's
        forward-then-counter interleave shows the counter-rotating beam
        reads the co-rotating beam's histogram at the consumer (bug #2).
        """
        profile = self._profile(active=True)
        with self.assertRaises(ValueError) as ctx:
            self._check(
                [profile, self._consumer(profile), profile, self._drift()]
            )
        self.assertIn("counter-rotating", str(ctx.exception))

    def test_interleave_corrupts_counter_beam_at_consumer(self):
        """
        Behavioural ground truth: the counter beam reads the wrong bunch.

        This reproduces the exact mainloop schedule -- forward tracks
        ``elements[k]`` then counter tracks ``elements[N-1-k]`` -- over the
        ``[profile, consumer, profile, drift]`` layout with two distinct
        bunches, and shows the counter-rotating beam reads the co-rotating
        bunch's histogram at the consumer.
        """
        from types import SimpleNamespace

        histogram = {"owner": None}  # which bunch last histogrammed profile
        prof = SimpleNamespace(active=True)
        reads = []

        def track(element, bunch):
            if element is prof:
                histogram["owner"] = bunch  # histogram in place
            elif getattr(element, "consumes", None) is prof:
                reads.append((bunch, histogram["owner"]))

        drift = SimpleNamespace()
        consumer = SimpleNamespace(consumes=prof)
        elements = [prof, consumer, prof, drift]
        n = len(elements)
        # Two turns so the histogram owner reaches steady state.
        for _turn in range(2):
            reads.clear()
            for element_ind in range(n):
                track(elements[element_ind], "forward")
                track(elements[n - 1 - element_ind], "counter")

        by_beam = {beam: owner for (beam, owner) in reads}
        self.assertEqual(by_beam["forward"], "forward")
        self.assertEqual(
            by_beam["counter"],
            "forward",
            "counter beam is corrupted by the forward bunch's histogram",
        )

    def test_shared_profile_instance_collected_once(self):
        """
        One profile object consumed via several paths is collected once.

        An element can reach the *same* profile instance through its own
        ``profile``, its local wakefield, and an attached feedback; the
        collection must deduplicate by identity so the placement check
        validates one shared profile, not three. Two *distinct* profile
        objects must both survive, even if they compare equal
        (``SimpleNamespace`` instances with equal attributes do), so the
        dedup has to be an ``is`` check, not ``in``/``==``.
        """
        from types import SimpleNamespace

        from blond.core.simulation.execution_models.conterrotating_beams import (
            MainloopCounterRotatingBeams,
        )

        profile = self._profile(active=True)
        station = SimpleNamespace(
            profile=profile,
            _local_wakefield=SimpleNamespace(profile=profile),
            cavity_feedback_list=[SimpleNamespace(profile=profile)],
        )
        collected = MainloopCounterRotatingBeams._live_consumed_profiles(
            station
        )
        self.assertEqual(len(collected), 1)
        self.assertIs(collected[0], profile)

        # equal-but-distinct profiles are NOT merged
        other = self._profile(active=True)
        self.assertEqual(profile, other)  # equal ...
        self.assertIsNot(profile, other)  # ... but distinct objects
        station = SimpleNamespace(
            profile=profile,
            _local_wakefield=SimpleNamespace(profile=other),
        )
        collected = MainloopCounterRotatingBeams._live_consumed_profiles(
            station
        )
        self.assertEqual(len(collected), 2)

    def test_padded_safe_live_layout_passes(self):
        """
        A genuinely-safe padded live layout is accepted.

        ``[profile, consumer, profile, drift, drift]`` is one of the
        layouts whose interleave leaves each beam reading its own
        histogram at the consumer (verified by exhaustive schedule
        simulation); the guard must not reject it.
        """
        profile = self._profile(active=True)
        self._check(
            [
                profile,
                self._consumer(profile),
                profile,
                self._drift(),
                self._drift(),
            ]
        )


if __name__ == "__main__":
    unittest.main()
