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
    Cupy32Bit,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    backend,
    momentum_compaction_factor,
    mu_plus,
    proton,
)
from blond.core.backends.backend import Numpy32Bit, NumpyBackend
from blond.core.beam.base import BeamBaseClass
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
        cavity1.phi_rf = 0

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

    def test__run_simulation_counterrotating_beam_no_int_effects(self):
        beam = Beam(intensity=1e9, particle_type=mu_plus)
        beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-10e9, 10e8, 100),
            reference_time=0,
            reference_total_energy=63e9,
        )
        beam_CR = deepcopy(beam)
        beam_CR._is_counter_rotating = True
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
        self.simulation.turn_i.value = 0
        self.simulation.mainloop_single_beam(
            beam=self.beam,
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
        self.simulation.turn_i.value = 0
        self.simulation.mainloop_single_beam(
            beam=self.beam,
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
            bits = "32"
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

        np.testing.assert_allclose(
            potential_well_pinned,
            copy_to_cpu(potential_well),
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
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
            profile_start_turn_i=10,
            profile_n_turns=20,
            beams=(self.beam,),
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
        phis = ts * cavity.calc_omega(
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
                particle_type=proton,
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
        cavity1.phi_rf = 0

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
        phis = ts * cavity.calc_omega(
            beam_beta=beam.reference.beta,
            ring_circumference=simulation.ring.circumference,
        )
        potential_well, factor, tilt_dt_per_dE = (
            simulation.get_potential_well_empiric(
                ts, particle_type=particle_type
            )
        )
        DEV_PLOT = False
        simulation.turn_i.value = 0
        phi_s = float(cavity.calc_phi_s_single_harmonic(beam=beam1))

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
    def test_compare_cpu_gpu(self):
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        DEV_DEBUG = False
        results = []
        for i, backend_type in enumerate((Cupy32Bit, Numpy32Bit)):
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
        np.testing.assert_allclose(
            results[0],
            results[1],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

    def test_current_t_rev(self):
        buffer = np.zeros(2)
        t_rev_effective = np.empty(10)
        t_rev_sim = np.empty(10)
        DEV_PLOT = False

        def callback(sim: Simulation, beam: Beam):
            buffer[0] = buffer[1]
            buffer[1] = beam.reference.time
            i = sim.turn_i.value
            t_rev_effective[i] = buffer[1] - buffer[0]
            t_rev_sim[i] = sim.current_t_rev
            if DEV_PLOT:
                plt.plot(i, buffer[1] - buffer[0], "o")
                plt.plot(i, sim.current_t_rev, "x")
            return

        self.simulation.run_simulation(
            self.beam,
            n_turns=10,
            callbacks=callback,
        )
        np.testing.assert_allclose(t_rev_effective, t_rev_sim)
        if DEV_PLOT:
            plt.show()


if __name__ == "__main__":
    unittest.main()
