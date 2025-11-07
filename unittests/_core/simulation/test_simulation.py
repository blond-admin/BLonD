import unittest
from copy import deepcopy
from unittest.mock import Mock, create_autospec

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    StaticProfile,
    mu_plus,
    proton,
)
from blond._core.beam.base import BeamBaseClass
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables import (
    BunchObservation,
    BunchObservationMetaParams,
    Observables,
)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        ring = Ring(circumference=26658.883)

        rf_station1 = SingleHarmonicRfStation()
        rf_station1.harmonic = 35640
        rf_station1.voltage = 6e6
        rf_station1.phi_rf = 0

        N_TURNS = int(1e3)
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.transition_gamma = 55.759505

        beam1 = Beam(intensity=1e9, particle_type=proton)
        beam1.setup_beam(
            dt=np.linspace(1, 10, 10),
            dE=np.linspace(11, 20, 10),
            reference_time=0,
            reference_total_energy=1,
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
            turn_i_init=1,
            beam=self.beam,
        )

    def test_error_throwing(self):
        with self.assertRaises(NotImplementedError):
            self.simulation.run_simulation(beams=(self.beam, self.beam, self.beam))

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
        transition_gamma = 1 / np.sqrt(11.4e-4)
        one_turn_model = []
        for rf_station_i in range(n_cavities):
            one_turn_model.extend(
                [
                    DriftSimple(  # for symmetry's sake for the CR bunch, we need to inject in the middle of a drift
                        transition_gamma=transition_gamma,
                        orbit_length=circumference / n_cavities / 2,
                        section_index=rf_station_i,
                    ),
                    SingleHarmonicRfStation(
                        voltage=total_voltage / n_cavities,
                        phi_rf=0,
                        harmonic=harmonic,
                        section_index=rf_station_i,
                    ),
                    DriftSimple(
                        transition_gamma=transition_gamma,
                        orbit_length=circumference / n_cavities / 2,
                        section_index=rf_station_i,
                    ),
                ]
            )
        ring.add_elements(one_turn_model, reorder=False)
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

        bunch_observation = BunchObservationMetaParams(
            each_turn_i=1, obs_per_turn=n_cavities, beam=beam
        )
        bunch_observation_CR = BunchObservationMetaParams(
            each_turn_i=1, obs_per_turn=n_cavities, beam=beam_CR
        )

        sim.run_simulation(
            beams=(beam, beam_CR),
            n_turns=n_turns,
            turn_i_init=0,
            observe=(bunch_observation, bunch_observation_CR),
        )
        for member in ["mean_dE", "mean_dt", "sigma_dE", "sigma_dt"]:
            assert np.allclose(
                getattr(bunch_observation, member),
                getattr(bunch_observation_CR, member),
            )

    def test__run_simulation_single_beam(self):
        observe = Mock(spec=Observables)

        def my_callback(simulation: Simulation, beam: Beam) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=True)
        self.simulation._run_simulation_single_beam(
            beam=self.beam,
            n_turns=10,
            turn_i_init=0,
            observe=(observe,),
            show_progressbar=True,
            callback=mock_func,
        )
        observe.update.assert_called()
        mock_func.assert_called()

    def test_magnetic_cycle(self):
        self.assertNotEqual(None, self.simulation.magnetic_cycle)

    @unittest.skip("Testet in setUp")
    def test_from_locals(self):
        # self.simulation.from_locals(locals=None)
        pass

    @unittest.skip
    def test_get_legacy_map(self):
        # TODO: implement test for `get_legacy_map`
        self.simulation.get_legacy_map()

    @unittest.skip
    def test_get_potential_well(self):
        # TODO: implement test for `get_potential_well`
        self.simulation.get_potential_well()

    @unittest.skip
    def test_get_potential_well_analytic(self):
        # TODO: implement test for `get_potential_well_analytic`
        self.simulation.get_potential_well_analytic()

    def test_get_potential_well_empiric(self):
        from blond.testing.simulation import SimulationTwoRfStations

        sim = SimulationTwoRfStations()
        potential_well = sim.simulation.get_potential_well_empiric(
            ts=np.linspace(-1e-9, 1e-9, 100),
            particle_type=proton,
        )
        SAVE_PINNED = False
        if SAVE_PINNED:
            np.savetxt(
                "resources/potential_well.csv",
                potential_well,
            )
        potential_well_pinned = np.loadtxt(
            callers_relative_path("resources/potential_well.csv", stacklevel=1)
        )
        np.testing.assert_allclose(
            potential_well_pinned,
            potential_well,
        )

    @unittest.skip
    def test_get_separatrix(self):
        # TODO: implement test for `get_separatrix`
        self.simulation.get_separatrix()

    def test_load_results(self):
        observation = BunchObservation(each_turn_i=10, beam=self.beam)
        kwargs = dict(
            beams=(self.beam,),
            n_turns=10,
            turn_i_init=0,
            observe=(observation,),
        )
        self.simulation.run_simulation(**kwargs)
        de_before_save = observation.dEs.copy()
        self.simulation.save_results(observe=(observation,))
        self.simulation.load_results(**kwargs)
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
            simulation=self.simulation, n_turns=10, turn_i_init=0, beam=beam
        )

    def test_print_one_turn_execution_order(self):
        self.simulation.print_one_turn_execution_order()

    def test_profiling(self):
        self.simulation.profiling(
            turn_i_init=0,
            profile_start_turn_i=10,
            profile_n_turns=20,
            beams=(self.beam,),
        )

    def test_ring(self):
        self.assertIsInstance(self.simulation.ring, Ring)

    def test_run_simulation(self):
        observe = BunchObservation(each_turn_i=10, beam=self.beam)

        def my_callback(simulation: Simulation, beam: BeamBaseClass) -> None:
            return

        mock_func = create_autospec(my_callback, return_value=False)

        self.simulation.run_simulation(
            n_turns=10,
            turn_i_init=0,
            observe=(observe,),
            show_progressbar=True,
            callback=mock_func,
            beams=(self.beam,),
        )
        mock_func.assert_called()


if __name__ == "__main__":
    unittest.main()
