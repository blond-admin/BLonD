import json
import unittest
from collections import deque
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c, e

from blond import Simulation, WakeField
from blond._core.beam.base import BeamBaseClass
from blond.physics.impedances.solvers import (
    InductiveImpedance,
    InductiveImpedanceSolver,
    MultiPassResonatorSolver,
    PeriodicFreqSolver,
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.physics.profiles import (
    DynamicProfileConstCutoff,
    DynamicProfileConstNBins,
    StaticProfile,
)


class TestInductiveImpedanceSolver(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance_solver = InductiveImpedanceSolver()
        beam = Mock(BeamBaseClass)
        beam.n_particles = 1e12
        beam.n_macroparticles_partial.return_value = 128
        beam.particle_type.charge = 1

        beam.reference_velocity = 123
        self.inductive_impedance_solver._beam = beam
        self.inductive_impedance_solver._Z_over_n = 12
        _parent_wakefield = Mock(WakeField)
        _parent_wakefield.profile.hist_step = 1
        self.inductive_impedance_solver._parent_wakefield = _parent_wakefield
        simulation = Mock(Simulation)
        simulation.ring.circumference = 123
        self.inductive_impedance_solver._simulation = simulation
        _parent_wakefield.profile.diff_hist_y = np.linspace(1, 3)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_calc_induced_voltage(self):
        self.inductive_impedance_solver.calc_induced_voltage(
            self.inductive_impedance_solver._beam
        )  # TODO Pin Physics case here!

    def test_on_wakefield_init_simulation(self):
        simulation = Mock(Simulation)
        simulation.turn_i = 0
        parent_wakefield = Mock(WakeField)
        indcutive_impedance = Mock(InductiveImpedance)
        indcutive_impedance.Z_over_n = 1
        parent_wakefield.sources = (indcutive_impedance,)
        self.inductive_impedance_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )


class TestPeriodicFreqSolver(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance = InductiveImpedance(
            Z_over_n=34.6669349520904 / 10e9 * 11e3
        )
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([400e6, 600e6, 1.2e9]),
            quality_factors=np.array([1, 2, 3]),
        )
        self.periodic_freq_solver = PeriodicFreqSolver(t_periodicity=10)

        self.periodic_freq_solver._parent_wakefield = Mock(WakeField)
        self.periodic_freq_solver._parent_wakefield.profile.beam_spectrum.return_value = np.linspace(
            0, 1, 6
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 1
        self.periodic_freq_solver._parent_wakefield.profile.n_bins = int(8)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_internal_data(self):
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._update_internal_data()
        self.assertEqual(self.periodic_freq_solver._n_time, 10)

    def test__update_internal_data2(self):
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 0.5e-9
        self.periodic_freq_solver.t_periodicity = 1e-8

    def test_calc_induced_voltage(self):
        self.periodic_freq_solver._parent_wakefield.profile.beam_spectrum.return_value = np.linspace(
            0, 1, 11
        )
        self.periodic_freq_solver._parent_wakefield.sources = (
            self.resonators,
        )
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 0.5e-9
        self.periodic_freq_solver._parent_wakefield.profile.n_bins = 20
        self.periodic_freq_solver.t_periodicity = 1e-8
        self.periodic_freq_solver._update_internal_data()
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(11e3)
        beam.n_macroparticles_partial.return_value = int(3e6)
        beam.particle_type.charge = 1
        induced_voltage = self.periodic_freq_solver.calc_induced_voltage(
            beam=beam,
        )  # TODO Pin Physics case here!
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(induced_voltage)
            plt.show()

    def test_on_wakefield_init_simulation(self):
        simulation = Mock(Simulation)
        parent_wakefield = Mock(WakeField)
        profile = Mock(StaticProfile)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1
        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        resonators.get_impedance.return_value = np.linspace(1, 2, 6)

        self.periodic_freq_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )


class TestAnalyticSingleTurnResonatorSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([5, 5, 5]),
        )
        self.analytical_single_turn_solver = (
            SingleTurnResonatorConvolutionSolver()
        )
        self.left_edge, self.right_edge, self.hist_step = -2e-9, 1e-9, 1e-10
        self.hist_x = np.linspace(
            self.left_edge,
            self.right_edge,
            int(np.round((self.right_edge - self.left_edge) / self.hist_step))
            + 1,
            endpoint=True,
        )

        self.analytical_single_turn_solver._parent_wakefield = Mock(WakeField)
        self.analytical_single_turn_solver._parent_wakefield.profile = Mock(
            spec=StaticProfile
        )
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_step = self.hist_step
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = (
            self.hist_x
        )

        profile = np.zeros_like(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
        )
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = (
            profile
        )

        self.analytical_single_turn_solver._parent_wakefield.sources = (
            self.resonators,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_potential_sources_profile_changes_array_lengths(
        self,
    ):  # in principle, this is a test for the dynamic profile, currently not implemented
        """
        ensure that the profile does not change on application of different profile lengths with 0-padding
        """
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e3)
        self.analytical_single_turn_solver._update_potential_sources(
            zero_pinning=True
        )
        initial_wake_pot = self.analytical_single_turn_solver._wake_pot_vals
        initial_wake_pot_time = (
            self.analytical_single_turn_solver._wake_pot_time
        )
        assert len(initial_wake_pot) == len(initial_wake_pot_time)
        initial_voltage = (
            self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        )
        initial_profile_len = len(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
        )
        assert initial_profile_len == len(initial_voltage)

        # extend profile with 0s towards the back, should not change the values, which are before the 0s
        new_right_edge = 2.0e-9
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_y,
            np.zeros(10),
        )
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x,
            np.arange(
                self.right_edge + self.hist_step,
                new_right_edge + self.hist_step,
                self.hist_step,
            ),
        )

        self.analytical_single_turn_solver._wake_pot_vals_needs_update = True
        self.analytical_single_turn_solver._update_potential_sources(
            zero_pinning=True
        )
        updated_voltage = (
            self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        )
        # check for correct length of profiles and voltages
        profile_len = len(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
        )
        assert profile_len == len(updated_voltage)
        assert len(initial_wake_pot_time) != len(
            self.analytical_single_turn_solver._wake_pot_time
        )

        # check for unchanging of voltage, which should not change
        shift_index = int(profile_len - initial_profile_len)
        assert np.allclose(
            self.analytical_single_turn_solver._wake_pot_vals[
                shift_index : shift_index + len(initial_wake_pot)
            ],
            initial_wake_pot,
        )
        assert np.allclose(
            updated_voltage[: len(initial_voltage)],
            initial_voltage,
        )

    def test__update_potential_sources_location_of_calculation_matching(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e3)
        _ = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        first_time = (
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
        )
        found = False
        for run_ind in range(
            len(self.analytical_single_turn_solver._wake_pot_time)
            - len(
                self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
            )
        ):
            if np.allclose(
                self.analytical_single_turn_solver._wake_pot_time[
                    run_ind : run_ind
                    + len(
                        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x
                    )
                ],
                self.analytical_single_turn_solver._parent_wakefield.profile.hist_x,
                atol=self.hist_step / 100,
            ):
                found = True
                break
        assert found

        local_copy = deepcopy(self.analytical_single_turn_solver)
        local_copy._wake_pot_vals_needs_update = True
        local_copy._parent_wakefield.profile.hist_x = (
            local_copy._parent_wakefield.profile.hist_x + 1e-10 / 2
        )
        _ = local_copy.calc_induced_voltage(beam=beam)

        found = False
        for run_ind in range(
            len(local_copy._wake_pot_time)
            - len(local_copy._parent_wakefield.profile.hist_x)
        ):
            if np.allclose(
                local_copy._wake_pot_time[
                    run_ind : run_ind
                    + len(local_copy._parent_wakefield.profile.hist_x)
                ],
                first_time,
                atol=self.hist_step / 100,
            ):
                found = True
                break
        assert found

    def test__update_potential_sources_result_values(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e2)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e2)
        self.analytical_single_turn_solver._update_potential_sources(
            zero_pinning=True
        )
        profile_width = int(
            (self.right_edge - self.left_edge) / self.hist_step
        )
        self.analytical_single_turn_solver._wake_pot_vals = np.zeros(
            profile_width * 2 + 1
        )
        self.analytical_single_turn_solver._wake_pot_vals[
            profile_width - 1 : profile_width + 2
        ] = 1 / 3 / e
        calced_voltage = (
            self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        )

        min_voltage = np.min(calced_voltage)  # negative due to positive charge
        assert np.isclose(min_voltage, -1 / 3)
        assert np.isclose(
            np.abs(calced_voltage - min_voltage).argmin(), profile_width // 3
        )
        assert np.sum(calced_voltage[0 : profile_width // 3 - 3]) == 0
        assert np.sum(calced_voltage[profile_width // 3 + 3 :]) == 0

        # same check, but with self.hist_step/2 shifted histogram, should have same values
        local_res = deepcopy(self.analytical_single_turn_solver)
        local_res._parent_wakefield.profile.hist_x = (
            self.hist_x + self.hist_step / 2
        )

        local_res._update_potential_sources(zero_pinning=True)

        calced_voltage = local_res.calc_induced_voltage(beam=beam)
        min_voltage = np.min(calced_voltage)

        assert np.isclose(min_voltage, -1 / 3)
        assert np.isclose(
            np.abs(calced_voltage - min_voltage).argmin(), profile_width // 3
        )
        assert np.sum(calced_voltage[0 : profile_width // 3 - 3]) == 0
        assert np.sum(calced_voltage[profile_width // 3 + 3 :]) == 0

    def test_against_CST_results(self):
        # CST settings: open BC at z, magnetic symmetry planes, ec1 parameters from https://cds.cern.ch/record/533324, f_cutoff = 2.5GHz, WF length = 5m
        # create bunch with sigma of 40mm --> set this as profile, convolute with potential to get wake for the first 5 meters
        sigma_z = 40e-3
        # R_over_Q = np.array([51.94, 13.7312, 0.0915, 2.638805, 2.132499, 2.712645, 4.064])
        # q_factor = np.array([4.15e8, 4.416e5, 38791, 70.629, 59.224, 35.6335, 23.2348])
        # freq = np.array([1.30192e9, 2.4508e9, 2.70038e9, 3.0675e9, 3.083e9, 3.34753e9, 3.42894e9])
        with open(
            "resources/TESLA_until_4.5GHz.json", "r", encoding="utf-8"
        ) as cst_modes_EM_file:
            cst_modes_dict = json.load(cst_modes_EM_file)
        freq, q_factor, R_over_Q = [], [], []
        for mode in cst_modes_dict:
            if cst_modes_dict[mode]["Qext"] < 200:
                continue
            freq.append(cst_modes_dict[mode]["freq"])
            q_factor.append(cst_modes_dict[mode]["Qext"])
            R_over_Q.append(cst_modes_dict[mode]["R/Q_||"])
        freq = np.array(freq)
        q_factor = np.array(q_factor)
        R_over_Q = np.array(R_over_Q)

        R_shunt = R_over_Q * q_factor

        res = Resonators(
            quality_factors=q_factor,
            shunt_impedances=R_shunt,
            center_frequencies=freq,
        )
        analy = SingleTurnResonatorConvolutionSolver()

        bunch_time = np.linspace(
            -sigma_z * 8.54 / c, 8.54 * sigma_z / c, 2**12
        )
        bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

        analy._parent_wakefield = Mock(WakeField)
        analy._parent_wakefield.profile.hist_step = (
            bunch_time[1] - bunch_time[0]
        )
        analy._parent_wakefield.profile.__
        analy._parent_wakefield.profile.hist_x = bunch_time
        analy._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

        analy._parent_wakefield.sources = (res,)

        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e3)
        beam.particle_type.charge = 1 / e
        beam.n_macroparticles_partial.return_value = int(1e3)
        # n_particles == n_macroparticles, integrated bunch is 1 --> all normalized to 1C

        analy._wake_pot_vals_needs_update = True

        calced_voltage = analy.calc_induced_voltage(beam=beam)

        cst_result = np.load("resources/TESLA_ec1_WF_pot.npz")
        # time_axis = cst_result["s_axis"] / c
        # pot_axis = cst_result["pot_axis"] * 1e12  # pC
        # plt.plot(np.interp(bunch_time, time_axis, pot_axis)[: len(calced_voltage)])
        # plt.plot(calced_voltage[: len(calced_voltage)])
        # plt.show()

        # assert np.allclose(np.interp(bunch_time, time_axis, pot_axis)[len(calced_voltage) // 2:], calced_voltage[len(calced_voltage) // 2:], atol=1e10)

    def test_calc_induced_voltage(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial.return_value = int(1e3)
        initial = self.analytical_single_turn_solver.calc_induced_voltage(
            beam=beam
        )
        first_nonzero_index = np.abs(initial).argmax() - 1
        beam.n_particles = int(1e4)
        assert (
            self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)[
                first_nonzero_index:
            ]
            != initial[first_nonzero_index:]
        ).all()

    def test__on_wakefield_simulation_init(self):
        parent_wakefield = Mock(WakeField)
        profile = Mock(StaticProfile)
        simulation = Mock(Simulation)
        profile.n_bins = 10
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1

        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        self.analytical_single_turn_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )

        with self.assertRaises(RuntimeError):
            profile_wrong = Mock(DynamicProfileConstCutoff)
            parent_wakefield.profile = profile_wrong
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(RuntimeError):
            profile_wrong = Mock(DynamicProfileConstNBins)
            parent_wakefield.profile = profile_wrong
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        parent_wakefield.profile = profile
        with self.assertRaises(RuntimeError):
            wrong_source = Mock(InductiveImpedance)
            wrong_source.is_dynamic = False
            parent_wakefield.sources = (wrong_source, resonators)
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(RuntimeError):
            wrong_source.is_dynamic = True
            parent_wakefield.sources = (wrong_source, resonators)
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )
        with self.assertRaises(ValueError):
            parent_wakefield.profile = None
            parent_wakefield.sources = (resonators,)
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )


class TestMultiPassResonatorSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([10e3, 10e3, 10e3]),
        )
        self.multi_pass_resonator_solver = MultiPassResonatorSolver()
        self.hist_step, self.hist_x = (
            1e-10,
            np.arange(-1e-9, 1e-9 + 1e-10, 1e-10),
        )

        self.multi_pass_resonator_solver._parent_wakefield = Mock(WakeField)
        self.multi_pass_resonator_solver._parent_wakefield.profile = Mock(
            StaticProfile
        )
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_step = self.hist_step
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x = (
            self.hist_x
        )

        self.profile = np.zeros_like(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
        )
        self.profile[9:12] = 1  # symmetric profile around centerpoint
        self.profile /= np.sum(self.profile)
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y = (
            self.profile
        )

        self.multi_pass_resonator_solver._parent_wakefield.sources = (
            self.resonators,
        )

        self.beam = Mock(BeamBaseClass)
        self.beam.n_particles = int(1e2)
        self.beam.particle_type.charge = 1
        self.beam.n_macroparticles_partial.return_value = int(1e2)

    def test_determine_storage_time_single_res(self):
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e3]),
        )
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )

    def test_determine_storage_time_multi_res(self):
        # Check for mixing with multiple resonators
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 10]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later, but similar amplitude
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[1],
        )  # mixing of signals

        # check if one properly overshadows the other with high R_shunt
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 1e9]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[0],
        )
        assert np.isclose(
            local_solv._maximum_storage_time,
            -np.log(local_solv._decay_fraction_threshold)
            / single_resonator._alpha[1],
        )  # no mixing due to 2nd one with way higher shunt impedance

    def test_remove_fully_decayed_wake_profiles(self):
        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_pot_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )

        self.multi_pass_resonator_solver._maximum_storage_time = 1.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=1
        )

        assert (
            len(self.multi_pass_resonator_solver._wake_pot_vals)
            == len(self.multi_pass_resonator_solver._wake_pot_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 2
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_vals[1]), 6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_time[1]), 3.6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6
        )

        # check that we don't crash for the empty array --> only one entry present
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )

        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_pot_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )

        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )
        assert (
            len(self.multi_pass_resonator_solver._wake_pot_vals)
            == len(self.multi_pass_resonator_solver._wake_pot_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 1
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )

        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )
        self.multi_pass_resonator_solver._wake_pot_time = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        )

        self.multi_pass_resonator_solver._maximum_storage_time = 2.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(
            indexes_to_check=2
        )
        assert (
            len(self.multi_pass_resonator_solver._wake_pot_vals)
            == len(self.multi_pass_resonator_solver._wake_pot_time)
            == len(self.multi_pass_resonator_solver._past_profile_times)
            == len(self.multi_pass_resonator_solver._past_profiles)
            == 2
        )
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[0]),
            0.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_vals[1]), 6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._wake_pot_time[1]), 3.6
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profile_times[1]),
            3.6,
        )
        assert np.isclose(
            np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6
        )

    def test_remove_fully_decayed_wake_profiles_physics(self):
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e3]),
        )  # 2nd one should be way later, but similar amplitude
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv.on_wakefield_init_simulation(
            simulation=simulation,
            parent_wakefield=local_solv._parent_wakefield,
        )
        assert np.isclose(
            local_solv._maximum_storage_time,
            1
            / -single_resonator._alpha[0]
            * np.log(local_solv._decay_fraction_threshold),
        )

    def test_update_past_profile_times_wake_times(self):
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [
                np.array([0.1, 0.2, 0.3]),
                np.array([1.1, 1.2, 1.3]),
                np.array([2.1, 2.2, 2.3]),
            ]
        )
        self.multi_pass_resonator_solver._wake_pot_time = deque(
            [
                np.array([4.1, 4.2, 4.3]),
                np.array([5.1, 5.2, 5.3]),
                np.array([6.1, 6.2, 6.3]),
            ]
        )
        sum_before_shift_prof = np.sum(
            self.multi_pass_resonator_solver._past_profile_times
        )
        sum_before_shift_wake = np.sum(
            self.multi_pass_resonator_solver._wake_pot_time
        )
        orig_ref = 1
        self.multi_pass_resonator_solver._last_reference_time = orig_ref
        delta_t = 1
        self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
            current_time=self.multi_pass_resonator_solver._last_reference_time
            + delta_t
        )
        assert np.isclose(
            sum_before_shift_prof + 9,
            np.sum(self.multi_pass_resonator_solver._past_profile_times),
        )
        assert np.isclose(
            sum_before_shift_wake + 9,
            np.sum(self.multi_pass_resonator_solver._wake_pot_time),
        )
        assert (
            self.multi_pass_resonator_solver._last_reference_time
            == orig_ref + delta_t
        )

        with self.assertRaises(AssertionError):
            self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
                current_time=self.multi_pass_resonator_solver._last_reference_time
                - delta_t
            )

    def test__update_past_profile_potentials_new_arr_init(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._past_profile_times.appendleft(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
        )
        local_res._past_profiles.appendleft(
            self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
        )
        local_res._update_past_profile_potentials(zero_pinning=True)

        assert len(local_res._wake_pot_time) == 1
        assert len(local_res._wake_pot_vals) == 1
        assert len(local_res._past_profile_times) == 1
        assert len(local_res._past_profiles) == 1

        assert len(local_res._wake_pot_vals[0]) == len(
            local_res._wake_pot_time[0]
        )

        assert np.allclose(local_res._past_profile_times[0], self.hist_x)
        assert np.allclose(local_res._past_profiles[0], self.profile)

    def test__update_past_profile_potentials_pushback_of_2nd_array(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._past_profile_times.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
            )
        )
        local_res._past_profiles.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
            )
        )
        local_res._update_past_profile_potentials(zero_pinning=True)
        local_res._update_past_profile_times_wake_times(1e-8)
        local_res._past_profile_times.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x
            )
        )
        local_res._past_profiles.appendleft(
            deepcopy(
                self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y
            )
        )
        local_res._update_past_profile_potentials(zero_pinning=True)

        # should have been pushed back --> [1] is the oder profile, [0] is the newest
        assert len(local_res._wake_pot_time) == 2
        assert len(local_res._wake_pot_vals) == 2
        assert len(local_res._past_profile_times) == 2
        assert len(local_res._past_profiles) == 2

        assert len(local_res._wake_pot_vals[0]) == len(
            local_res._wake_pot_time[0]
        )
        assert len(local_res._wake_pot_vals[1]) == len(
            local_res._wake_pot_time[1]
        )

        assert np.allclose(local_res._past_profile_times[0], self.hist_x)
        assert np.allclose(local_res._past_profiles[0], self.profile)
        assert np.allclose(
            local_res._past_profile_times[1], self.hist_x + 1e-8
        )
        assert np.allclose(local_res._past_profiles[1], self.profile + 1e-8)

        assert np.not_equal(
            local_res._wake_pot_vals[0], local_res._wake_pot_vals[1]
        ).any()
        assert np.allclose(
            local_res._wake_pot_time[0],
            local_res._wake_pot_time[1] - 1e-8,
            atol=1e-10,
        )

    def test__update_potential_sources(self):
        """
        test presence of arrays and correct shifting of timing
        """
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._update_potential_sources()

        local_res._wake_pot_vals_needs_update = True
        tsteps = [0.5, 1.0, 1.6]
        local_res._maximum_storage_time = 1.5
        local_res._update_potential_sources(tsteps[0])

        assert (
            len(local_res._wake_pot_time)
            == len(local_res._wake_pot_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 2
        )
        assert np.mean(local_res._wake_pot_time[1]) == np.mean(
            local_res._wake_pot_time[0] + tsteps[0]
        )
        assert np.mean(local_res._past_profile_times[1]) == np.mean(
            local_res._past_profile_times[0] + tsteps[0]
        )
        assert np.allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )

        # repeat another time, first array should be kicked out due to delay
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(tsteps[1])
        assert (
            len(local_res._wake_pot_time)
            == len(local_res._wake_pot_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 3
        )
        assert np.mean(local_res._wake_pot_time[1]) == np.mean(
            local_res._wake_pot_time[0] + tsteps[1] - tsteps[0]
        )
        assert np.mean(local_res._past_profile_times[1]) == np.mean(
            local_res._past_profile_times[0] + tsteps[1] - tsteps[0]
        )
        assert np.allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )

        # kick out oldest profile
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(tsteps[2])
        assert (
            len(local_res._wake_pot_time)
            == len(local_res._wake_pot_vals)
            == len(local_res._past_profile_times)
            == len(local_res._past_profiles)
            == 3
        )
        assert np.isclose(
            np.mean(local_res._wake_pot_time[1]),
            np.mean(local_res._wake_pot_time[0] + tsteps[2] - tsteps[1]),
        )
        assert np.isclose(
            np.mean(local_res._past_profile_times[1]),
            np.mean(local_res._past_profile_times[0] + tsteps[2] - tsteps[1]),
        )
        assert np.isclose(
            np.mean(local_res._wake_pot_time[2]),
            np.mean(local_res._wake_pot_time[1] + tsteps[1] - tsteps[0]),
        )
        assert np.isclose(
            np.mean(local_res._past_profile_times[2]),
            np.mean(local_res._past_profile_times[1] + tsteps[1] - tsteps[0]),
        )

        assert np.allclose(
            local_res._past_profiles[0], local_res._past_profiles[1]
        )
        assert np.allclose(
            local_res._past_profiles[1], local_res._past_profiles[2]
        )

    def test__update_potential_sources_hist_step(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._update_potential_sources()

        local_res._maximum_storage_time = 1.5

        local_res._parent_wakefield.profile.hist_x *= 2
        local_res._wake_pot_vals_needs_update = True
        with self.assertRaises(
            AssertionError,
            msg="profile bin size needs to be constant: hist_step might be too small with casting to delta_t precision",
        ):
            local_res._update_potential_sources(1.0)

    def test_calc_induced_voltage_array_lengths(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._update_potential_sources()
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

        local_res._maximum_storage_time = 1.5
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(1.0)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

    def test_calc_induced_voltage_two_passages(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(
            simulation=sim,
            parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield,
        )
        local_res._update_potential_sources()
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

        local_res._maximum_storage_time = 1.5
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(1.0)

        assert len(ind_volt) == len(local_res._parent_wakefield.profile.hist_x)

    def test_calc_induced_voltage_vals(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e12]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e5]),
        )

        local_res = MultiPassResonatorSolver()

        sigma_z = 40e-3
        sigma_length = 15
        bunch_time = np.linspace(
            -sigma_z * sigma_length / c, sigma_length * sigma_z / c, 2**10
        )
        bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

        local_res._parent_wakefield = Mock(WakeField)
        local_res._parent_wakefield.profile = Mock(spec=StaticProfile)
        local_res._parent_wakefield.profile.hist_step = (
            bunch_time[1] - bunch_time[0]
        )
        local_res._parent_wakefield.profile.hist_x = bunch_time
        local_res._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

        local_res._parent_wakefield.sources = (resonators,)

        sim = Mock(Simulation)

        local_res.on_wakefield_init_simulation(
            simulation=sim, parent_wakefield=local_res._parent_wakefield
        )
        local_res._update_potential_sources()  # does this correctly not throw out the first one?

        ind_volt_init = local_res.calc_induced_voltage(beam=self.beam)

        local_res._wake_pot_vals_needs_update = True
        t_rf = 1 / resonators._center_frequencies[0]
        delay_time = (
            np.floor((1 / resonators._alpha[0]) / t_rf) * t_rf
        )  # multiple of t_r
        local_res._update_potential_sources(
            delay_time
        )  # first one should've fallen to 1/e by this time
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        # ensure perfect addition of in-phase component
        assert not np.allclose(ind_volt, ind_volt_init)
        assert np.argmax(ind_volt) == np.argmax(ind_volt_init)
        assert np.isclose(
            np.min(ind_volt), np.min(ind_volt_init) * (1 + 1 / np.exp(1))
        )

        # assert equality for fully decayed case
        local_res._wake_pot_vals_needs_update = True
        local_res._maximum_storage_time = (
            local_res._maximum_storage_time * 1000
        )
        delay_time = (
            np.floor((1 / resonators._alpha[0]) / t_rf) * t_rf * 100
        )  # multiple of t_r
        local_res._update_potential_sources(
            delay_time
        )  # first one should be 0
        ind_volt = local_res.calc_induced_voltage(beam=self.beam)

        # ensure perfect addition of in-phase component
        assert np.allclose(ind_volt, ind_volt_init)
        assert np.argmax(ind_volt) == np.argmax(ind_volt_init)

    def test_compare_to_analytical_resonator_solver_for_results(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e12, 1e10]),
            center_frequencies=np.array([500e6, 1000e6]),
            quality_factors=np.array([10e5, 10e4]),
        )

        local_res = MultiPassResonatorSolver()

        sigma_z = 40e-3
        sigma_length = 8.54
        for delta_t in [0, 0.5e-8, -0.5e-8]:
            bunch_time = np.linspace(
                -sigma_z * sigma_length / c + delta_t,
                sigma_length * sigma_z / c + delta_t,
                2**10,
            )
            bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

            local_res._parent_wakefield = Mock(WakeField)
            local_res._parent_wakefield.profile = Mock(spec=StaticProfile)
            local_res._parent_wakefield.profile.hist_step = (
                bunch_time[1] - bunch_time[0]
            )
            local_res._parent_wakefield.profile.hist_x = bunch_time
            local_res._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

            local_res._parent_wakefield.sources = (resonators,)
            local_res._wake_pot_vals_needs_update = True

            sim = Mock(Simulation)

            local_res.on_wakefield_init_simulation(
                simulation=sim, parent_wakefield=local_res._parent_wakefield
            )
            local_res._update_potential_sources()

            local_res_analy = SingleTurnResonatorConvolutionSolver()
            local_res_analy._parent_wakefield = Mock(WakeField)
            local_res_analy._parent_wakefield.profile.hist_step = (
                bunch_time[1] - bunch_time[0]
            )
            local_res_analy._parent_wakefield.profile.hist_x = bunch_time
            local_res_analy._parent_wakefield.profile.hist_y = bunch / np.sum(
                bunch
            )
            local_res_analy._parent_wakefield.sources = (resonators,)

            local_res_analy._wake_pot_vals_needs_update = True

            assert np.allclose(
                local_res.calc_induced_voltage(beam=self.beam),
                local_res_analy.calc_induced_voltage(beam=self.beam),
            )
