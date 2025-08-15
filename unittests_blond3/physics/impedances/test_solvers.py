import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt

from blond3 import WakeField, Simulation
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import (
    PeriodicFreqSolver,
    InductiveImpedance,
    InductiveImpedanceSolver,
    AnalyticSingleTurnResonatorSolver,
    MultiPassResonatorSolver,
)
from collections import deque
from copy import deepcopy
from blond3.physics.profiles import StaticProfile, DynamicProfileConstCutoff, DynamicProfileConstNBins
from scipy.constants import e, c
import json


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
        self.periodic_freq_solver._parent_wakefield.sources = (self.resonators,)
        self.periodic_freq_solver._update_internal_data()
        self.assertEqual(self.periodic_freq_solver._n_time, 10)

    def test__update_internal_data2(self):
        self.periodic_freq_solver._parent_wakefield.sources = (self.resonators,)
        self.periodic_freq_solver._parent_wakefield.profile.hist_step = 0.5e-9
        self.periodic_freq_solver.t_periodicity = 1e-8

    def test_calc_induced_voltage(self):
        self.periodic_freq_solver._parent_wakefield.profile.beam_spectrum.return_value = np.linspace(
            0, 1, 11
        )
        self.periodic_freq_solver._parent_wakefield.sources = (self.resonators,)
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
        self.analytical_single_turn_solver = AnalyticSingleTurnResonatorSolver()
        self.left_edge, self.right_edge, self.bin_size = -1e-9, 1e-9, 1e-10
        self.hist_x = np.arange(self.left_edge, self.right_edge + self.bin_size, self.bin_size)

        self.analytical_single_turn_solver._parent_wakefield = Mock(WakeField)
        self.analytical_single_turn_solver._parent_wakefield.profile.bin_size = self.bin_size
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = self.hist_x

        profile = np.zeros_like(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = profile

        self.analytical_single_turn_solver._parent_wakefield.sources = (self.resonators,)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_potential_sources_profile_changes_array_lengths(
            self):  # TODO: in principle, this is a test for the dynamic profile, currently not implemented
        """
        ensure that the profile does not change on application of different profile lengths with 0-padding
        """
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial = int(1e3)
        self.analytical_single_turn_solver._update_potential_sources(zero_pinning=True)
        initial_wake_pot = self.analytical_single_turn_solver._wake_pot_vals
        initial_wake_pot_time = self.analytical_single_turn_solver._wake_pot_time
        assert len(initial_wake_pot) == len(initial_wake_pot_time)
        initial_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        initial_profile_len = len(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        assert initial_profile_len == len(initial_voltage)

        # extend profile with 0s towards the back, should not change the values, which are before the 0s
        new_cut_right = 2.0e-9
        self.analytical_single_turn_solver._parent_wakefield.profile.cut_right = new_cut_right
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_y, np.zeros(10))
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x,
            np.arange(self.right_edge + self.bin_size,
                      new_cut_right + self.bin_size,
                      self.bin_size))

        self.analytical_single_turn_solver._wake_pot_vals_needs_update = True
        self.analytical_single_turn_solver._update_potential_sources(zero_pinning=True)
        updated_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        # check for correct length of profiles and voltages
        profile_len = len(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        assert profile_len == len(updated_voltage)
        assert len(initial_wake_pot_time) != len(self.analytical_single_turn_solver._wake_pot_time)

        # check for unchanging of voltage, which should not change
        index_offset = (profile_len - initial_profile_len) // 2
        assert np.allclose(self.analytical_single_turn_solver._wake_pot_vals[
                index_offset:len(initial_wake_pot) + index_offset], initial_wake_pot)
        assert np.allclose(updated_voltage[index_offset:len(initial_voltage) + index_offset], initial_voltage)

    def test__update_potential_sources_location_of_calculation_matching(self):
        # TODO: check for matching
        pass

    def test__update_potential_sources_result_values(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e2)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial = int(1e2)
        self.analytical_single_turn_solver._update_potential_sources()
        profile_width = int((self.right_edge - self.left_edge) / self.bin_size)
        self.analytical_single_turn_solver._wake_pot_vals = np.zeros(profile_width * 2 + 1)
        self.analytical_single_turn_solver._wake_pot_vals[profile_width - 1:profile_width + 2] = 1 / 3 / e
        calced_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)

        min_voltage = np.min(calced_voltage)  # negative due to positive charge
        assert np.isclose(min_voltage, -1 / 3)
        assert np.isclose(np.abs(calced_voltage - min_voltage).argmin(), profile_width // 2)
        assert np.sum(calced_voltage[0:profile_width // 2 - 3]) == 0
        assert np.sum(calced_voltage[profile_width + 2:]) == 0

    def test_against_CST_results(self):
        # CST settings: open BC at z, magnetic symmetry planes, ec1 parameters from https://cds.cern.ch/record/533324, f_cutoff = 2.5GHz, WF length = 5m
        # create bunch with sigma of 40mm --> set this as profile, convolute with potential to get wake for the first 5 meters
        sigma_z = 40e-3
        # R_over_Q = np.array([51.94, 13.7312, 0.0915, 2.638805, 2.132499, 2.712645, 4.064])
        # q_factor = np.array([4.15e8, 4.416e5, 38791, 70.629, 59.224, 35.6335, 23.2348])
        # freq = np.array([1.30192e9, 2.4508e9, 2.70038e9, 3.0675e9, 3.083e9, 3.34753e9, 3.42894e9])
        with open("resources/TESLA_until_4.5GHz.json", "r", encoding="utf-8") as cst_modes_EM_file:
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

        res = Resonators(quality_factors=q_factor,
                         shunt_impedances=R_shunt,
                         center_frequencies=freq)
        analy = AnalyticSingleTurnResonatorSolver()

        bunch_time = np.linspace(-sigma_z * 8.54 / c, 8.54 * sigma_z / c, 2 ** 12)
        bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

        analy._parent_wakefield = Mock(WakeField)
        analy._parent_wakefield.profile.cut_left = -sigma_z * 8.54 / c
        analy._parent_wakefield.profile.cut_right = 8.54 * sigma_z / c
        analy._parent_wakefield.profile.bin_size = bunch_time[1] - bunch_time[0]
        analy._parent_wakefield.profile.hist_x = bunch_time
        analy._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

        analy._parent_wakefield.sources = (res,)

        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e3)
        beam.particle_type.charge = 1 / e
        beam.n_macroparticles_partial = int(1e3)
        # n_particles == n_macroparticles, integrated bunch is 1 --> all normalized to 1C

        analy._wake_pot_vals_needs_update = True

        calced_voltage = analy.calc_induced_voltage(beam=beam)

        cst_result = np.load("resources/TESLA_ec1_WF_pot.npz")
        time_axis = cst_result["s_axis"] / c
        pot_axis = cst_result["pot_axis"] * 1e12  # pC
        plt.plot(np.interp(bunch_time, time_axis, pot_axis)[:len(calced_voltage)])
        plt.plot(calced_voltage[:len(calced_voltage)])
        plt.show()

        # assert np.allclose(np.interp(bunch_time, time_axis, pot_axis)[len(calced_voltage) // 2:], calced_voltage[len(calced_voltage) // 2:], atol=1e10)

    def test_calc_induced_voltage(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial = int(1e3)
        initial = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        first_nonzero_index = np.abs(initial).argmax() - 1
        beam.n_particles = int(1e4)
        assert (self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)[first_nonzero_index:] != initial[
                                                                                                            first_nonzero_index:]).all()

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
        self.cut_left, self.cut_right, self.bin_size, self.hist_x = -1e-9, 1e-9, 1e-10, np.arange(-1e-9, 1e-9 + 1e-10,
                                                                                                  1e-10)

        self.multi_pass_resonator_solver._parent_wakefield = Mock(WakeField)
        self.multi_pass_resonator_solver._parent_wakefield.profile.cut_left = self.cut_left
        self.multi_pass_resonator_solver._parent_wakefield.profile.cut_right = self.cut_right
        self.multi_pass_resonator_solver._parent_wakefield.profile.bin_size = self.bin_size
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x = self.hist_x

        self.profile = np.zeros_like(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x)
        self.profile[9:12] = 1  # symmetric profile around centerpoint
        self.profile /= np.sum(self.profile)
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y = self.profile

        self.multi_pass_resonator_solver._parent_wakefield.sources = (self.resonators,)

    def test_determine_storage_time_single_res(self):
        profile = Mock(StaticProfile)
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1]),
            center_frequencies=np.array([500e6]),
            quality_factors=np.array([10e3]),
        )
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(simulation=simulation,
                                                parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert np.isclose(local_solv._maximum_storage_time,
                          -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[0])

    def test_determine_storage_time_multi_res(self):
        # Check for mixing with multiple resonators
        simulation = Mock(Simulation)
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 10]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later, but similar amplitude
        local_solv = deepcopy(self.multi_pass_resonator_solver)
        local_solv.on_wakefield_init_simulation(simulation=simulation,
                                                parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(local_solv._maximum_storage_time,
                              -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[0])
        assert not np.isclose(local_solv._maximum_storage_time,
                              -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[
                                  1])  # mixing of signals

        # check if one properly overshadows the other with high R_shunt
        single_resonator = Resonators(
            shunt_impedances=np.array([1, 1e9]),
            center_frequencies=np.array([500e6, 500e6]),
            quality_factors=np.array([10e3, 10e6]),
        )  # 2nd one should be way later
        local_solv._parent_wakefield.sources = (single_resonator,)
        local_solv._determine_storage_time()
        assert not np.isclose(local_solv._maximum_storage_time,
                              -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[0])
        assert np.isclose(local_solv._maximum_storage_time,
                          -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[
                              1])  # no mixing due to 2nd one with way higher shunt impedance

    def test_remove_fully_decayed_wake_profiles(self):
        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])
        self.multi_pass_resonator_solver._wake_pot_time = deque([np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]),
                                                                 np.array([2.1, 2.2,
                                                                           2.3])])  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]), np.array([2.1, 2.2, 2.3])])
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])

        self.multi_pass_resonator_solver._maximum_storage_time = 1.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(indexes_to_check=1)

        assert (len(self.multi_pass_resonator_solver._wake_pot_vals) == len(
            self.multi_pass_resonator_solver._wake_pot_time) == len(
            self.multi_pass_resonator_solver._past_profile_times) == len(
            self.multi_pass_resonator_solver._past_profiles) == 2)
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profile_times[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_vals[1]), 6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_time[1]), 3.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profile_times[1]), 3.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6)

        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])
        self.multi_pass_resonator_solver._wake_pot_time = deque([np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]),
                                                                 np.array([2.1, 2.2,
                                                                           2.3])])  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]), np.array([2.1, 2.2, 2.3])])
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])

        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(indexes_to_check=2)
        assert (len(self.multi_pass_resonator_solver._wake_pot_vals) == len(
            self.multi_pass_resonator_solver._wake_pot_time) == len(
            self.multi_pass_resonator_solver._past_profile_times) == len(
            self.multi_pass_resonator_solver._past_profiles) == 1)
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profile_times[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3)

        self.multi_pass_resonator_solver._wake_pot_vals = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])
        self.multi_pass_resonator_solver._wake_pot_time = deque([np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]),
                                                                 np.array([2.1, 2.2,
                                                                           2.3])])  # technically not correct length but doesnt matter here
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]), np.array([2.1, 2.2, 2.3])])
        self.multi_pass_resonator_solver._past_profiles = deque(
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])])

        self.multi_pass_resonator_solver._maximum_storage_time = 2.0
        self.multi_pass_resonator_solver._remove_fully_decayed_wake_profiles(indexes_to_check=2)
        assert (len(self.multi_pass_resonator_solver._wake_pot_vals) == len(
            self.multi_pass_resonator_solver._wake_pot_time) == len(
            self.multi_pass_resonator_solver._past_profile_times) == len(
            self.multi_pass_resonator_solver._past_profiles) == 2)
        # check correct values in both elements --> to ensure last one got kicked
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_vals[0]), 3)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_time[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profile_times[0]), 0.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profiles[0]), 3)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_vals[1]), 6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._wake_pot_time[1]), 3.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profile_times[1]), 3.6)
        assert np.isclose(np.sum(self.multi_pass_resonator_solver._past_profiles[1]), 6)

    def test_update_past_profile_times_wake_times(self):
        self.multi_pass_resonator_solver._past_profile_times = deque(
            [np.array([0.1, 0.2, 0.3]), np.array([1.1, 1.2, 1.3]), np.array([2.1, 2.2, 2.3])])
        self.multi_pass_resonator_solver._wake_pot_time = deque(
            [np.array([4.1, 4.2, 4.3]), np.array([5.1, 5.2, 5.3]), np.array([6.1, 6.2, 6.3])])
        sum_before_shift_prof = np.sum(self.multi_pass_resonator_solver._past_profile_times)
        sum_before_shift_wake = np.sum(self.multi_pass_resonator_solver._wake_pot_time)
        orig_ref = 1
        self.multi_pass_resonator_solver._last_reference_time = orig_ref
        delta_t = 1
        self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
            current_time=self.multi_pass_resonator_solver._last_reference_time + delta_t)
        assert np.isclose(sum_before_shift_prof + 9, np.sum(self.multi_pass_resonator_solver._past_profile_times))
        assert np.isclose(sum_before_shift_wake + 9, np.sum(self.multi_pass_resonator_solver._wake_pot_time))
        assert self.multi_pass_resonator_solver._last_reference_time == orig_ref + delta_t

        with self.assertRaises(AssertionError):
            self.multi_pass_resonator_solver._update_past_profile_times_wake_times(
                current_time=self.multi_pass_resonator_solver._last_reference_time - delta_t)

    def test__update_past_profile_potentials_new_arr_init(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(simulation=sim,
                                               parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_res._past_profile_times.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x)
        local_res._past_profiles.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y)
        local_res._update_past_profile_potentials(zero_pinning=True)

        assert len(local_res._wake_pot_time) == 1
        assert len(local_res._wake_pot_vals) == 1
        assert len(local_res._past_profile_times) == 1
        assert len(local_res._past_profiles) == 1

        assert len(local_res._wake_pot_vals[0]) == len(local_res._wake_pot_time[0])

        assert np.allclose(local_res._past_profile_times[0], self.hist_x)
        assert np.allclose(local_res._past_profiles[0], self.profile)

    def test__update_past_profile_potentials_pushback_of_2nd_array(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(simulation=sim,
                                               parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_res._past_profile_times.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x)
        local_res._past_profiles.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y)
        local_res._update_past_profile_potentials(zero_pinning=True)

        local_res._past_profile_times.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x + 1)
        local_res._past_profiles.appendleft(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y + 1)
        local_res._update_past_profile_potentials(zero_pinning=True)

        # should have been pushed back --> [1] is the oder profile, [0] is the newest
        assert len(local_res._wake_pot_time) == 2
        assert len(local_res._wake_pot_vals) == 2
        assert len(local_res._past_profile_times) == 2
        assert len(local_res._past_profiles) == 2

        assert len(local_res._wake_pot_vals[0]) == len(local_res._wake_pot_time[0])
        assert len(local_res._wake_pot_vals[1]) == len(local_res._wake_pot_time[1])

        assert np.allclose(local_res._past_profile_times[1], self.hist_x)
        assert np.allclose(local_res._past_profiles[1], self.profile)
        assert np.allclose(local_res._past_profile_times[0], self.hist_x + 1)
        assert np.allclose(local_res._past_profiles[0], self.profile + 1)

        assert np.not_equal(local_res._wake_pot_vals[0], local_res._wake_pot_vals[1]).any()
        assert np.allclose(local_res._wake_pot_time[0], local_res._wake_pot_time[1] + 1)

    def test__update_potential_sources(self):
        """
        test presence of arrays and correct shifting of timing
        """
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(simulation=sim,
                                               parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_res._update_potential_sources()

        local_res._wake_pot_vals_needs_update = True
        tsteps = [0.5, 1.0, 1.6]
        local_res._maximum_storage_time = 1.5
        local_res._update_potential_sources(tsteps[0])

        assert len(local_res._wake_pot_time) == len(local_res._wake_pot_vals) == len(local_res._past_profile_times) == len(local_res._past_profiles) == 2
        assert np.mean(local_res._wake_pot_time[1]) == np.mean(local_res._wake_pot_time[0] + tsteps[0])
        assert np.mean(local_res._past_profile_times[1]) == np.mean(local_res._past_profile_times[0] + tsteps[0])
        assert np.allclose(local_res._past_profiles[0], local_res._past_profiles[1])

        # repeat another time, first array should be kicked out due to delay
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(tsteps[1])
        assert len(local_res._wake_pot_time) == len(local_res._wake_pot_vals) == len(
            local_res._past_profile_times) == len(local_res._past_profiles) == 3
        assert np.mean(local_res._wake_pot_time[1]) == np.mean(local_res._wake_pot_time[0] + tsteps[1] - tsteps[0])
        assert np.mean(local_res._past_profile_times[1]) == np.mean(local_res._past_profile_times[0] + tsteps[1] - tsteps[0])
        assert np.allclose(local_res._past_profiles[0], local_res._past_profiles[1])

        # kick out oldest profile
        local_res._wake_pot_vals_needs_update = True
        local_res._update_potential_sources(tsteps[2])
        assert len(local_res._wake_pot_time) == len(local_res._wake_pot_vals) == len(
            local_res._past_profile_times) == len(local_res._past_profiles) == 3
        assert np.mean(local_res._wake_pot_time[1]) == np.mean(local_res._wake_pot_time[0] + tsteps[2] - tsteps[1])
        assert np.mean(local_res._past_profile_times[1]) == np.mean(local_res._past_profile_times[0] + tsteps[2] - tsteps[1])
        assert np.mean(local_res._wake_pot_time[2]) == np.mean(local_res._wake_pot_time[1] + tsteps[1] - tsteps[0])
        assert np.mean(local_res._past_profile_times[2]) == np.mean(local_res._past_profile_times[1] + tsteps[1] - tsteps[0])

        assert np.allclose(local_res._past_profiles[0], local_res._past_profiles[1])
        assert np.allclose(local_res._past_profiles[1], local_res._past_profiles[2])

    def test__update_potential_sources_bin_size(self):
        sim = Mock(Simulation)

        local_res = deepcopy(self.multi_pass_resonator_solver)
        local_res.on_wakefield_init_simulation(simulation=sim,
                                               parent_wakefield=self.multi_pass_resonator_solver._parent_wakefield)
        local_res._update_potential_sources()

        local_res._maximum_storage_time = 1.5

        local_res._parent_wakefield.profile.hist_x *= 2
        local_res._wake_pot_vals_needs_update = True
        with self.assertRaises(AssertionError, msg="profile bin size needs to be constant"):
            local_res._update_potential_sources(1.0)


    def test_calc_induced_voltage(self):
        # check for array lengths --> array must have same length as profile, check symmetry, asymmetry.
        # check symmetry --> how to do this properly for 2 passes? --> new wakefield simulation with different parameters
        pass

    def compare_to_analytical_resonator_solver_for_results(self):
        # compare to single resonator, if the same results get reached
        pass
