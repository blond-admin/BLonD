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
        self.cut_left, self.cut_right, self.bin_size, self.hist_x = -1e-9, 1e-9, 1e-10, np.arange(-1e-9, 1e-9 + 1e-10, 1e-10)

        self.analytical_single_turn_solver._parent_wakefield = Mock(WakeField)
        self.analytical_single_turn_solver._parent_wakefield.profile.cut_left = self.cut_left
        self.analytical_single_turn_solver._parent_wakefield.profile.cut_right = self.cut_right
        self.analytical_single_turn_solver._parent_wakefield.profile.bin_size = self.bin_size
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = self.hist_x

        profile = np.zeros_like(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = profile

        self.analytical_single_turn_solver._parent_wakefield.sources = (self.resonators,)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__update_potential_sources_profile_changes_array_lengths(self):  # TODO: in principle, this is a test for the dynamic profile, currently not implemented
        """
        ensure that the profile does not change on application of different profile lengths with 0-padding
        """
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e9)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial = int(1e3)
        self.analytical_single_turn_solver._update_potential_sources()
        initial = self.analytical_single_turn_solver._wake_pot_vals
        initial_wake_pot_time = self.analytical_single_turn_solver._wake_pot_time
        assert len(initial) == len(initial_wake_pot_time)
        initial_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        profile_len = len(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        assert profile_len == len(initial_voltage)

        # extend profile with 0s towards the back, should not change the values, which are before the 0s
        new_cut_right = 2.0e-9
        self.analytical_single_turn_solver._parent_wakefield.profile.cut_right = new_cut_right
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_y = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_y, np.zeros(10))
        self.analytical_single_turn_solver._parent_wakefield.profile.hist_x = np.append(
            self.analytical_single_turn_solver._parent_wakefield.profile.hist_x,
            np.arange(self.cut_right + self.bin_size,
                      new_cut_right + self.bin_size,
                      self.bin_size))

        self.analytical_single_turn_solver._wake_pot_vals_needs_update = True
        self.analytical_single_turn_solver._update_potential_sources()
        updated_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)
        # check for correct length of profiles and voltages
        profile_len = len(self.analytical_single_turn_solver._parent_wakefield.profile.hist_x)
        assert profile_len == len(updated_voltage)
        assert len(initial_wake_pot_time) != len(self.analytical_single_turn_solver._wake_pot_time)

        # check for unchanging of voltage, which should not change
        index_offset = np.abs(self.analytical_single_turn_solver._wake_pot_vals[:len(initial)]).argmax() - np.abs(
            initial).argmax()
        assert (self.analytical_single_turn_solver._wake_pot_vals[
                index_offset:len(initial) + index_offset] == initial).all()
        assert (updated_voltage[index_offset:len(initial_voltage) + index_offset] == initial_voltage).all()

    def test__update_potential_sources_result_values(self):
        beam = Mock(BeamBaseClass)
        beam.n_particles = int(1e2)
        beam.particle_type.charge = 1
        beam.n_macroparticles_partial = int(1e2)
        self.analytical_single_turn_solver._update_potential_sources()
        profile_width = int((self.cut_right - self.cut_left) / self.bin_size)
        self.analytical_single_turn_solver._wake_pot_vals = np.zeros(profile_width * 2 + 1)
        self.analytical_single_turn_solver._wake_pot_vals[profile_width - 1:profile_width + 2] = 1 / 3 / e
        calced_voltage = self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)

        min_voltage = np.min(calced_voltage)  # negative due to positive charge
        assert np.isclose(min_voltage, -1/3)
        assert np.isclose(np.abs(calced_voltage - min_voltage).argmin(), profile_width // 2)
        assert np.sum(calced_voltage[0:profile_width //2 - 3]) == 0
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

        bunch_time = np.linspace(-sigma_z * 8.54 / c, 8.54 * sigma_z / c, 2**12)
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
        pot_axis = cst_result["pot_axis"] * 1e12 # pC
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
        assert (self.analytical_single_turn_solver.calc_induced_voltage(beam=beam)[first_nonzero_index:] != initial[first_nonzero_index:]).all()


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
            parent_wakefield.sources = (resonators, )
            self.analytical_single_turn_solver.on_wakefield_init_simulation(
                simulation=simulation, parent_wakefield=parent_wakefield
            )

        # resonators.get_wake.return_value = np.array([1 / 3, 1 / 3, 1 / 3])


class TestMultiPassResonatorSolver(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 1.5e9]),
            quality_factors=np.array([10e3, 10e3, 10e3]),
        )
        self.multi_pass_resonator_solver = MultiPassResonatorSolver()
        self.cut_left, self.cut_right, self.bin_size, self.hist_x = -1e-9, 1e-9, 1e-10, np.arange(-1e-9, 1e-9 + 1e-10, 1e-10)

        self.multi_pass_resonator_solver._parent_wakefield = Mock(WakeField)
        self.multi_pass_resonator_solver._parent_wakefield.profile.cut_left = self.cut_left
        self.multi_pass_resonator_solver._parent_wakefield.profile.cut_right = self.cut_right
        self.multi_pass_resonator_solver._parent_wakefield.profile.bin_size = self.bin_size
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x = self.hist_x

        profile = np.zeros_like(self.multi_pass_resonator_solver._parent_wakefield.profile.hist_x)
        profile[9:12] = 1  # symmetric profile around centerpoint
        profile /= np.sum(profile)
        self.multi_pass_resonator_solver._parent_wakefield.profile.hist_y = profile

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
                            -np.log(local_solv._decay_fraction_threshold) / single_resonator._alpha[1])  # mixing of signals

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