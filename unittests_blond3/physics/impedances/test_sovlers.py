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
)
from blond3.physics.profiles import StaticProfile

hist_y = np.array(
    [
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        2,
        2,
        1,
        2,
        4,
        1,
        0,
        3,
        3,
        6,
        5,
        8,
        7,
        8,
        10,
        10,
        13,
        13,
        13,
        11,
        13,
        20,
        18,
        24,
        24,
        31,
        32,
        33,
        46,
        34,
        54,
        60,
        54,
        59,
        81,
        70,
        85,
        102,
        88,
        105,
        123,
        116,
        142,
        140,
        148,
        163,
        165,
        187,
        190,
        164,
        211,
        187,
        186,
        190,
        160,
        188,
        263,
        230,
        231,
        196,
        212,
        246,
        213,
        230,
        204,
        224,
        187,
        238,
        180,
        214,
        191,
        215,
        175,
        174,
        178,
        177,
        146,
        145,
        150,
        138,
        127,
        139,
        100,
        87,
        102,
        83,
        84,
        80,
        62,
        51,
        46,
        55,
        41,
        48,
        39,
        37,
        37,
        22,
        22,
        18,
        22,
        12,
        16,
        13,
        10,
        12,
        8,
        6,
        9,
        7,
        10,
        4,
        0,
        2,
        1,
        0,
        4,
        4,
        0,
        2,
        1,
        1,
    ]
)
beam = Mock(BeamBaseClass)
beam.n_particles = 1e12
beam.n_macroparticles_partial.return_value = 128
beam.particle_type.charge = 1

beam.reference_velocity = 123


class TestInductiveImpedanceSolver(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance_solver = InductiveImpedanceSolver()

        self.inductive_impedance_solver._beam = beam
        self.inductive_impedance_solver._Z_over_n = 12
        _parent_wakefield = Mock(WakeField)
        static_profile = StaticProfile(-5, 5, n_bins=128)
        static_profile._hist_y = hist_y
        _parent_wakefield.profile = static_profile
        self.inductive_impedance_solver._parent_wakefield = _parent_wakefield
        simulation = Mock(Simulation)
        simulation.ring.circumference = 123
        self.inductive_impedance_solver._simulation = simulation

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_calc_induced_voltage(self):
        vals = self.inductive_impedance_solver.calc_induced_voltage(
            self.inductive_impedance_solver._beam
        )
        vals /= vals.max()
        vals_correct = -np.gradient(hist_y)
        vals_correct /= vals_correct.max()
        np.testing.assert_allclose(vals_correct[1:-1], vals[1:-1])

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
