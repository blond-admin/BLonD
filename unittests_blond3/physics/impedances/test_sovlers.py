import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt

from blond3 import WakeField, Simulation
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.sovlers import (
    PeriodicFreqSolver,
    InductiveImpedance,
    InductiveImpedanceSolver,
)
from blond3.physics.profiles import StaticProfile


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
        self.periodic_freq_solver._parent_wakefield.profile.n_bins = 64
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
        parent_wakefield.profile = profile
        parent_wakefield.profile.hist_step = 1
        resonators = Mock(Resonators)
        resonators.is_dynamic = False
        parent_wakefield.sources = (resonators,)
        resonators.get_impedance.return_value = np.linspace(1, 2, 6)

        self.periodic_freq_solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=parent_wakefield
        )
