import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light as c0

from blond3._core.beam.base import BeamBaseClass
from blond3._core.simulation.simulation import Simulation
from blond3.physics.impedances.sources import InductiveImpedance, Resonators


class TestImpedanceTable(unittest.TestCase):
    @unittest.skip
    def test_from_file(self):
        # TODO: implement test for `from_file`
        self.impedance_table.from_file(filepath=None, reader=None)


class TestImpedanceTableFreq(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.impedance_table_freq = ImpedanceTableFreq(freq_x=None, freq_y=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test__get_freq_y(self):
        # TODO: implement test for `_get_freq_y`
        self.impedance_table_freq._get_freq_y()

    @unittest.skip
    def test_from_file(self):
        # TODO: implement test for `from_file`
        self.impedance_table_freq.from_file(filepath=None, reader=None)

    @unittest.skip
    def test_get_freq_y(self):
        # TODO: implement test for `get_freq_y`
        self.impedance_table_freq.get_freq_y(freq_x=None, sim=None)


class TestImpedanceTableTime(unittest.TestCase):
    @unittest.skip
    def test_from_file(self):
        # TODO: implement test for `from_file`
        self.impedance_table_time.from_file(filepath=None, reader=None)


class TestInductiveImpedance(unittest.TestCase):
    def setUp(self):
        self.inductive_impedance = InductiveImpedance(
            Z_over_n=34.6669349520904 / 10e9 * 11e3
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_get_freq_y(self):
        simulation = Mock(Simulation)
        simulation.ring.effective_circumference = 27e3
        beam = Mock(BeamBaseClass)
        beam.reference_velocity = 0.8 / c0
        freq_x = np.linspace(0, 1e9, 30)
        freq_y = self.inductive_impedance.get_impedance(
            freq_x=freq_x,
            simulation=simulation,
            beam=beam,
        )
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()
        # TODO PIN VALUE!


class TestResonators(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([400e6, 600e6, 1.2e9]),
            quality_factors=np.array([1, 2, 3]),
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_get_wake(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)

        time = np.linspace(-1e-9, 1e-9, 30)

        wake_impedance = self.resonators.get_wake_impedance(
            time=time,
            simulation=simulation,
            beam=beam,
        )
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(time, wake_impedance)
            plt.show()
        # TODO PIN VALUE!

    def test_get_impedance(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        freq_x = np.linspace(0, 1e9, 30)
        freq_y = self.resonators.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()
        # TODO PIN VALUE!
