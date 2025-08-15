import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light as c0
from scipy.constants import pi
from scipy.signal import find_peaks

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
        simulation.ring.circumference = 27e3
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
            center_frequencies=np.array([500e6, 750e6, 2.0e9]),
            quality_factors=np.array([5, 5, 5]),
        )  # values chosen such that they are easily reproducible in test of test_get_impedance

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test___init__wrong_lengths(self):
        with self.assertRaises(AssertionError):
            self.resonators = Resonators(
                shunt_impedances=np.array([1, 2, 3]),
                center_frequencies=np.array([400e6, 600e6, 1.2e9]),
                quality_factors=np.array([1, 2]),
            )
        with self.assertRaises(AssertionError):
            self.resonators = Resonators(
                shunt_impedances=np.array([1, 2, 3]),
                center_frequencies=np.array([400e6, 600e6]),
                quality_factors=np.array([1, 2, 3]),
            )

    def test___init__neg_freq(self):
        with self.assertRaises(RuntimeError):
            self.resonators = Resonators(
                shunt_impedances=np.array([1]),
                center_frequencies=np.array([-400e6]),
                quality_factors=np.array([1]),
            )

    def test___init__small_Q(self):
        with self.assertRaises(RuntimeError):
            self.resonators = Resonators(
                shunt_impedances=np.array([1]),
                center_frequencies=np.array([400e6]),
                quality_factors=np.array([0.49]),
            )

    def test_get_wake(self):
        freq, q_factor, shut_imp = (
            self.resonators._center_frequencies[0],
            1e10,
            self.resonators._shunt_impedances[0],
        )
        res = Resonators(
            shunt_impedances=np.array([shut_imp]),
            center_frequencies=np.array([freq]),
            quality_factors=np.array([q_factor]),
        )  # high Q to avoid smearing of frequency --> minimum getting
        time = np.linspace(-1e-9, 1.5e-9, 751)

        wake_potential = res.get_wake(time=time)
        assert wake_potential.shape == time.shape

        # check value at 0-time
        assert np.isclose(
            wake_potential[np.abs(time).argmin()],
            0.5 * np.max(wake_potential),
            rtol=1e-2,
        )
        # maximum point will only be true maximum with infinite points, hence high rtol

        # check maximum value
        assert np.isclose(
            wake_potential[wake_potential.argmax()],
            2 * 2 * pi * freq * shut_imp / (2 * q_factor),
            rtol=1e-4,
        )  # *2 from heaviside

        # check periodicity
        t_min = 1 / res._center_frequencies[0]
        assert np.isclose(time[wake_potential.argmin()], t_min / 2)

        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(time, wake_potential)
            plt.show()

    def test_get_impedance(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        min_freq, max_freq, num = 0, 4e9, 801
        freq_x = np.linspace(min_freq, max_freq, num)
        freq_y = self.resonators.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()
        assert np.allclose(
            self.resonators._center_frequencies,
            freq_x[find_peaks(freq_y)[0]],
            atol=(max_freq - min_freq) / num / 2,
        )  # closeness of peaks to centre frequency
        for freq_ind in range(
            0, len(self.resonators._shunt_impedances)
        ):  # has to be single resonator, otherwise overlaps will occur
            local_res = Resonators(
                shunt_impedances=np.array(
                    [self.resonators._shunt_impedances[freq_ind]]
                ),
                center_frequencies=np.array(
                    [self.resonators._center_frequencies[freq_ind]]
                ),
                quality_factors=np.array([self.resonators._quality_factors[freq_ind]]),
            )
            freq_y = local_res.get_impedance(
                freq_x=freq_x, simulation=simulation, beam=beam
            )
            assert np.allclose(
                self.resonators._shunt_impedances[freq_ind],
                np.abs(freq_y[find_peaks(freq_y)[0]]),
            )
            assert np.isclose(
                self.resonators._shunt_impedances[freq_ind]
                / (1 - 1.5j * self.resonators._quality_factors[freq_ind]),
                freq_y[
                    np.abs(
                        freq_x - self.resonators._center_frequencies[freq_ind] / 2
                    ).argmin()
                ],
            )

    def test_get_wake_impedance(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = np.linspace(-1e-9, 1e-9, int(1e3))
        wake_imp = self.resonators.get_wake_impedance(
            time=time, simulation=simulation, beam=beam
        )
        wake_freq = self.resonators.get_wake_impedance_freq(time=time)
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(wake_freq, np.abs(wake_imp))
            plt.xlim(0, 1.5e9)
            plt.show()
        # TODO PIN VALUE!
