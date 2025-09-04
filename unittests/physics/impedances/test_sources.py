import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import pi
from scipy.constants import speed_of_light as c0
from scipy.signal import find_peaks

from blond._core.beam.base import BeamBaseClass
from blond._core.simulation.simulation import Simulation
from blond.physics.impedances.sources import (
    ImpedanceTableFreq,
    InductiveImpedance,
    Resonators,
)


class TestImpedanceTable(unittest.TestCase):
    @unittest.skip
    def test_from_file(self):
        # TODO: implement test for `from_file`
        self.impedance_table.from_file(filepath=None, reader=None)


class TestImpedanceTableFreq(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.impedance_table_freq = ImpedanceTableFreq(
            freq_x=None, freq_y=None
        )

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

        pinned_freq_y = np.array(
            # this might need to change if found that
            # get_impedance physics is incorrect
            [
                0j,
                1.327867777722188e16j,
                2.64016751983167e16j,
                3.9215137116993096e16j,
                5.156883740771454e16j,
                6.331794022966821e16j,
                7.432469809457386e16j,
                8.446006683004016e16j,
                9.360521850449973e16j,
                1.0165293457606154e17j,
                1.0850886293188341e17j,
                1.1409262408042843e17j,
                1.1833875352751144e17j,
                1.2119746928763856e17j,
                1.22635255532271e17j,
                1.22635255532271e17j,
                1.2119746928763856e17j,
                1.1833875352751144e17j,
                1.1409262408042845e17j,
                1.0850886293188341e17j,
                1.0165293457606154e17j,
                9.360521850449973e16j,
                8.446006683004018e16j,
                7.432469809457386e16j,
                6.331794022966825e16j,
                5.156883740771458e16j,
                3.921513711699308e16j,
                2.640167519831673e16j,
                1.3278677777221898e16j,
                -39.500437208149435j,
            ]
        )
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()
        # This is NOT a test if the physics is correct!
        # It should just allow to change internals of `get_impedance`
        # and guarantee that the result did not change
        np.testing.assert_allclose(freq_y, pinned_freq_y)


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

    def test_get_impedance_pinned(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        freq_x = np.linspace(0, 1e9, 30)
        local_res = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([400e6, 600e6, 1.2e9]),
            quality_factors=np.array([1, 2, 3]),
        )
        freq_y = local_res.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )

        pinned_freq_y = np.array(
            # this might need to change if found that
            # get_impedance physics is incorrect
            [
                0j,
                (0.00942358651488795 + 0.17257266539124913j),
                (0.03847829128074856 + 0.34598502512183227j),
                (0.08951350665665338 + 0.5204706751495055j),
                (0.1663642473536873 + 0.694876830308083j),
                (0.27394129680743806 + 0.8655824712339767j),
                (0.4169903786734268 + 1.0252003539257513j),
                (0.597367508904515 + 1.1617714626882443j),
                (0.8098132299529293 + 1.260286679759775j),
                (1.0387297231581394 + 1.3088371527047062j),
                (1.2617888995045035 + 1.3082130088125705j),
                (1.4634874229891084 + 1.2759901511026408j),
                (1.649396831946504 + 1.2354615637236905j),
                (1.8457486837443287 + 1.194271207370826j),
                (2.080707633067911 + 1.1284228930251694j),
                (2.3538572306159256 + 0.9836550243814897j),
                (2.606598464998768 + 0.7073482747932069j),
                (2.7338902681213844 + 0.3126019580772236j),
                (2.6698698182119367 - 0.09491115416842288j),
                (2.4546067744675466 - 0.39781425368699297j),
                (2.184014624168805 - 0.5539547699273479j),
                (1.931984543412328 - 0.5867513809204155j),
                (1.731407279490519 - 0.5372728771383196j),
                (1.5898770746147266 - 0.4387931168944583j),
                (1.505515550177968 - 0.31291770276423536j),
                (1.47515159586377 - 0.17301463290083663j),
                (1.4974580247385165 - 0.028301503021182173j),
                (1.5737339298195334 + 0.11277261150134943j),
                (1.7074625414604356 + 0.2395805248705769j),
                (1.902597177481868 + 0.33642772231686235j),
            ]
        )
        DEV_DEBBUG = True
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()

        # This is NOT a test if the physics is correct!
        # It should just allow to change internals of `get_impedance`
        # and guarantee that the result did not change
        np.testing.assert_allclose(freq_y, pinned_freq_y)

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
            with plt.rc_context({"font.size": 22}):
                plt.plot(time * 1e9, wake_potential, linewidth=3)
                plt.xlabel("time [ns]")
                plt.ylabel("Wake kernel [V/pC]")
                plt.tight_layout()
                # plt.savefig("")
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
                quality_factors=np.array(
                    [self.resonators._quality_factors[freq_ind]]
                ),
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
                        freq_x
                        - self.resonators._center_frequencies[freq_ind] / 2
                    ).argmin()
                ],
            )

    def test_get_wake_impedance(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = np.linspace(-1e-9, 1e-9, int(1e3))
        wake_imp = self.resonators.get_wake_impedance(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        wake_freq = self.resonators.get_wake_impedance_freq(time=time)
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(wake_freq, np.abs(wake_imp))
            plt.xlim(0, 1.5e9)
            plt.show()
        # TODO PIN VALUE!
