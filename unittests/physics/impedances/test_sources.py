import unittest
from pathlib import Path
from unittest import skipIf
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy import ndarray as NumpyArray
from scipy.constants import pi
from scipy.constants import speed_of_light as c0
from scipy.signal import find_peaks

from blond import Cupy32Bit, Cupy64Bit, Numpy32Bit, Numpy64Bit, backend
from blond.core.beam.base import BeamBaseClass
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.simulation.simulation import Simulation
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.readers import (
    CsvReader,
    ExampleImpedanceReader2,
    ModesExampleReader2,
)
from blond.physics.impedances.sources import (
    ImpedanceTableFreq,
    ImpedanceTableTime,
    InductiveImpedance,
    Resonators,
    TravelingWaveCavity,
)


class TestImpedanceTable(unittest.TestCase):
    @unittest.skip
    def test_from_file(self):
        # TODO: implement test for `from_file`
        self.impedance_table.from_file(filepath=None, reader=None)


class TestImpedanceTableFreq(unittest.TestCase):
    def setUp(self):
        pass
        # # TODO: implement test for `__init__`
        # self.impedance_table_freq = ImpedanceTableFreq(
        #     freq_x=None, freq_y=None
        # )

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

    def test_hashing(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)

        reader = ExampleImpedanceReader2(mode=ModesExampleReader2.SHORTED)
        freq_table_short = ImpedanceTableFreq.from_file(
            Path(
                callers_relative_path(
                    "../../../blond/examples/resources/EX_02_Finemet.txt",
                    stacklevel=1,
                )
            ),
            reader,
        )

        freq_x = backend.linspace(0, 1e9, 30)
        hash_before = freq_table_short._cache_impedance_hash
        _ = freq_table_short.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        assert hash_before != freq_table_short._cache_impedance_hash

        hash_before = freq_table_short._cache_impedance_hash
        _ = freq_table_short.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        assert hash_before == freq_table_short._cache_impedance_hash


class TestImpedanceTableTime(unittest.TestCase):
    def test_from_file(self):
        impedance_table = ImpedanceTableTime.from_file(
            filepath=callers_relative_path(
                "resources/example_impedance_table.csv", stacklevel=1
            ),
            reader=CsvReader(delimiter=","),
        )
        np.testing.assert_allclose(
            copy_to_cpu(impedance_table._wake_x), np.arange(1, 6)
        )
        np.testing.assert_allclose(
            copy_to_cpu(impedance_table._wake_y), 10 * np.arange(1, 6)
        )
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.linspace(0, 100)

        wake_impedance = impedance_table.get_wake_impedance(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        wake_impedance2 = impedance_table.get_wake_impedance(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        wake_impedance3 = impedance_table.get_wake_impedance(
            time=time * 2, simulation=simulation, beam=beam, n_fft=len(time)
        )
        # assert cache hit
        np.testing.assert_allclose(
            copy_to_cpu(wake_impedance), copy_to_cpu(wake_impedance2)
        )
        # assert cache miss
        self.assertTrue(
            np.all(
                copy_to_cpu(wake_impedance3) != copy_to_cpu(wake_impedance2)
            )
        )

    def test_hashing(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)

        impedance_table = ImpedanceTableTime.from_file(
            filepath=callers_relative_path(
                "resources/example_impedance_table.csv", stacklevel=1
            ),
            reader=CsvReader(delimiter=","),
        )

        t_arr = backend.linspace(0, 1e-9, 30)
        hash_before = impedance_table._cache_wake_impedance_hash
        _ = impedance_table.get_wake_impedance(
            time=t_arr, n_fft=30, simulation=simulation, beam=beam
        )
        assert hash_before != impedance_table._cache_wake_impedance_hash

        hash_before = impedance_table._cache_wake_impedance_hash
        _ = impedance_table.get_wake_impedance(
            time=t_arr, n_fft=30, simulation=simulation, beam=beam
        )
        assert hash_before == impedance_table._cache_wake_impedance_hash


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
        beam.reference = Mock(ReferenceCoordinates)

        beam.reference.velocity = 0.8 / c0
        freq_x = backend.linspace(0, 1e9, 30)
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
        np.testing.assert_allclose(copy_to_cpu(freq_y), pinned_freq_y)

    def test_hashing(self):
        simulation = Mock(Simulation)
        simulation.ring.circumference = 27e3
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)

        beam.reference.velocity = 0.8 / c0

        hash_before = self.inductive_impedance._cache_wake_impedance_hash
        _ = self.inductive_impedance.get_wake_impedance(
            time=backend.array([0.5, 1.5]),
            n_fft=5,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before != self.inductive_impedance._cache_wake_impedance_hash
        )
        hash_before = self.inductive_impedance._cache_wake_impedance_hash
        _ = self.inductive_impedance.get_wake_impedance(
            time=backend.array([0.5, 1.5]),
            n_fft=5,
            simulation=simulation,
            beam=beam,
        )
        # already hashed
        assert (
            hash_before == self.inductive_impedance._cache_wake_impedance_hash
        )

        hash_before = self.inductive_impedance._cache_derivative_hash
        freq_x = backend.linspace(0, 1e9, 30)
        _ = self.inductive_impedance.get_impedance(
            freq_x=freq_x,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before != self.inductive_impedance._cache_derivative_hash

        hash_before = self.inductive_impedance._cache_derivative_hash
        _ = self.inductive_impedance.get_impedance(
            freq_x=freq_x,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before == self.inductive_impedance._cache_derivative_hash


class TestResonators(unittest.TestCase):
    def setUp(self):
        self.resonators = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([500e6, 750e6, 2.0e9]),
            quality_factors=np.array([5, 5, 5]),
            shunt_impedances_counter_rotating=np.array([-1, -2, -3]),
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

    def test___init__floats_counter_rotating(self):
        _ = Resonators(
            shunt_impedances=float(1),
            center_frequencies=float(1),
            quality_factors=float(1),
            shunt_impedances_counter_rotating=float(1),
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

    def test___init__float_values(self):
        with self.assertRaises(RuntimeError):
            self.resonators = Resonators(
                shunt_impedances=float(1),
                center_frequencies=400e6,
                quality_factors=0.49,
            )

    def test_get_impedance_pinned(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        freq_x = backend.linspace(0, 1e9, 30)
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
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()

        # This is NOT a test if the physics is correct!
        # It should just allow to change internals of `get_impedance`
        # and guarantee that the result did not change
        np.testing.assert_allclose(copy_to_cpu(freq_y), pinned_freq_y)

    def test_get_impedance(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        min_freq, max_freq, num = 0, 4e9, 801
        freq_x = backend.linspace(min_freq, max_freq, num)

        before_hashes = (
            self.resonators._cache_impedance_hash
        )  # check when hashed get changed and when not
        _ = self.resonators.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        assert before_hashes != self.resonators._cache_impedance_hash
        in_between_hashes = self.resonators._cache_impedance_hash
        freq_y = self.resonators.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )  # should not be recalculated as time did not change
        assert in_between_hashes == self.resonators._cache_impedance_hash

        DEV_DEBBUG = False
        if DEV_DEBBUG:
            plt.plot(freq_x, np.abs(freq_y))
            plt.show()
        assert np.allclose(
            self.resonators._center_frequencies,
            copy_to_cpu(freq_x[find_peaks(copy_to_cpu(freq_y))[0]]),
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
                shunt_impedances_counter_rotating=np.array(
                    [-self.resonators._shunt_impedances[freq_ind]]
                ),
            )
            freq_y = local_res.get_impedance(
                freq_x=freq_x, simulation=simulation, beam=beam
            )
            assert np.allclose(
                copy_to_cpu(self.resonators._shunt_impedances[freq_ind]),
                copy_to_cpu(
                    np.abs(freq_y[find_peaks(copy_to_cpu(freq_y))[0]])
                ),
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

            freq_y_counterrot = local_res.get_impedance(
                freq_x=freq_x,
                simulation=simulation,
                beam=beam,
                counter_rotation=True,
            )
            np.testing.assert_allclose(
                copy_to_cpu(freq_y), copy_to_cpu(-freq_y_counterrot)
            )

    def test_hashing(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        local_res = Resonators(
            shunt_impedances=np.array([1, 2, 3]),
            center_frequencies=np.array([400e6, 600e6, 1.2e9]),
            quality_factors=np.array([1, 2, 3]),
            shunt_impedances_counter_rotating=np.array([-1, -2, -3]),
        )
        hash_before = local_res._cache_wake_impedance_hash
        _ = local_res.get_wake_impedance(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before != local_res._cache_wake_impedance_hash
        hash_before = local_res._cache_wake_impedance_hash
        _ = local_res.get_wake_impedance(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before == local_res._cache_wake_impedance_hash

        hash_before = local_res._cache_wake_impedance_counter_rotation_hash
        _ = local_res.get_wake_impedance_counter_rotation(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before
            != local_res._cache_wake_impedance_counter_rotation_hash
        )
        hash_before = local_res._cache_wake_impedance_counter_rotation_hash
        _ = local_res.get_wake_impedance_counter_rotation(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before
            == local_res._cache_wake_impedance_counter_rotation_hash
        )

        freq_x = backend.linspace(0, 1e9, 30)

        hash_before = local_res._cache_impedance_hash
        _ = local_res.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        assert hash_before != local_res._cache_impedance_hash

        hash_before = local_res._cache_impedance_hash
        _ = local_res.get_impedance(
            freq_x=freq_x, simulation=simulation, beam=beam
        )
        assert hash_before == local_res._cache_impedance_hash

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
        time = backend.linspace(-1e-9, 1.5e-9, 751, dtype=float)
        print(time[300])
        print(np.argmin(np.abs(time)))

        wake_potential = res.get_wake(time=time)
        wake_potential = copy_to_cpu(wake_potential)
        time = copy_to_cpu(time)
        assert wake_potential.shape == time.shape

        # check value at 0-time
        assert np.isclose(
            (wake_potential[np.abs((time)).argmin()]),
            0.5 * np.max((wake_potential)),
            rtol=1e-2,
        )
        # maximum point will only be true maximum with infinite points, hence high rtol

        # check maximum value
        assert np.isclose(
            copy_to_cpu(wake_potential)[copy_to_cpu(wake_potential).argmax()],
            2 * 2 * pi * freq * shut_imp / (2 * q_factor),
            rtol=1e-4,
        )  # *2 from heaviside

        # check periodicity
        t_min = 1 / res._center_frequencies[0]
        assert np.isclose(
            copy_to_cpu(time)[copy_to_cpu(wake_potential).argmin()], t_min / 2
        )

        DEV_DEBBUG = False
        if DEV_DEBBUG:
            with plt.rc_context({"font.size": 22}):
                plt.plot(time * 1e9, wake_potential, linewidth=3)
                plt.xlabel("time [ns]")
                plt.ylabel("Wake kernel [V/pC]")
                plt.tight_layout()
                # plt.savefig("")
                plt.show()

    def test_get_wake_counterrotation(self):
        freq, q_factor, shut_imp = (
            self.resonators._center_frequencies[0],
            1e10,
            self.resonators._shunt_impedances[0],
        )
        res = Resonators(
            shunt_impedances=np.array([shut_imp]),
            center_frequencies=np.array([freq]),
            quality_factors=np.array([q_factor]),
            shunt_impedances_counter_rotating=np.array([-shut_imp]),
        )  # high Q to avoid smearing of frequency --> minimum getting
        time = backend.linspace(-1e-9, 1.5e-9, 751)

        wake_potential = res.get_wake_counter_rotation(time=time)
        assert wake_potential.shape == time.shape
        DEV_DEBBUG = False
        if DEV_DEBBUG:
            with plt.rc_context({"font.size": 22}):
                plt.plot(
                    copy_to_cpu(time) * 1e9,
                    copy_to_cpu(wake_potential),
                    linewidth=3,
                )
                plt.xlabel("time [ns]")
                plt.ylabel("Wake kernel [V/pC]")
                plt.tight_layout()
                # plt.savefig("")
                plt.show()

        # check value at 0-time
        wake_potential: NumpyArray = copy_to_cpu(wake_potential)
        np.testing.assert_allclose(
            wake_potential[np.abs(copy_to_cpu(time)).argmin()],
            0.5 * np.min(wake_potential),
            rtol=1e-2,
        )
        # equivalent to above, just that the induced voltage should be negative

        # check maximum value
        assert np.isclose(
            wake_potential[wake_potential.argmax()],
            2 * 2 * pi * freq * shut_imp / (2 * q_factor),
            rtol=1e-4,
        )  # *2 from heaviside

        wake_potential_corot = copy_to_cpu(res.get_wake(time=time))
        np.testing.assert_allclose(wake_potential, -wake_potential_corot)

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

    def test_get_wake_impedance(self):
        if backend.float != np.float32:
            self.skipTest("test only configured for float32")

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.linspace(-1e-9, 1e-9, int(1e3))

        before_hashes = (
            self.resonators._cache_wake_impedance_hash
        )  # check when hashed get changed and when not
        _ = self.resonators.get_wake_impedance(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        assert before_hashes != self.resonators._cache_wake_impedance_hash
        in_between_hashes = self.resonators._cache_wake_impedance_hash
        wake_imp = self.resonators.get_wake_impedance(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )  # should not be recalculated as time did not change
        assert in_between_hashes == self.resonators._cache_wake_impedance_hash

        wake_freq = self.resonators.get_wake_impedance_freq(time=time)

        pinned_result = np.load(
            callers_relative_path(
                "resources/get_wake_impedance_pinning.npz", stacklevel=1
            )
        )
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(copy_to_cpu(np.abs(wake_imp)))
            plt.plot(copy_to_cpu(np.abs(pinned_result["wake_imp"])), "--")
            plt.show()

        np.testing.assert_allclose(
            copy_to_cpu(wake_imp),
            pinned_result["wake_imp"],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )
        np.testing.assert_allclose(
            copy_to_cpu(wake_freq),
            pinned_result["wake_freq"],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

    def test_get_wake_impedance_counterrotation(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.linspace(-1e-9, 1e-9, int(1e3))
        wake_imp_counter_rotation = (
            self.resonators.get_wake_impedance_counter_rotation(
                time=time, simulation=simulation, beam=beam, n_fft=len(time)
            )
        )
        wake_imp = self.resonators.get_wake_impedance(
            time=time,
            simulation=simulation,
            beam=beam,
            n_fft=len(time),
        )
        wake_freq = self.resonators.get_wake_impedance_freq(time=time)

        np.testing.assert_allclose(
            copy_to_cpu(wake_imp_counter_rotation), copy_to_cpu(-wake_imp)
        )
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.plot(wake_freq, np.abs(wake_imp))
            plt.xlim(0, 1.5e9)
            plt.show()

        save_cr_wake_imp = self.resonators._shunt_impedances_counter_rotating
        with self.assertRaises(RuntimeError):
            self.resonators._shunt_impedances_counter_rotating = None
            self.resonators.get_wake_counter_rotation(time=time)
        self.resonators._shunt_impedances_counter_rotating = save_cr_wake_imp


class TestTravelingWaveCavity(unittest.TestCase):
    def setUp(self):
        R_S = [1, 2, 3]
        frequency_R = [1, 2, 3]
        a_factor = [1, 2, 3]
        self.twc = TravelingWaveCavity(R_S, frequency_R, a_factor)
        self.twc_floats = TravelingWaveCavity(3.0, 3.0, 3.0)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @pytest.mark.backend_mutation
    def test_get_wake_impedance(self):
        if isinstance(backend, Numpy32Bit):
            backend.change_backend(Numpy64Bit)
        if isinstance(backend, Cupy32Bit):
            backend.change_backend(Cupy64Bit)
        wake_impedance = self.twc.get_wake_impedance(
            time=backend.linspace(1, 1e-9),
            simulation=Mock(Simulation),
            beam=Mock(BeamBaseClass),
            n_fft=None,
        )
        # pinned to an arbitrary value, physics is not checked or guaranteed
        # to work
        SAVE_PINNED = True
        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    "resources/TWC_wake_impedance_array_source.csv",
                    stacklevel=1,
                ),
                np.column_stack((wake_impedance.real, wake_impedance.imag)),
            )
        wake_impedance_pinned = np.loadtxt(
            callers_relative_path(
                "resources/TWC_wake_impedance_array_source.csv", stacklevel=1
            )
        )
        wake_impedance_pinned = (
            wake_impedance_pinned[:, 0] + 1j * wake_impedance_pinned[:, 1]
        )
        np.testing.assert_allclose(
            copy_to_cpu(wake_impedance),
            wake_impedance_pinned,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        wake_impedance_float = self.twc_floats.get_wake_impedance(
            time=backend.linspace(1, 1e-9),
            simulation=Mock(Simulation),
            beam=Mock(BeamBaseClass),
            n_fft=None,
        )

        # pinned to an arbitrary value, physics is not checked or guaranteed
        # to work
        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    "resources/TWC_wake_impedance_float_source.csv",
                    stacklevel=1,
                ),
                np.column_stack(
                    (wake_impedance_float.real, wake_impedance_float.imag)
                ),
            )
        wake_impedance_pinned_float = np.loadtxt(
            callers_relative_path(
                "resources/TWC_wake_impedance_float_source.csv", stacklevel=1
            )
        )
        wake_impedance_pinned_float = (
            wake_impedance_pinned_float[:, 0]
            + 1j * wake_impedance_pinned_float[:, 1]
        )
        np.testing.assert_allclose(
            copy_to_cpu(wake_impedance_float),
            wake_impedance_pinned_float,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

    @pytest.mark.backend_mutation
    def test_get_impedance(self):
        if isinstance(backend, Numpy32Bit):
            backend.change_backend(Numpy64Bit)
        if isinstance(backend, Cupy32Bit):
            backend.change_backend(Cupy64Bit)
        impedance = self.twc.get_impedance(
            freq_x=backend.linspace(0, 10),
            simulation=Mock(Simulation),
            beam=Mock(BeamBaseClass),
        )
        # pinned to an arbitrary value, physics is not checked or guaranteed
        # to work
        SAVE_PINNED = False
        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    "resources/TWC_impedance_array_source.csv", stacklevel=1
                ),
                np.column_stack((impedance.real, impedance.imag)),
            )
        impedance_pinned = np.loadtxt(
            callers_relative_path(
                "resources/TWC_impedance_array_source.csv", stacklevel=1
            )
        )
        impedance_pinned = impedance_pinned[:, 0] + 1j * impedance_pinned[:, 1]
        np.testing.assert_allclose(
            copy_to_cpu(impedance),
            impedance_pinned,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        impedance_float = self.twc_floats.get_impedance(
            freq_x=backend.linspace(0, 10),
            simulation=Mock(Simulation),
            beam=Mock(BeamBaseClass),
        )
        # pinned to an arbitrary value, physics is not checked or guaranteed
        # to work
        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    "resources/TWC_impedance_float_source.csv", stacklevel=1
                ),
                np.column_stack((impedance_float.real, impedance_float.imag)),
            )
        impedance_pinned_float = np.loadtxt(
            callers_relative_path(
                "resources/TWC_impedance_float_source.csv", stacklevel=1
            )
        )
        impedance_pinned_float = (
            impedance_pinned_float[:, 0] + 1j * impedance_pinned_float[:, 1]
        )
        np.testing.assert_allclose(
            copy_to_cpu(impedance_float),
            impedance_pinned_float,
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
