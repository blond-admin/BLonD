import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy import ndarray as NumpyArray
from scipy.constants import pi
from scipy.constants import speed_of_light as c0
from scipy.signal import find_peaks

from blond import Numpy64Bit, backend
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
    fit_poles,
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
                    "../../../../blond/examples/scripts/resources/EX_02_Finemet.txt",
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

        impedance_from_wake = impedance_table.get_impedance_from_wake(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        impedance_from_wake2 = impedance_table.get_impedance_from_wake(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        impedance_from_wake3 = impedance_table.get_impedance_from_wake(
            time=time * 2, simulation=simulation, beam=beam, n_fft=len(time)
        )
        # assert cache hit
        np.testing.assert_allclose(
            copy_to_cpu(impedance_from_wake), copy_to_cpu(impedance_from_wake2)
        )
        # assert cache miss
        self.assertTrue(
            np.all(
                copy_to_cpu(impedance_from_wake3)
                != copy_to_cpu(impedance_from_wake2)
            )
        )

    def test_get_impedance_from_wake_within_bounds_no_warning(self):
        impedance_table = ImpedanceTableTime.from_file(
            filepath=callers_relative_path(
                "resources/example_impedance_table.csv", stacklevel=1
            ),
            reader=CsvReader(delimiter=","),
        )
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        # _wake_x is [1,2,3,4,5]; time within bounds triggers neither warning
        time = backend.linspace(2, 4, 10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            impedance_table.get_impedance_from_wake(
                time=time, simulation=simulation, beam=beam, n_fft=len(time)
            )
        boundary_warnings = [
            x for x in w if "outside boundaries" in str(x.message)
        ]
        self.assertEqual(len(boundary_warnings), 0)

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
        hash_before = impedance_table._cache_impedance_from_wake_hash
        _ = impedance_table.get_impedance_from_wake(
            time=t_arr, n_fft=30, simulation=simulation, beam=beam
        )
        assert hash_before != impedance_table._cache_impedance_from_wake_hash

        hash_before = impedance_table._cache_impedance_from_wake_hash
        _ = impedance_table.get_impedance_from_wake(
            time=t_arr, n_fft=30, simulation=simulation, beam=beam
        )
        assert hash_before == impedance_table._cache_impedance_from_wake_hash


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

        hash_before = self.inductive_impedance._cache_impedance_from_wake_hash
        _ = self.inductive_impedance.get_impedance_from_wake(
            time=backend.array([0.5, 1.5]),
            n_fft=5,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before
            != self.inductive_impedance._cache_impedance_from_wake_hash
        )
        hash_before = self.inductive_impedance._cache_impedance_from_wake_hash
        _ = self.inductive_impedance.get_impedance_from_wake(
            time=backend.array([0.5, 1.5]),
            n_fft=5,
            simulation=simulation,
            beam=beam,
        )
        # already hashed
        assert (
            hash_before
            == self.inductive_impedance._cache_impedance_from_wake_hash
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
            # R_CR = +R: the counter-rotating witness experiences the
            # inverted wake (witness-direction sign is part of the
            # parameter; an asymmetric fundamental mode has R_CR = -R).
            shunt_impedances_counter_rotating=np.array([1, 2, 3]),
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
        hash_before = local_res._cache_impedance_from_wake_hash
        _ = local_res.get_impedance_from_wake(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before != local_res._cache_impedance_from_wake_hash
        hash_before = local_res._cache_impedance_from_wake_hash
        _ = local_res.get_impedance_from_wake(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert hash_before == local_res._cache_impedance_from_wake_hash

        hash_before = (
            local_res._cache_impedance_from_wake_counter_rotation_hash
        )
        _ = local_res.get_impedance_from_wake_counter_rotation(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before
            != local_res._cache_impedance_from_wake_counter_rotation_hash
        )
        hash_before = (
            local_res._cache_impedance_from_wake_counter_rotation_hash
        )
        _ = local_res.get_impedance_from_wake_counter_rotation(
            time=backend.array([0.5, 1.5]),
            n_fft=6,
            simulation=simulation,
            beam=beam,
        )
        assert (
            hash_before
            == local_res._cache_impedance_from_wake_counter_rotation_hash
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
            (wake_potential[np.abs(time).argmin()]),
            0.5 * np.max(wake_potential),
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

    def test_calculate_envelope(self):
        time_axis = backend.linspace(
            0,
            backend.max(
                self.resonators._quality_factors / self.resonators._omega
            )
            * 20,
            100000,
        )
        env_time, envelope = self.resonators.calculate_envelope()
        ent_time_2, envelope_2 = self.resonators.calculate_envelope(
            time_axis=time_axis
        )

        np.testing.assert_allclose(
            copy_to_cpu(env_time),
            copy_to_cpu(ent_time_2),
            rtol=1e-12 if backend is Numpy64Bit else 1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            copy_to_cpu(envelope),
            copy_to_cpu(envelope_2),
            rtol=1e-12 if backend is Numpy64Bit else 1e-12,
            atol=0,
        )

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
            # R_CR = +R: the counter-rotating witness experiences the
            # inverted wake (the witness-direction sign is part of the
            # parameter's definition; an asymmetric fundamental mode has
            # R_CR = -R).
            shunt_impedances_counter_rotating=np.array([shut_imp]),
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

    def test_corot_counterrot_wake_decay_match_over_long_time(self):
        """
        Co- and counter-rotating wakes must share the envelope exp(-alpha t).

        Both ``get_wake`` and ``get_wake_counter_rotation`` describe the same
        resonator and must decay with the standard amplitude envelope
        ``exp(-alpha * t)``, ``alpha = omega_r / (2 * Q)`` -- consistent with
        ``calculate_envelope`` and the analytic resonator wake.
        """
        shunt, freq, q_factor = 1e3, 1e9, 50.0
        res = Resonators(
            shunt_impedances=np.array([shunt]),
            center_frequencies=np.array([freq]),
            quality_factors=np.array([q_factor]),
            shunt_impedances_counter_rotating=np.array([-shunt]),
        )
        alpha = float(copy_to_cpu(res._alpha[0]))

        # Several decay times, finely sampled to resolve the oscillation.
        time = backend.linspace(0.0, 6.0 / alpha, 400000)
        time_cpu = copy_to_cpu(time)
        corot = copy_to_cpu(res.get_wake(time=time))
        counter = copy_to_cpu(res.get_wake_counter_rotation(time=time))

        def envelope_fit(wake: NumpyArray):
            # Fit log(|peak amplitude|) vs time of the oscillation lobes.
            peak_idx, _ = find_peaks(np.abs(wake))
            interior = (time_cpu[peak_idx] > 0.2 / alpha) & (
                time_cpu[peak_idx] < 5.0 / alpha
            )
            peak_idx = peak_idx[interior]
            slope, intercept = np.polyfit(
                time_cpu[peak_idx], np.log(np.abs(wake[peak_idx])), 1
            )
            return float(-slope), peak_idx, slope, intercept

        rate_corot, peaks_corot, slope_corot, icpt_corot = envelope_fit(corot)
        rate_counter, peaks_counter, slope_counter, icpt_counter = (
            envelope_fit(counter)
        )

        DEV_DEBBUG = False
        if DEV_DEBBUG:
            with plt.rc_context({"font.size": 16}):
                fig, ax = plt.subplots()
                t_ns = time_cpu * 1e9
                ax.semilogy(
                    t_ns, np.abs(corot), color="C0", alpha=0.3, linewidth=1
                )
                ax.semilogy(
                    t_ns, np.abs(counter), color="C1", alpha=0.3, linewidth=1
                )
                ax.semilogy(
                    t_ns[peaks_corot],
                    np.abs(corot[peaks_corot]),
                    ".",
                    color="C0",
                    label=f"co-rot peaks (rate={rate_corot / alpha:.2f} alpha)",
                )
                ax.semilogy(
                    t_ns[peaks_counter],
                    np.abs(counter[peaks_counter]),
                    ".",
                    color="C1",
                    label=(
                        f"counter-rot peaks "
                        f"(rate={rate_counter / alpha:.2f} alpha)"
                    ),
                )
                ax.semilogy(
                    t_ns,
                    np.exp(icpt_corot + slope_corot * time_cpu),
                    "--",
                    color="C0",
                )
                ax.semilogy(
                    t_ns,
                    np.exp(icpt_counter + slope_counter * time_cpu),
                    "--",
                    color="C1",
                )
                # Expected physical envelope ~ exp(-alpha t), anchored at
                # the co-rotating intercept for visual reference.
                ax.semilogy(
                    t_ns,
                    np.exp(icpt_corot - alpha * time_cpu),
                    "k:",
                    label="expected exp(-alpha t)",
                )
                ax.set_xlabel("time [ns]")
                ax.set_ylabel("|wake| [V/pC]")
                ax.set_title("Resonator wake decay: co- vs counter-rotating")
                ax.legend()
                plt.tight_layout()
                plt.show()

        # Both wakes must decay at the physical rate alpha, and hence agree.
        np.testing.assert_allclose(rate_corot, alpha, rtol=0.05)
        np.testing.assert_allclose(rate_counter, alpha, rtol=0.05)
        np.testing.assert_allclose(rate_counter, rate_corot, rtol=0.05)

    def test_get_impedance_from_wake(self):
        if backend.float != np.float32:
            self.skipTest("test only configured for float32")

        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.linspace(-1e-9, 1e-9, int(1e3))

        before_hashes = (
            self.resonators._cache_impedance_from_wake_hash
        )  # check when hashed get changed and when not
        _ = self.resonators.get_impedance_from_wake(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )
        assert before_hashes != self.resonators._cache_impedance_from_wake_hash
        in_between_hashes = self.resonators._cache_impedance_from_wake_hash
        wake_imp = self.resonators.get_impedance_from_wake(
            time=time, simulation=simulation, beam=beam, n_fft=len(time)
        )  # should not be recalculated as time did not change
        assert (
            in_between_hashes
            == self.resonators._cache_impedance_from_wake_hash
        )

        wake_freq = self.resonators.get_impedance_from_wake_freq(time=time)

        pinned_result = np.load(
            callers_relative_path(
                "resources/get_impedance_from_wake_pinning.npz", stacklevel=1
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

    def test_get_impedance_from_wake_counterrotation(self):
        simulation = Mock(Simulation)
        beam = Mock(BeamBaseClass)
        time = backend.linspace(-1e-9, 1e-9, int(1e3))
        wake_imp_counter_rotation = (
            self.resonators.get_impedance_from_wake_counter_rotation(
                time=time, simulation=simulation, beam=beam, n_fft=len(time)
            )
        )
        wake_imp = self.resonators.get_impedance_from_wake(
            time=time,
            simulation=simulation,
            beam=beam,
            n_fft=len(time),
        )
        wake_freq = self.resonators.get_impedance_from_wake_freq(time=time)

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

    def test_get_vectorfit(self):
        from blond.testing.helpers import allclose_tolerances

        DEV_PLOT = False  # todo false
        resonators = Resonators(
            shunt_impedances=np.array(
                [
                    1e6,
                ]
            ),
            center_frequencies=np.array([1e9]),
            quality_factors=np.array(
                [
                    500,
                ]
            ),
        )  # values chosen such that they are easily reproducible in test of test_get_impedance

        freq = np.linspace(0, 4 * 1e9, 1000)
        imp = copy_to_cpu(
            resonators.get_impedance(backend.array(freq), None, None, False)
        )
        poles, residues, _ = resonators.get_vectorfit()
        imp2 = residues[0] / (1j * 2 * np.pi * freq - poles[0])
        imp2 += np.conjugate(residues[0]) / (
            1j * 2 * np.pi * freq - np.conjugate(poles[0])
        )
        if DEV_PLOT:
            plt.subplot(2, 1, 1)
            plt.plot(freq, imp.real)
            plt.plot(freq, imp2.real, "--")
            plt.subplot(2, 1, 2)
            plt.plot(freq, imp.imag)
            plt.plot(freq, imp2.imag, "--")
            plt.show()

        np.testing.assert_allclose(
            imp.real,
            imp2.real,
            **allclose_tolerances(imp.real),
        )
        np.testing.assert_allclose(
            imp.imag,
            imp2.imag,
            **allclose_tolerances(imp.imag),
        )


class TestFitPoles(unittest.TestCase):
    def test_recovers_resonator_impedance(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e6]),
            center_frequencies=np.array([1e9]),
            quality_factors=np.array([500]),
        )
        freq = np.linspace(0, 4e9, 1000)
        Z = copy_to_cpu(
            resonators.get_impedance(backend.array(freq), None, None, False)
        )

        poles, residues, rms_error, prop_coeff, const_coeff = fit_poles(
            freqs=freq,
            Z=Z,
            n_pole=1,
            max_iterations=20,
        )

        self.assertEqual(len(poles), 1)
        self.assertEqual(residues.shape, (1, 1))
        # rms_error is normalized — small value means a good fit
        self.assertLess(rms_error, 1e-3)

        # Reconstruct and compare to the analytical impedance
        residue = residues[0, 0]
        pole = poles[0]
        omega = 2j * np.pi * freq
        imp_fit = (
            residue / (omega - pole)
            + np.conjugate(residue) / (omega - np.conjugate(pole))
            + prop_coeff * 1j * 2 * np.pi * freq
            + const_coeff
        )
        # Compare on the resonance peak where amplitude is large
        peak = int(np.argmax(np.abs(Z)))
        sel = slice(max(0, peak - 50), min(len(freq), peak + 50))
        np.testing.assert_allclose(
            np.abs(imp_fit[sel]), np.abs(Z[sel]), rtol=0.05
        )

    def test_max_iterations_branch(self):
        resonators = Resonators(
            shunt_impedances=np.array([1e6]),
            center_frequencies=np.array([1e9]),
            quality_factors=np.array([500]),
        )
        freq = np.linspace(0, 4e9, 200)
        Z = copy_to_cpu(
            resonators.get_impedance(backend.array(freq), None, None, False)
        )
        # max_iterations=None branch
        _ = fit_poles(freqs=freq, Z=Z, n_pole=1)


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
    def test_get_impedance_from_wake(self):
        impedance_from_wake = self.twc.get_impedance_from_wake(
            time=backend.linspace(1, 1e-9),
            simulation=Mock(Simulation),
            beam=Mock(BeamBaseClass),
            n_fft=None,
        )
        # pinned to an arbitrary value, physics is not checked or guaranteed
        # to work
        SAVE_PINNED = False
        if SAVE_PINNED:
            np.savetxt(
                callers_relative_path(
                    "resources/TWC_impedance_from_wake_array_source.csv",
                    stacklevel=1,
                ),
                np.column_stack(
                    (impedance_from_wake.real, impedance_from_wake.imag)
                ),
            )
        impedance_from_wake_pinned = np.loadtxt(
            callers_relative_path(
                "resources/TWC_impedance_from_wake_array_source.csv",
                stacklevel=1,
            )
        )
        impedance_from_wake_pinned = (
            impedance_from_wake_pinned[:, 0]
            + 1j * impedance_from_wake_pinned[:, 1]
        )

        np.testing.assert_allclose(
            copy_to_cpu(impedance_from_wake),
            impedance_from_wake_pinned,
            rtol=1e-12,
        )

        impedance_from_wake_float = self.twc_floats.get_impedance_from_wake(
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
                    "resources/TWC_impedance_from_wake_float_source.csv",
                    stacklevel=1,
                ),
                np.column_stack(
                    (
                        impedance_from_wake_float.real,
                        impedance_from_wake_float.imag,
                    )
                ),
            )
        impedance_from_wake_pinned_float = np.loadtxt(
            callers_relative_path(
                "resources/TWC_impedance_from_wake_float_source.csv",
                stacklevel=1,
            )
        )
        impedance_from_wake_pinned_float = (
            impedance_from_wake_pinned_float[:, 0]
            + 1j * impedance_from_wake_pinned_float[:, 1]
        )

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            copy_to_cpu(impedance_from_wake_float),
            impedance_from_wake_pinned_float,
            rtol=1e-12,
        )

    @pytest.mark.backend_mutation
    def test_get_impedance(self):
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

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            copy_to_cpu(impedance),
            impedance_pinned,
            rtol=1e-12,
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

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            copy_to_cpu(impedance_float),
            impedance_pinned_float,
            rtol=1e-12,
        )

    def test_division_by_zero(self):
        pinned_values = [  # visual confirmation with DEV_DRAW lead to pinned
            # values.
            (7 + 0j),
            (3.506997824563613 - 0.3026719592604971j),
            (3.5 - 2.9166666666666576e-13j),
        ]
        DEV_DRAW = False
        for i, a_factor in enumerate((3e-12, 3, 3e12)):
            twc_floats = TravelingWaveCavity(
                3.5,
                4,
                a_factor,
            )

            impedance = twc_floats.get_impedance(
                freq_x=np.linspace(
                    twc_floats.frequency_R[0],
                    (1 + 1e-12) * twc_floats.frequency_R[0],
                ),
                beam=None,
                simulation=None,
            )
            if DEV_DRAW:
                plt.plot(impedance)
                plt.show()
            self.assertAlmostEqual(pinned_values[i].real, impedance[0].real)
            self.assertAlmostEqual(pinned_values[i].imag, impedance[0].imag)


if __name__ == "__main__":
    unittest.main()
