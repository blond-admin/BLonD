from __future__ import annotations

import unittest

try:
    import cupy as cp
except ModuleNotFoundError:
    raise unittest.SkipTest('Cupy not found!')
import numpy as np
from cupy.cuda import Device
from cupyx.profiler._time import _PerfCaseResult, benchmark
from blond.beam.beam import Proton, Beam
from blond.gpu import GPU_DEV
from blond.gpu.beam_distributed import (
    DistributedMultiGpuArray,
    BeamDistributedSingleNode,
)
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.utils import bmath as bm


def _float_min(x):
    return cp.min(x)


def count_nonzero(x):
    return cp.count_nonzero(x)


def _total_runtime(res: _PerfCaseResult):
    return np.mean(res.gpu_times + res.cpu_times)


class TestMultiGpuArray(unittest.TestCase):
    def setUp(self):
        GPU_DEV.block_size /= 4
        self.multi_gpu_array = DistributedMultiGpuArray(
            array_cpu=np.arange(1, 101), axis=0, mock_n_gpus=4
        )
        self.multi_gpu_array_1thread = DistributedMultiGpuArray(
            array_cpu=np.arange(1, 101), axis=0, mock_n_gpus=1
        )

    def tearDown(self):
        GPU_DEV.block_size *= 4

    def test_map(self):
        self.multi_gpu_array.map_float(_float_min)
        results = self.multi_gpu_array.get_buffer()
        print(results)
        np.testing.assert_almost_equal(
            results, np.array([1.0, 26.0, 51.0, 76.0])
        )

    def test_map_int(self):
        self.multi_gpu_array.map_int(count_nonzero)
        results = self.multi_gpu_array.get_buffer_int()
        np.testing.assert_almost_equal(results, 25 * np.ones(4))

    def test_min(self):
        results = self.multi_gpu_array.min()
        print(results)
        np.testing.assert_almost_equal(results, 1)

    def test_max(self):
        results = self.multi_gpu_array.max()
        print(results, type(results))
        np.testing.assert_almost_equal(results, 100)

    def test_download_array(self):
        res = self.multi_gpu_array.download_array()
        np.testing.assert_array_equal(res, np.arange(1, 101))


class TestBeamDistributedSingleNode(unittest.TestCase):
    def setUp(self):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs"""
        n_particles = int(1000)
        id_ = np.arange(1, n_particles + 1)
        id_[: (n_particles // 2)] = 0

        ring = Ring(
            ring_length=2 * np.pi * 1100.009,
            alpha_0=1 / 18.0**2,
            synchronous_data=25.92e9,
            particle=Proton(),
            n_turns=1,
        )
        from blond.utils import precision

        dE = np.arange(0, n_particles).astype(precision.real_t)
        dt = np.invert(np.arange(0, n_particles)).astype(precision.real_t)
        self.dE_org = dE.copy()
        self.dt_org = dt.copy()
        self.id_org = id_.copy()
        intensity = 1e3
        self.beam_distributed = BeamDistributedSingleNode(
            ring=ring,
            intensity=intensity,
            dE=dE.copy(),
            dt=dt.copy(),
            id_=id_.copy(),
            mock_n_gpus=4,
        )
        self.beam_ = Beam(
            ring=ring,
            n_macroparticles=len(dt),
            intensity=intensity,
            dt=dt.copy(),
            dE=dE.copy(),
        )
        self.beam_.id = id_
        self.ring = ring
        self.rf_station = RFStation(
            ring=self.ring, harmonic=4620, voltage=4.5e6, phi_rf_d=0.0
        )

    def test_from_beam(self):
        BeamDistributedSingleNode.from_beam(
            beam=self.beam_, ring=self.ring, mock_n_gpus=4
        )

    def test___init_profile_multi_gpu(self):
        self.beam_distributed._init_profile_multi_gpu(64)

    def test_n_gpus(self):
        self.assertIsInstance(self.beam_distributed.n_gpus, int)

    def test_n_macroparticles_alive(self):
        self.assertEqual(
            self.beam_distributed.n_macroparticles_alive,
            self.beam_.n_macroparticles_alive,
        )

    def test_download_ids(self):
        np.testing.assert_array_equal(
            self.beam_distributed.download_ids(), self.id_org
        )

    def test_download_dts(self):
        np.testing.assert_array_equal(
            self.beam_distributed.download_dts(), self.dt_org
        )

    def test_download_dEs(self):
        np.testing.assert_array_equal(
            self.beam_distributed.download_dEs(), self.dE_org
        )

    def test_eliminate_lost_particles(self):
        self.beam_.eliminate_lost_particles()
        self.beam_distributed.eliminate_lost_particles()
        self.assertEqual(
            self.beam_.n_macroparticles_alive,
            self.beam_distributed.n_macroparticles_alive,
        )
        self.assertEqual(
            self.beam_.n_macroparticles,
            self.beam_distributed.n_macroparticles,
        )
        np.testing.assert_array_equal(
            self.beam_.id, self.beam_distributed.id_multi_gpu.download_array()
        )
        np.testing.assert_array_equal(
            self.beam_.dE, self.beam_distributed.dE_multi_gpu.download_array()
        )
        np.testing.assert_array_equal(
            self.beam_.dt, self.beam_distributed.dt_multi_gpu.download_array()
        )

    def test_statistics(self):
        self.beam_distributed.statistics()
        self.beam_.statistics()

        self.assertEqual(
            self.beam_distributed.mean_dt,
            self.beam_.mean_dt,
        )

        self.assertEqual(
            self.beam_distributed.sigma_dt,
            self.beam_.sigma_dt,
        )

        self.assertEqual(
            self.beam_distributed.mean_dE,
            self.beam_.mean_dE,
        )

        self.assertEqual(
            self.beam_distributed.sigma_dE,
            self.beam_.sigma_dE,
        )

    def test_losses_separatrix(self):
        self.beam_.losses_separatrix(
            ring=self.ring,
            rf_station=self.rf_station,
        )
        bm.use_gpu()
        self.beam_distributed.losses_separatrix(
            ring=self.ring,
            rf_station=self.rf_station,
        )

        ids = self.beam_distributed.download_ids()
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()
        np.testing.assert_array_equal(ids, self.beam_.id)

    def test_losses_longitudinal_cut(self):
        bm.use_gpu()

        for beam in (self.beam_distributed, self.beam_):
            beam.losses_longitudinal_cut(dt_min=100 - 75., dt_max=100 - 85.)

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

    def test_losses_energy_cut(self):
        bm.use_gpu()

        for beam in (self.beam_distributed, self.beam_):
            beam.losses_energy_cut(dE_min=float(75), dE_max=float(85))

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

    def test_losses_below_energy(self):
        bm.use_gpu()
        for beam in (self.beam_distributed, self.beam_):
            beam.losses_below_energy(dE_min=float(75))

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

    def test_dE_mean(self):
        for ignore_id_0 in (False, True):
            self.assertEqual(
                self.beam_.dE_mean(ignore_id_0=ignore_id_0),
                self.beam_distributed.dE_mean(ignore_id_0=ignore_id_0),
            )

    def test_dE_std(self):
        for ignore_id_0 in (False,):
            self.assertEqual(
                self.beam_.dE_std(ignore_id_0=ignore_id_0),
                self.beam_distributed.dE_std(ignore_id_0=ignore_id_0),
            )

    def test_dE_std_ignore_id_0(self):
        for ignore_id_0 in (True,):
            self.assertEqual(
                self.beam_.dE_std(ignore_id_0=ignore_id_0),
                self.beam_distributed.dE_std(ignore_id_0=ignore_id_0),
            )

    def test_dt_mean(self):
        for ignore_id_0 in (False, True):
            self.assertEqual(
                self.beam_.dt_mean(ignore_id_0=ignore_id_0),
                self.beam_distributed.dt_mean(ignore_id_0=ignore_id_0),
            )

    def test_dt_std(self):
        for ignore_id_0 in (False,):
            self.assertEqual(
                np.float64(cp.std(self.beam_distributed.download_dts())),
                self.beam_distributed.dt_std(ignore_id_0=ignore_id_0),
            )
            self.assertEqual(
                self.beam_.dt_std(ignore_id_0=ignore_id_0),
                self.beam_distributed.dt_std(ignore_id_0=ignore_id_0),
            )

    def test_dt_std_ignore_id_0(self):
        for ignore_id_0 in (True,):
            self.assertEqual(
                self.beam_.dt_std(ignore_id_0=ignore_id_0),
                self.beam_distributed.dt_std(ignore_id_0=ignore_id_0),
            )

    def test_dt_min(self):
        assert np.min(self.dt_org) == self.beam_distributed.dt_min()

    def test_dE_min(self):
        assert np.min(self.dE_org) == self.beam_distributed.dE_min()

    def test_dt_max(self):
        assert np.max(self.dt_org) == self.beam_distributed.dt_max()

    def test_dE_max(self):
        assert np.max(self.dE_org) == self.beam_distributed.dE_max()

    def test_histogram(self):
        bm.use_gpu()
        hist = cp.empty(64)
        self.beam_distributed.slice_beam(
            profile=hist, cut_left=1, cut_right=150
        )
        hist_npy = np.histogram(self.beam_.dt, bins=64, range=(1, 65))[0]

        np.testing.assert_array_equal(hist_npy, hist.get())
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

    def test_kick_NDArray(self):
        acceleration_kicks = 1e3 * np.random.randn(5)
        bm.use_gpu()
        self.beam_distributed.kick(
            rf_station=self.rf_station,
            acceleration_kicks=acceleration_kicks,
            turn_i=1,
        )
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

    def test_kick2_CupyNDArray(self):
        acceleration_kicks = 1e3 * np.random.randn(5)
        bm.use_gpu()
        self.rf_station.to_gpu() # voltage, omega_rf, phi_rf is sent to GPU
        self.beam_distributed.kick(
            rf_station=self.rf_station,
            acceleration_kicks=acceleration_kicks,
            turn_i=1,
        )
        Device().synchronize() # Will crash if a MemoryError happened before
        #  todo make comparison
        bm.use_cpu()

    def test_drift(self):
        bm.use_gpu()

        self.beam_distributed.drift(
            solver="simple", rf_station=self.rf_station, turn_i=0
        )
        Device().synchronize() # Will crash if a MemoryError happened before
        #  todo make comparison
        bm.use_cpu()

    def test_linear_interp_kick(self):
        bm.use_gpu()
        for i in range(2):
            self.beam_distributed.linear_interp_kick(
                voltage=cp.random.randn(64),  # TODO
                bin_centers=cp.arange(64, dtype=float),
                charge=self.beam_distributed.particle.charge,
                acceleration_kick=32.0,
            )
        Device().synchronize() # Will crash if a MemoryError happened before
        #  todo make comparison
        bm.use_cpu()

    def test_kickdrift_considering_periodicity(self):
        bm.use_gpu()
        self.beam_distributed.kickdrift_considering_periodicity(
            acceleration_kicks=np.zeros(
                self.rf_station.n_turns, dtype=float
            ),
            rf_station=self.rf_station,
            solver="simple",
            turn_i=0,
        ) #  todo make comparison
        Device().synchronize() # Will crash if a MemoryError happened before
        bm.use_cpu()

if __name__ == "__main__":
    unittest.main()
