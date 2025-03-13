from __future__ import annotations

import unittest

import cupy as cp
import numpy as np
from cupyx.profiler._time import _PerfCaseResult

from blond.beam.beam import Proton, Beam
from blond.gpu.beam_distributed import (
    DistributedMultiGpuArray,
    BeamDistributedSingleNode,
)
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.utils import bmath as bm


def _float_min(x):
    return cp.min(x)


def _total_runtime(res: _PerfCaseResult):
    return np.mean(res.gpu_times + res.cpu_times)


class TestMultiGpuArray(unittest.TestCase):
    def setUp(self):
        self.multi_gpu_array = DistributedMultiGpuArray(
            array_cpu=np.arange(1, 101), axis=0, mock_n_gpus=4
        )
        self.multi_gpu_array_1thread = DistributedMultiGpuArray(
            array_cpu=np.arange(1, 101), axis=0, mock_n_gpus=1
        )

    def test_map(self):
        self.multi_gpu_array.map_float(_float_min)
        results = self.multi_gpu_array.get_buffer()
        print(results)
        np.testing.assert_almost_equal(
            results, np.array([1.0, 26.0, 51.0, 76.0])
        )
    def test_min(self):
        results = self.multi_gpu_array.min()
        print(results)
        np.testing.assert_almost_equal(
            results, 1
        )
    def test_max(self):
        results = self.multi_gpu_array.max()
        print(results, type(results))
        np.testing.assert_almost_equal(
            results, 100
        )

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
        intensity = 1e3
        self.beam_distributed = BeamDistributedSingleNode(
            ring=ring,
            intensity=intensity,
            dE=dE.copy(),
            dt=dt.copy(),
            id=id_.copy(),
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

    def test_n_gpus(self):
        self.assertIsInstance(self.beam_distributed.n_gpus, int)

    def test_n_macroparticles_alive(self):
        self.assertEqual(
            self.beam_distributed.n_macroparticles_alive,
            self.beam_.n_macroparticles_alive,
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
        for beam in (self.beam_distributed, self.beam_):
            beam.losses_separatrix(
                ring=self.ring,
                rf_station=self.rf_station,
            )

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)

    def test_losses_longitudinal_cut(self):
        for beam in (self.beam_distributed, self.beam_):
            beam.losses_longitudinal_cut(dt_min=100 - 75, dt_max=100 - 85)

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)

    def test_losses_energy_cut(self):
        for beam in (self.beam_distributed, self.beam_):
            beam.losses_energy_cut(dE_min=float(75), dE_max=float(85))

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)

    def test_losses_below_energy(self):
        for beam in (self.beam_distributed, self.beam_):
            beam.losses_below_energy(dE_min=float(75))

        ids = self.beam_distributed.download_ids()
        np.testing.assert_array_equal(ids, self.beam_.id)

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

    def test_histogram(self):
        bm.use_gpu()
        hist = self.beam_distributed.histogram(
            out=np.empty(64), cut_left=1, cut_right=150
        )
        hist_npy = np.histogram(self.beam_.dt, bins=64, range=(1, 65))[0]

        np.testing.assert_array_equal(hist_npy, hist.get())
        bm.use_cpu()

    def test_kick(self):
        n_rf = 5
        charge = 1.0
        acceleration_kick = 1e3 * np.random.rand()
        voltage = np.random.randn(n_rf)
        omega_rf = np.random.randn(n_rf)
        phi_rf = np.random.randn(n_rf)
        bm.use_gpu()
        self.beam_distributed.kick(
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            n_rf=n_rf,
            acceleration_kick=acceleration_kick,
        )
        bm.use_cpu()

    def test_drift(self):
        t_rev = np.random.rand()
        length_ratio = np.random.uniform()
        eta_0, eta_1, eta_2 = np.random.randn(3)
        alpha_0, alpha_1, alpha_2 = np.random.randn(3)
        beta = np.random.rand()
        energy = np.random.rand()
        bm.use_gpu()
        self.beam_distributed.drift(
            solver="simple",
            t_rev=t_rev,
            length_ratio=length_ratio,
            alpha_order=1,
            eta_0=eta_0,
            eta_1=eta_1,
            eta_2=eta_2,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            beta=beta,
            energy=energy,
        )
        bm.use_cpu()


if __name__ == "__main__":
    unittest.main()
