from __future__ import annotations

import unittest

import cupy as cp
import numpy as np

from blond.beam.beam import Proton, Beam
from blond.beam.beam_distributed import (
    MultiGpuArray,
    BeamDistributedSingleNode,
)
from blond.input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation


class TestMultiGpuArray(unittest.TestCase):
    def setUp(self):
        self.multi_gpu_array = MultiGpuArray(array_cpu=np.ones(100), axis=0)

    def test_map(self):
        results = self.multi_gpu_array.map(lambda x: cp.add(x, 2))
        np.testing.assert_almost_equal(results, 3)

    def test_map_inplace(self):
        out = np.empty(100)
        self.multi_gpu_array.map_inplace(lambda x: cp.add(x, 2), out=out)
        np.testing.assert_almost_equal(out, 3)


class TestBeamDistributedSingleNode(unittest.TestCase):
    def setUp(self):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs"""
        id_ = np.arange(1, 100 + 1)
        id_[:50] = 0

        ring = Ring(
            ring_length=2 * np.pi * 1100.009,
            alpha_0=1 / 18.0**2,
            synchronous_data=25.92e9,
            particle=Proton(),
            n_turns=1,
        )
        dE = np.arange(0, 100)
        dt = np.invert(np.arange(0, 100))
        intensity = 1e3
        self.beam_distributed = BeamDistributedSingleNode(
            ring=ring,
            intensity=intensity,
            dE=dE,
            dt=dt,
            id=id_,
        )
        self.beam_ = Beam(
            ring=ring,
            n_macroparticles=len(dt),
            intensity=intensity,
            dt=dt,
            dE=dE,
        )
        self.ring = ring

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
            self.beam_distributed.n_macroparticles_alive,
            self.beam_.n_macroparticles_alive,
        )
        self.assertEqual(
            self.beam_distributed.n_macroparticles,
            self.beam_.n_macroparticles,
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
                rf_station=RFStation(),  # TODO
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
                self.beam_distributed.dE_mean(ignore_id_0=ignore_id_0),
                self.beam_.dE_mean(ignore_id_0=ignore_id_0),
            )

    def test_dE_std(self):
        for ignore_id_0 in (False, True):
            self.assertEqual(
                self.beam_distributed.dE_std(ignore_id_0=ignore_id_0),
                self.beam_.dE_std(ignore_id_0=ignore_id_0),
            )

    def test_dt_mean(self):
        for ignore_id_0 in (False, True):
            self.assertEqual(
                self.beam_distributed.dt_mean(ignore_id_0=ignore_id_0),
                self.beam_.dt_mean(ignore_id_0=ignore_id_0),
            )

    def test_dt_std(self):
        for ignore_id_0 in (False, True):
            self.assertEqual(
                self.beam_distributed.dt_std(ignore_id_0=ignore_id_0),
                self.beam_.dt_std(ignore_id_0=ignore_id_0),
            )
