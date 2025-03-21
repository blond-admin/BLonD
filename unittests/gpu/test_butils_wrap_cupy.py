import unittest

import numpy as np
import pytest

from blond.beam.beam import Beam, Proton

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.utilities import is_in_separatrix as is_in_separatrix_cpu
from blond.utils import bmath as bm


class TestLossesSeparatrix(unittest.TestCase):
    def setUp(self):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs"""
        n_particles = int(10)
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

        dE = 1e9 * np.arange(0, n_particles).astype(precision.real_t)
        dt = 1e-8 * np.invert(np.arange(0, n_particles)).astype(
            precision.real_t)
        intensity = 1e3

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

    @pytest.importorskip('cupy')
    def test_executable(self):
        from blond.gpu.butils_wrap_cupy import (
            losses_separatrix as losses_separatrix_gpu,
        )
        bm.use_gpu()
        self.beam_.to_gpu()
        losses_separatrix_gpu(
            ring=self.ring,
            rf_station=self.rf_station,
            beam=self.beam_,
            dt=self.beam_.dt,
            dE=self.beam_.dE,
            id=self.beam_.id,
        )
        bm.use_cpu()

    @pytest.importorskip('cupy')
    def test_correct(self):
        from blond.gpu.butils_wrap_cupy import (
            losses_separatrix as losses_separatrix_gpu,
        )
        ids_correct = self.beam_.id.copy()
        lost_index = ~is_in_separatrix_cpu(
            ring=self.ring,
            rf_station=self.rf_station,
            beam=self.beam_,
            dt=self.beam_.dt.copy(),
            dE=self.beam_.dE.copy(),
        )
        assert np.any(lost_index), "Rewrite this testcase"

        ids_correct[lost_index] = 0
        bm.use_gpu()
        self.beam_.to_gpu()
        losses_separatrix_gpu(
            ring=self.ring,
            rf_station=self.rf_station,
            beam=self.beam_,
            dt=self.beam_.dt,
            dE=self.beam_.dE,
            id=self.beam_.id,
        )
        bm.use_cpu()
        self.beam_.to_cpu()
        np.testing.assert_array_equal(ids_correct, self.beam_.id)


if __name__ == "__main__":
    unittest.main()
