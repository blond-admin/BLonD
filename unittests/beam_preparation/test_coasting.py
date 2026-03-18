import unittest
import unittest.mock as mock

import numpy as np

from blond.beam_preparation import coasting
from blond.core.beam import beams
from blond.core.beam.particle_types import proton
from blond.cycles.magnetic_cycle import ConstantMagneticCycle
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import multi_backend_testcase


class TestCoasting(unittest.TestCase):
    def setUp(self):
        cycle = ConstantMagneticCycle(
            proton, 0.2, in_unit="bending field", bending_radius=8
        )

        ring = mock.Mock()
        ring.circumference = 100
        self.simulation = mock.Mock()
        self.simulation.ring = ring
        self.simulation.magnetic_cycle = cycle

        self.t_rev = cycle.get_t_rev_init(
            self.simulation.ring.circumference, proton
        )

        self.beam = beams.Beam(0, proton)

        self.simulation.magnetic_cycle.get_total_energy_init(
            particle_type=self.beam.particle_type,
        )

    @multi_backend_testcase
    def test_init(self):
        kwargs = {
            "energy_bins": np.array([1, 2, 3]),
            "energy_profile": np.array([0, 1, 0]),
            "start_time": 1,
            "stop_time": 2,
            "energy_offset": 3,
        }
        coast = coasting.Coasting(**kwargs, seed=4, n_macroparticles=1 << 30)

        for attr, target in kwargs.items():
            value = getattr(coast, attr)
            if isinstance(target, np.ndarray):
                np.testing.assert_array_equal(target, copy_to_cpu(value))
            else:
                self.assertEqual(target, value)

        self.assertEqual(4, coast._seed)
        self.assertEqual(1 << 30, coast._n_macroparticles_local)

    def test_init_exception(self):
        with self.assertRaises(ValueError):
            coasting.Coasting(0, [], [], start_time=1, stop_time=0)

    @multi_backend_testcase
    def test_prepare_beam_defaults(self):
        bins = np.linspace(-1, 1, 1000)
        dens = -(bins**2) + 1

        coast = coasting.Coasting(
            1 << 16, bins.tolist(), dens.tolist(), seed=0
        )

        coast.prepare_beam(self.simulation, self.beam)

        np.testing.assert_almost_equal(
            self.beam._dE.max(), bins[-1], decimal=2
        )
        np.testing.assert_almost_equal(self.beam._dE.min(), bins[0], decimal=2)
        np.testing.assert_almost_equal(self.beam._dE.mean(), 0, decimal=2)

        np.testing.assert_almost_equal(
            self.beam._dt.max(), self.t_rev, decimal=2
        )
        np.testing.assert_almost_equal(self.beam._dt.min(), 0, decimal=2)
        np.testing.assert_almost_equal(
            self.beam._dt.mean(), self.t_rev / 2, decimal=2
        )

    @multi_backend_testcase
    def test_prepare_beam_custom(self):
        bins = np.linspace(-1, 1, 1000)
        dens = -(bins**2) + 1

        coast = coasting.Coasting(
            1 << 16,
            bins,
            dens,
            start_time=self.t_rev * 0.25,
            stop_time=self.t_rev * 0.75,
            energy_offset=1,
        )

        coast.prepare_beam(self.simulation, self.beam)

        np.testing.assert_almost_equal(
            self.beam._dE.max(), bins[-1] + 1, decimal=2
        )
        np.testing.assert_almost_equal(
            self.beam._dE.min(), bins[0] + 1, decimal=2
        )
        np.testing.assert_almost_equal(self.beam._dE.mean(), 1, decimal=2)

        np.testing.assert_almost_equal(
            self.beam._dt.max(), self.t_rev * 0.75, decimal=2
        )
        np.testing.assert_almost_equal(
            self.beam._dt.min(), self.t_rev * 0.25, decimal=2
        )
        np.testing.assert_almost_equal(
            self.beam._dt.mean(), self.t_rev / 2, decimal=2
        )

    def test_prepare_beam_varying_offset(self):
        bins = np.linspace(-1, 1, 1000)
        dens = -(bins**2) + 1

        coast = coasting.Coasting(
            1 << 16,
            bins,
            dens,
            energy_offset=np.array([[0, self.t_rev], [-1, 1]]),
        )

        coast.prepare_beam(self.simulation, self.beam)

        np.testing.assert_almost_equal(
            self.beam._dE.max(), bins[-1] + 1, decimal=1
        )
        np.testing.assert_almost_equal(
            self.beam._dE.min(), bins[0] - 1, decimal=1
        )
        np.testing.assert_almost_equal(self.beam._dE.mean(), 0, decimal=1)

        np.testing.assert_almost_equal(
            self.beam._dt.max(), self.t_rev, decimal=2
        )
        np.testing.assert_almost_equal(self.beam._dt.min(), 0, decimal=2)
        np.testing.assert_almost_equal(
            self.beam._dt.mean(), self.t_rev / 2, decimal=2
        )

        time_sorted = np.argsort(self.beam._dt.array_local)
        self.assertTrue(self.beam._dE.array_local[time_sorted[0]] < 0)
        self.assertTrue(self.beam._dE.array_local[time_sorted[-1]] > 0)


if __name__ == "__main__":
    unittest.main()
