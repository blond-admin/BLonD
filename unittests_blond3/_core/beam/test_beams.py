import unittest
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np

from blond3 import Beam, proton, Simulation
from blond3._core.beam.beams import ProbeBeam


class TestBeam(unittest.TestCase):
    def setUp(self):
        self.beam = Beam(
            n_particles=1e12, particle_type=proton, is_counter_rotating=False
        )
        self.beam.setup_beam(dE=np.linspace(1, 10, 10), dt=np.linspace(20, 30, 10))

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_common_array_size(self):
        self.assertEqual(10, self.beam.common_array_size)

    def test_dE_min(self):
        self.assertEqual(1, self.beam.dE_min)

    def test_dE_max(self):
        self.assertEqual(10, self.beam.dE_max)

    def test_dt_min(self):
        self.assertEqual(20, self.beam.dt_min)

    def test_dt_max(self):
        self.assertEqual(30, self.beam.dt_max)

    def test_invalidate_cache(self):
        self.beam.invalidate_cache()

    def test_invalidate_cache_dE1(self):
        before = self.beam.dE_max
        self.beam.setup_beam(dE=np.linspace(1, 20, 10), dt=np.linspace(20, 30, 10))
        after2 = self.beam.dE_max
        self.assertNotEqual(before, after2)

    def test_invalidate_cache_dE2(self):
        before = self.beam.dE_max
        self.beam._dE[:] = 100  # changes, but cache unchanged
        # so result should be unchanged, even though _dE changed
        self.assertEqual(before, self.beam.dE_max)

        self.beam.invalidate_cache_dE()
        self.assertEqual(100, self.beam.dE_max)

    def test_invalidate_cache_dE_min1(self):
        before = self.beam.dE_min
        self.beam.setup_beam(dE=np.linspace(2, 20, 10), dt=np.linspace(20, 30, 10))
        after2 = self.beam.dE_min
        self.assertNotEqual(before, after2)

    def test_invalidate_cache_dE_min2(self):
        before = self.beam.dE_min
        self.beam._dE[:] = 100
        self.assertEqual(before, self.beam.dE_min)

        self.beam.invalidate_cache_dE()
        self.assertEqual(100, self.beam.dE_min)

    def test_invalidate_cache_dt_max1(self):
        before = self.beam.dt_max
        self.beam.setup_beam(dE=np.linspace(1, 20, 10), dt=np.linspace(10, 40, 10))
        after2 = self.beam.dt_max
        self.assertNotEqual(before, after2)

    def test_invalidate_cache_dt_max2(self):
        before = self.beam.dt_max
        self.beam._dt[:] = 50
        self.assertEqual(before, self.beam.dt_max)

        self.beam.invalidate_cache_dt()
        self.assertEqual(50, self.beam.dt_max)

    def test_invalidate_cache_dt_min1(self):
        before = self.beam.dt_min
        self.beam.setup_beam(dE=np.linspace(1, 20, 10), dt=np.linspace(5, 25, 10))
        after2 = self.beam.dt_min
        self.assertNotEqual(before, after2)

    def test_invalidate_cache_dt_min2(self):
        before = self.beam.dt_min
        self.beam._dt[:] = 5
        self.assertEqual(before, self.beam.dt_min)

        self.beam.invalidate_cache_dt()
        self.assertEqual(5, self.beam.dt_min)

    def test_on_init_simulation(self):
        simulation = Mock(spec=Simulation)
        self.beam.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(spec=Simulation)
        simulation.energy_cycle.get_target_total_energy.return_value = 11
        self.beam.on_run_simulation(simulation=simulation, n_turns=10, turn_i_init=1)

    def test_plot_hist2d_executes(self):
        self.beam.plot_hist2d()
        plt.gcf().clf()

    def test_setup_beam(self):
        with self.assertRaises(AssertionError):
            self.beam.setup_beam(dE=np.ones(10), dt=np.ones(11))
        with self.assertRaises(AssertionError):
            self.beam.setup_beam(dE=np.ones(10), dt=np.ones(10), flags=np.ones(11))


class TestProbeBunch(unittest.TestCase):
    def setUp(self):
        self.probe_bunch = ProbeBeam(particle_type=proton, dt=np.ones(10))

    def test___init__1(self):
        self.probe_bunch = ProbeBeam(particle_type=proton, dt=np.ones(10))

    def test___init__2(self):
        self.probe_bunch = ProbeBeam(particle_type=proton, dE=np.ones(10))

    def test___init__3(self):
        with self.assertRaises(ValueError):
            self.probe_bunch = ProbeBeam(particle_type=proton)


class TestWeightenedBeam(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.weightened_beam = WeightenedBeam(n_particles=None, particle_type=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_from_beam(self):
        # TODO: implement test for `from_beam`
        self.weightened_beam.from_beam(beam=None)

    @unittest.skip
    def test_setup_beam(self):
        # TODO: implement test for `setup_beam`
        self.weightened_beam.setup_beam(dt=None, dE=None, flags=None, weights=None)


if __name__ == "__main__":
    unittest.main()
