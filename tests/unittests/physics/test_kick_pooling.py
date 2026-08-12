import unittest

import numpy as np

from blond import backend
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.experimental import PooledInterpolationKick
from blond.handle_results.helpers import callers_relative_path


class TestPooledInterpolationKick(unittest.TestCase):
    def setUp(self):
        self.pooled_kick = PooledInterpolationKick(maxsize=3)

    def test___init__(self):
        pass  # calls `setUp()`

    def test_clear_buffer(self):
        self.pooled_kick.register(
            time_axis=np.ones(1),
            voltage=np.ones(1),
        )
        self.pooled_kick.clear_buffer()

        self.assertEqual(len(self.pooled_kick._buffer_voltage), 0)
        self.assertEqual(len(self.pooled_kick._buffer_time_axis), 0)

    def test_register(self):
        for i in range(self.pooled_kick._maxsize + 1):  # intentional overflow
            self.pooled_kick.register(
                time_axis=np.ones(1) * i + 1,
                voltage=np.ones(1),
            )
        vals = [v[0] for v in self.pooled_kick._buffer_time_axis.values()]
        assert 0 not in vals

    def test__track(self):
        time_axis = backend.linspace(
            0,
            1,
            100,
        )
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=np.sin(time_axis),
        )
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=time_axis.copy(),
            reference_total_energy=1e12,
        )
        self.pooled_kick._track(beam=beam)
        beam_dt_copy_as_numpy_pinned = np.loadtxt(
            callers_relative_path(
                "resources/beam_dt_copy_as_numpy_pinned.npy", stacklevel=1
            )
        )

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            beam.dt.copy_as_numpy()[:-1],
            beam_dt_copy_as_numpy_pinned,
            rtol=1e-12,
        )

    def test_register_and_track_with_sparse_metadata(self):
        # minimal 2-bucket sparse layout, bucket 0 and 1 both filled
        time_axis = np.array([0.125, 0.375, 0.625, 0.875])
        voltage = np.array([1.0, 2.0, 3.0, 4.0])
        sparse_metadata = {
            "first_left_cut": 0.0,
            "left_cut_distance": 0.5,
            "cut_width": 0.5,
            "bins_per_profile": 2,
            "filling_pattern": np.array([True, True]),
            "bucket_index_to_memory_index": np.array([0, 2], dtype=np.int32),
        }
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=voltage,
            sparse_metadata=sparse_metadata,
        )
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=np.array([0.125]),
            reference_total_energy=1e12,
        )
        self.pooled_kick._track(beam=beam)
        self.assertNotEqual(beam.dE.copy_as_numpy()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
