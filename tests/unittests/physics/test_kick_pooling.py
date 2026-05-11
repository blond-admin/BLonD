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
        np.testing.assert_allclose(
            beam.dt.copy_as_numpy()[:-1],
            beam_dt_copy_as_numpy_pinned,
            rtol=1e-6 if backend.float == np.float32 else 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
