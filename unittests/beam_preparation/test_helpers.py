import unittest

import numpy as np

from blond import Beam, make_multibunch_beam, uranium_29


class TestCallables(unittest.TestCase):
    def test_make_multibunch_beam_fails(self):
        beam = Beam(
            intensity=1, particle_type=uranium_29, is_counter_rotating=False
        )
        with self.assertRaisesRegex(AssertionError, "Please set up beam"):
            make_multibunch_beam(
                beam=beam,
                n_times=3,
                t_distance=222,
                common_offset=111,
            )

    def test_make_multibunch_beam_executes(self):
        beam = Beam(
            intensity=1, particle_type=uranium_29, is_counter_rotating=False
        )
        beam.setup_beam(dt=[1, 2, 3], dE=[1e3, 2e3, 3e3])
        beam = make_multibunch_beam(
            beam=beam,
            n_times=3,
            t_distance=222,
            common_offset=111,
        )

        np.testing.assert_allclose(
            beam._dt.array_local,
            [
                112.0,  # dt[0] + common_offset
                334.0,  # dt[0] + common_offset + t_distance
                556.0,  # dt[0] + common_offset + 2 * t_distance
                113.0,
                335.0,
                557.0,
                114.0,
                336.0,
                558.0,
            ],
        )
        np.testing.assert_allclose(
            beam._dE.array_local,
            [
                1000.0,  # dE[0]
                1000.0,  # dE[0]
                1000.0,  # dE[0]
                2000.0,
                2000.0,
                2000.0,
                3000.0,
                3000.0,
                3000.0,
            ],
        )
        self.assertEqual(beam.particle_type, uranium_29)
        self.assertEqual(beam.is_counter_rotating, False)


if __name__ == "__main__":
    unittest.main()
