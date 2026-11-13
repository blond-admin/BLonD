import unittest

import numpy as np
import pytest

from blond.core.backends.backend import Numpy64Bit, backend
from blond.testing.backend_testing import BLonDTestCase


class TestEX_31_Multibunch_semi_empiric_matcher(BLonDTestCase):
    @classmethod
    def tearDownClass(cls):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    @pytest.mark.backend_mutation
    def test_executable_python(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")
        from blond.examples.scripts import (
            EX_31_Multibunch_semi_empiric_matcher as example,
        )

        # full script, reduced in size
        beam, matcher, t_rf = example.main(
            n_macroparticles=int(2e4),
            maxiter_intensity_effects=4,
            internal_grid_shape=(127, 127),
        )

        self.assertEqual(
            len(matcher.selected_buckets), len(example.filled_buckets)
        )
        dt = beam.read_partial_dt()
        bucket = np.rint(dt / t_rf).astype(int)
        self.assertEqual(set(bucket), set(example.filled_buckets))
        # the pilot bunch has half the intensity of the others
        n_per_bucket = {k: np.count_nonzero(bucket == k) for k in set(bucket)}
        self.assertAlmostEqual(
            n_per_bucket[example.pilot_bucket] / n_per_bucket[1],
            0.5,
            delta=0.01,
        )


if __name__ == "__main__":
    unittest.main()
