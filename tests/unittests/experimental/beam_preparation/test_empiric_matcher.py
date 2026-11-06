import unittest

import numpy as np

from blond.experimental.beam_preparation.empiric_matcher import EmpiricMatcher
from blond.testing.backend_testing import BLonDTestCase


def _make_matcher(**kwargs) -> EmpiricMatcher:
    return EmpiricMatcher(
        grid_base_dt=np.linspace(0, 2.5e-9, 10),
        grid_base_dE=np.linspace(-1e9, 1e9, 10),
        n_macroparticles=100,
        **kwargs,
    )


class TestEmpiricMatcher(BLonDTestCase):
    def test_animate_pause_time_defaults_to_previous_value(self):
        self.assertEqual(_make_matcher().animate_pause_time, 0.1)

    def test_animate_pause_time_can_be_set(self):
        matcher = _make_matcher(animate_pause_time=0.0)
        self.assertEqual(matcher.animate_pause_time, 0.0)


if __name__ == "__main__":
    unittest.main()
