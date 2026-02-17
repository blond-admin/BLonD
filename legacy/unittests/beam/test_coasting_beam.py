import unittest

from blond.legacy.blond2.beam.coasting_beam import generate_coasting_beam


class TestFunctions(unittest.TestCase):
    @unittest.skip  # type: ignore
    def test_generate_coasting_beam(self):
        # TODO: implement test for `generate_coasting_beam`
        generate_coasting_beam(
            beam=None,
            t_start=None,
            t_stop=None,
            spread=None,
            spread_type=None,
            energy_offset=None,
            distribution=None,
            user_distribution=None,
            user_probability=None,
        )
