import unittest

from blond3.cycles.noise_generators.base import NoiseGenerator


class TestNoiseGenerator(unittest.TestCase):
    @unittest.skip("Abstract class")
    def setUp(self):
        # TODO: implement test for `__init__`
        self.noise_generator = NoiseGenerator()

    @unittest.skip("Abstract class")
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip("Abstract class")
    def test_get_noise(self):
        # TODO: implement test for `get_noise`
        self.noise_generator.get_noise(n_turns=None)
