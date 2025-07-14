import unittest

from blond3.beam_preparation.emittance import EmittanceMatcher


class TestEmittanceMatcher(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.emittance_matcher = EmittanceMatcher(some_emittance=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_on_prepare_beam(self):
        # TODO: implement test for `on_prepare_beam`
        self.emittance_matcher.prepare_beam(simulation=None)
