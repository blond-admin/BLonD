import unittest

from blond.testing.backend_testing import BLonDTestCase


class TestBeamPreparationRoutine(BLonDTestCase):
    @unittest.skip("Abstract class")
    def test_prepare_beam(self):
        # TODO: implement test for `prepare_beam`
        self.beam_preparation_routine.prepare_beam(simulation=None)
