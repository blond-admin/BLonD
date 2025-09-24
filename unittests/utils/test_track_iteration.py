import unittest

from blond.utils.track_iteration import TrackIteration



class TestTrackIteration(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.track_iteration = TrackIteration(track_map=None, init_turn=None, final_turn=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test___next__(self):
        # TODO: implement test for `__next__`
        self.track_iteration.__next__()

