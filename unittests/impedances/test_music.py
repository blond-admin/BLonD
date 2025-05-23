import unittest

from blond.impedances.music import Music


class TestMusic(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.music = Music(
            Beam=None,
            resonator=None,
            n_macroparticles=None,
            n_particles=None,
            t_rev=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_track_classic(self):
        # TODO: implement test for `track_classic`
        self.music.track_classic()

    @unittest.skip
    def test_track_py(self):
        # TODO: implement test for `track_py`
        self.music.track_py()

    @unittest.skip
    def test_track_py_multi_turn(self):
        # TODO: implement test for `track_py_multi_turn`
        self.music.track_py_multi_turn()
