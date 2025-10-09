import unittest


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_slice_beam(self):
        # TODO: implement test for `slice_beam`
        from blond.gpu.butils_wrap_cupy import slice_beam

        slice_beam(dt=None, profile=None, cut_left=None, cut_right=None)
