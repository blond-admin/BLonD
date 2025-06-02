import unittest

from blond.utils import bmath


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_get_gpu_device(self):
        # TODO: implement test for `get_gpu_device`
        bmath.get_gpu_device()

    @unittest.skip
    def test_use_cpu(self):
        # TODO: implement test for `use_cpu`
        bmath.use_cpu()

    @unittest.skip
    def test_use_gpu(self):
        # TODO: implement test for `use_gpu`
        bmath.use_gpu(gpu_id=None)

    @unittest.skip
    def test_use_precision(self):
        # TODO: implement test for `use_precision`
        bmath.use_precision(_precision=None)
