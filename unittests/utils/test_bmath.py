import unittest

from blond.utils.bmath import get_gpu_device, use_cpu, use_gpu, use_precision


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_get_gpu_device(self):
        # TODO: implement test for `get_gpu_device`
        get_gpu_device()

    @unittest.skip
    def test_use_cpu(self):
        # TODO: implement test for `use_cpu`
        use_cpu()

    @unittest.skip
    def test_use_gpu(self):
        # TODO: implement test for `use_gpu`
        use_gpu(gpu_id=None)

    @unittest.skip
    def test_use_precision(self):
        # TODO: implement test for `use_precision`
        use_precision(_precision=None)
