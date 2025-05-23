import unittest

from blond.gpu import GPUDev


class TestGPUDev(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.gpu_dev = GPUDev()

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_func(self):
        # TODO: implement test for `func`
        self.gpu_dev.func(name=None)

    @unittest.skip
    def test_load_library(self):
        # TODO: implement test for `load_library`
        self.gpu_dev.load_library(_precision=None)

    @unittest.skip
    def test_report_attributes(self):
        # TODO: implement test for `report_attributes`
        self.gpu_dev.report_attributes()
