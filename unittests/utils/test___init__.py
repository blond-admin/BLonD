import unittest

from blond.utils import c_complex, c_real, PrecisionClass, c_complex128, c_complex64


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_c_complex(self):
        # TODO: implement test for `c_complex`
        c_complex(scalar=None)

    @unittest.skip
    def test_c_real(self):
        # TODO: implement test for `c_real`
        c_real(scalar=None)


class TestPrecisionClass(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.precision_class = PrecisionClass(_precision=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp


class Testc_complex128(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.c_complex128 = c_complex128(pycomplex=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_to_complex(self):
        # TODO: implement test for `to_complex`
        self.c_complex128.to_complex()


class Testc_complex64(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.c_complex64 = c_complex64(pycomplex=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_to_complex(self):
        # TODO: implement test for `to_complex`
        self.c_complex64.to_complex()
