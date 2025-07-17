import unittest


class TestFunctions(unittest.TestCase):
    def test_my_callback(self):
        from blond3.examples import EX_01_Acceleration  # NOQA will run the
        # full script. just checking if it crashes
