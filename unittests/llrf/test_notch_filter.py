import unittest

from blond.llrf.notch_filter import impedance_notches


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_impedance_notches(self):
        # TODO: implement test for `impedance_notches`
        impedance_notches(
            f_rev=None,
            frequencies=None,
            imp_source=None,
            list_harmonics=None,
            list_width_depth=None,
        )
