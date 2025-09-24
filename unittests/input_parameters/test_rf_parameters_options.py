import unittest

from blond.input_parameters.rf_parameters_options import (
    combine_rf_functions,
    RFStationOptions,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_combine_rf_functions(self):
        # TODO: implement test for `combine_rf_functions`
        combine_rf_functions(
            function_list=None, merge_type=None, resolution=None, Ring=None, main_h=None
        )


class TestRFStationOptions(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.rf_station_options = RFStationOptions(
            interpolation=None,
            smoothing=None,
            plot=None,
            figdir=None,
            figname=None,
            sampling=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_reshape_data(self):
        # TODO: implement test for `reshape_data`
        self.rf_station_options.reshape_data(
            input_data=None, n_turns=None, n_rf=None, interp_time=None, t_start=None
        )
