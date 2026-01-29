import unittest

from blond.llrf.transfer_function import tf_estimate, TransferFunction


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_tf_estimate(self):
        # TODO: implement test for `tf_estimate`
        tf_estimate(x=None, y=None)


class TestTransferFunction(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.transfer_function = TransferFunction(
            signal_in=None, signal_out=None, T_s=None, plot=None
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_analyse(self):
        # TODO: implement test for `analyse`
        self.transfer_function.analyse(data_cut=None)

    @unittest.skip
    def test_estimate_transfer_function(self):
        # TODO: implement test for `estimate_transfer_function`
        self.transfer_function.estimate_transfer_function(
            input=None, output=None, T_s=None, plot=None, logger=None
        )

    @unittest.skip
    def test_input_signal_spectrum(self):
        # TODO: implement test for `input_signal_spectrum`
        self.transfer_function.input_signal_spectrum(
            input=None, n=None, f_s=None, plot=None
        )

    @unittest.skip
    def test_plot_magnitude_and_phase(self):
        # TODO: implement test for `plot_magnitude_and_phase`
        self.transfer_function.plot_magnitude_and_phase(freq=None, H=None)
