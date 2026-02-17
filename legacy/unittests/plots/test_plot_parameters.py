import unittest

from blond.plots.plot_parameters import plot_voltage_programme


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_plot_voltage_programme(self):
        # TODO: implement test for `plot_voltage_programme`
        plot_voltage_programme(
            time=None, voltage=None, sampling=None, dirname=None, figno=None
        )
