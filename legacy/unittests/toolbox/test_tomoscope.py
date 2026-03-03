import unittest

from blond.legacy.blond2.toolbox.tomoscope import (
    distribution_from_tomoscope_data)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_distribution_from_tomoscope_data(self):
        # TODO: implement test for `distribution_from_tomoscope_data`
        distribution_from_tomoscope_data(
            dataDir=None,
            nPart=None,
            cutoff=None,
            seed=None,
            plotFig=None,
            saveDistr=None,
        )
