import unittest
from random import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from xpart.longitudinal.rfbucket_matching import (
    ParabolicDistribution,
    QGaussianDistribution,
    ThermalDistribution,
)

from blond import SingleHarmonicCavity
from blond.handle_results.helpers import callers_relative_path
from blond.interefaces.xsuite.beam_preparation.rfbucket_matching import (
    XsuiteRFBucketMatcher,
)
from blond.testing.simulation import ExampleSimulation01


class TestXsuiteRFBucketMatcher(unittest.TestCase):
    def setUp(self):
        self.example = ExampleSimulation01()

    def _test_something(self, voltage, phase, routine):
        simulation = self.example.simulation
        cavity = simulation.ring.elements.get_element(SingleHarmonicCavity)
        cavity.voltage = voltage
        cavity.phi_rf = phase
        zmax = simulation.ring.circumference / (2 * np.amin(cavity.harmonic))
        simulation.prepare_beam(
            beam=self.example.beam1,
            preparation_routine=XsuiteRFBucketMatcher(
                distribution_type=routine,
                sigma_z=zmax / 4,
                n_macroparticles=int(1e4),
            ),
        )

    def test_distribution_is_matched_thermal(self):
        random.seed(42)
        self._test_something(voltage=6e6, phase=0, routine=ThermalDistribution)
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            self.example.beam1._dt,
            self.example.beam1._dE,
        )

        filepath = callers_relative_path(
            "resources/hist_ThermalDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-5)

    def test_distribution_is_matched_qgaussian(self):
        random.seed(42)
        self._test_something(
            voltage=6e6, phase=0, routine=QGaussianDistribution
        )
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            self.example.beam1._dt,
            self.example.beam1._dE,
        )

        filepath = callers_relative_path(
            "resources/hist_QGaussianDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-5)

    @unittest.skip("test takes too long")
    def test_distribution_is_matched_parabolic(self):
        random.seed(42)
        self._test_something(
            voltage=6e6, phase=0, routine=ParabolicDistribution
        )
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            self.example.beam1._dt,
            self.example.beam1._dE,
        )

        filepath = callers_relative_path(
            "resources/hist_ParabolicDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
