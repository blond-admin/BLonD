import unittest
import unittest.mock
from random import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from scipy.constants import c as c_light

from blond import DriftSimple, SingleHarmonicRFStation
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.handle_results.helpers import callers_relative_path
from blond.testing.simulation import ExampleSimulation01


class TestXsuiteRFBucketMatcher(unittest.TestCase):
    def setUp(self):
        self.example = ExampleSimulation01()

    def _execute_test(self, voltage, phase, routine):
        try:
            from blond.interfaces.xsuite.beam_preparation.rfbucket_matching import (
                XsuiteRFBucketMatcher,
            )
        except ImportError:
            self.skipTest("xpart or xsuite interface not installed")

        simulation = self.example.simulation
        cavity = simulation.ring.elements.get_element(SingleHarmonicRFStation)
        cavity.voltage = voltage
        cavity.phi_rf_design = phase
        zmax = simulation.ring.circumference / (2 * np.amin(cavity.harmonic))
        simulation.prepare_beam(
            beam=self.example.beam1,
            preparation_routine=XsuiteRFBucketMatcher(
                distribution_type=routine,
                sigma_z=zmax / 4,
                n_macroparticles=int(1e4),
                seed=42,
            ),
        )

        drift = self.example.simulation.ring.elements.get_element(DriftSimple)
        drift.momentum_compaction_factor = None

        with self.assertRaises(ValueError):
            simulation.prepare_beam(
                beam=self.example.beam1,
                preparation_routine=XsuiteRFBucketMatcher(
                    distribution_type=routine,
                    sigma_z=zmax / 4,
                    n_macroparticles=int(1e1),
                    seed=42,
                ),
            )

    def test_distribution_is_matched_thermal(self):
        try:
            from xpart.longitudinal.rfbucket_matching import (
                ThermalDistribution,
            )
        except ImportError:
            self.skipTest("xpart not installed")

        random.seed(42)
        self._execute_test(voltage=6e6, phase=0, routine=ThermalDistribution)
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            copy_to_cpu(self.example.beam1.read_partial_dt()),
            copy_to_cpu(self.example.beam1.read_partial_dE()),
        )

        filepath = callers_relative_path(
            "resources/hist_ThermalDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-12)

    def test_distribution_is_matched_qgaussian(self):
        try:
            from xpart.longitudinal.rfbucket_matching import (
                QGaussianDistribution,
            )
        except ImportError:
            self.skipTest("xpart not installed")

        random.seed(42)
        self._execute_test(voltage=6e6, phase=0, routine=QGaussianDistribution)
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            copy_to_cpu(self.example.beam1.read_partial_dt()),
            copy_to_cpu(self.example.beam1.read_partial_dE()),
        )

        filepath = callers_relative_path(
            "resources/hist_QGaussianDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-12)

    @unittest.skip("test takes too long")
    def test_distribution_is_matched_parabolic(self):
        try:
            from xpart.longitudinal.rfbucket_matching import (
                ParabolicDistribution,
            )
        except ImportError:
            self.skipTest("xpart not installed")

        random.seed(42)
        self._execute_test(voltage=6e6, phase=0, routine=ParabolicDistribution)
        DEV_PLOT = False
        if DEV_PLOT:
            self.example.beam1.plot_hist2d()
            plt.show()

        counts, _, _, image = plt.hist2d(
            copy_to_cpu(self.example.beam1._dt.array_local),
            copy_to_cpu(self.example.beam1._dE.array_local),
        )

        filepath = callers_relative_path(
            "resources/hist_ParabolicDistribution.txt", stacklevel=1
        )
        expected_counts = np.loadtxt(filepath)
        np.testing.assert_allclose(expected_counts, counts, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()


class TestXsuiteCoordinateConversion(unittest.TestCase):
    """The (zeta, delta) -> (dt, dE) conversion of `XsuiteRFBucketMatcher`.

    XSuite generates the distribution in the PyHEADTAIL bucket coordinates
    ``zeta = -beta * c * dt`` [m] and ``delta = dp / p0`` [], both of which
    involve the reference ``beta``. A test at LHC energy cannot see whether
    it is accounted for, because there ``beta = 0.999998``; this one runs at
    ``beta ~ 0.73`` so that a missing factor is unmissable.
    """

    MOMENTUM = 1e9  # [eV/c], proton, giving beta ~ 0.73

    def _low_energy_simulation(self):
        from blond import (
            Beam,
            MagneticCyclePerTurn,
            Ring,
            Simulation,
            momentum_compaction_factor,
            proton,
        )

        ring = Ring(circumference=157.08)
        rf_station = SingleHarmonicRFStation()
        rf_station.harmonic = 1
        rf_station.voltage = 8e3
        rf_station.phi_rf_design = 0

        energy_cycle = MagneticCyclePerTurn(
            value_init=self.MOMENTUM,
            values_after_turn=np.linspace(self.MOMENTUM, self.MOMENTUM, 5),
            reference_particle=proton,
            in_unit="momentum",
        )
        drift1 = DriftSimple(orbit_length=157.08)
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=4.0
        )
        beam = Beam(intensity=1e11, particle_type=proton)
        simulation = Simulation.from_locals(locals())
        return simulation, beam, rf_station

    def test_dt_and_dE_account_for_the_reference_beta(self):
        try:
            from xpart.longitudinal import rfbucket_matching

            from blond.interfaces.xsuite.beam_preparation.rfbucket_matching import (  # noqa: E501
                XsuiteRFBucketMatcher,
            )
        except ImportError:
            self.skipTest("xpart or xsuite interface not installed")

        # On the scale of the h = 1 bucket, so that the beta in
        # zeta -> dt is not swamped by the half-period offset.
        zeta = np.array([-39.27, 0.0, 39.27])
        delta = np.array([-1e-3, 0.0, 1e-3])

        simulation, beam, rf_station = self._low_energy_simulation()
        with unittest.mock.patch.object(
            rfbucket_matching, "RFBucketMatcher"
        ) as matcher:
            matcher.return_value.generate.return_value = (zeta, delta)
            simulation.prepare_beam(
                beam=beam,
                preparation_routine=XsuiteRFBucketMatcher(
                    distribution_type=None,
                    sigma_z=1.0,
                    n_macroparticles=len(zeta),
                    seed=42,
                ),
            )

        beta = beam.reference.beta
        total_energy = beam.reference.total_energy
        self.assertLess(beta, 0.8, "the test must run away from beta = 1")

        omega_rf = rf_station.calc_omega_rf_design(
            beam_beta=beta,
            ring_circumference=simulation.ring.circumference,
        )
        rf_period = 2 * np.pi / omega_rf

        # zeta = -beta * c * dt, i.e. dt = -zeta / (beta * c)
        expected_dt = -zeta / (beta * c_light) + rf_period / 2
        # dE = beta * c * dp = beta * p0c * delta = beta**2 * E0 * delta
        expected_dE = delta * beta**2 * total_energy

        np.testing.assert_allclose(
            copy_to_cpu(beam.read_partial_dt()), expected_dt, rtol=1e-12
        )
        np.testing.assert_allclose(
            copy_to_cpu(beam.read_partial_dE()), expected_dE, rtol=1e-12
        )
