import unittest

import numpy as np

from blond.utils import butils_wrap_cpp, butils_wrap_python


class TestFunctions(unittest.TestCase):
    def test_rf_volt_comp_n_rf_1(self):
        np.random.seed(0)
        voltages = np.random.randn(1)
        omega_rf = np.random.randn(1)
        phi_rf = np.random.randn(1)
        bin_centers = np.linspace(1e-5, 1e-6, 64)
        actual = butils_wrap_cpp.rf_volt_comp(
            voltages=voltages, omega_rf=omega_rf, phi_rf=phi_rf, bin_centers=bin_centers
        )
        desired = butils_wrap_python.rf_volt_comp(
            voltages=voltages, omega_rf=omega_rf, phi_rf=phi_rf, bin_centers=bin_centers
        )
        np.testing.assert_allclose(actual, desired, atol=1e-12)

    def test_rf_volt_comp_n_rf_2(self):
        np.random.seed(0)
        voltages = np.random.randn(2)
        omega_rf = np.random.randn(2)
        phi_rf = np.random.randn(2)
        bin_centers = np.linspace(1e-5, 1e-6, 64)
        actual = butils_wrap_cpp.rf_volt_comp(
            voltages=voltages, omega_rf=omega_rf, phi_rf=phi_rf, bin_centers=bin_centers
        )
        desired = butils_wrap_python.rf_volt_comp(
            voltages=voltages, omega_rf=omega_rf, phi_rf=phi_rf, bin_centers=bin_centers
        )
        np.testing.assert_allclose(actual, desired, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
