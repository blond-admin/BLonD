import unittest

import numpy as np

from blond.utils import butils_wrap_cpp, butils_wrap_python


class TestFunctions(unittest.TestCase):
    def test_rf_volt_comp_n_rf_1(self):
        np.random.seed(0)
        n_rf = 1
        voltages = 1e6 * np.ones(n_rf, dtype=float)
        omega_rf = 2 * np.pi * 400e6 * np.ones(n_rf, dtype=float)
        phi_rf = 1.5 * np.pi * np.ones(n_rf, dtype=float)
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
        n_rf = 2
        voltages = 1e6 * np.ones(n_rf, dtype=float)
        omega_rf = 2 * np.pi * 400e6 * np.ones(n_rf, dtype=float)
        phi_rf = 1.5 * np.pi * np.ones(n_rf, dtype=float)
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
