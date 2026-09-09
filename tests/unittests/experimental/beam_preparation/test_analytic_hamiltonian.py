"""Tests for the analytic 2D Hamiltonian building blocks."""

import unittest

import numpy as np
import pytest

from blond.core.backends.backend import backend
from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import multi_backend_testcase

# LHC-like reference (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9  # eV, ~ proton at 450 GeV/c

EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)


class TestCalcEomFactorDE(unittest.TestCase):
    def test_formula(self):
        factor = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
        expected = abs(ETA_0) / (2.0 * BETA**2 * TOTAL_ENERGY)
        np.testing.assert_allclose(factor, expected)
        self.assertGreater(factor, 0.0)

    def test_below_transition_gives_same_factor(self):
        # Below transition the kinetic factor is identical (|eta_0|).
        self.assertEqual(
            calc_eom_factor_dE(-ETA_0, BETA, TOTAL_ENERGY),
            calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY),
        )


class TestHamiltonianGrid(unittest.TestCase):
    @staticmethod
    def _single_bucket_well(n_time=400, amplitude=53.6):
        # Symmetric single-bucket well: maxima at both edges, minimum at
        # the centre, known potential-well amplitude.
        time = backend.linspace(0.0, 2.5e-9, n_time, dtype=backend.float)
        well = (
            amplitude
            * backend.cos(
                np.pi
                * backend.arange(n_time, dtype=backend.float)
                / (n_time - 1)
            )
            ** 2
        )
        return time, well

    @multi_backend_testcase
    @pytest.mark.backend_mutation
    def test_grid_shape_and_convention(self):
        time, well = self._single_bucket_well(n_time=300)
        time_grid, deltaE_grid, hamilton = hamiltonian_grid(
            time, well, eom_factor_dE=EOM_FACTOR_DE, n_points_deltaE=200
        )
        self.assertEqual(time_grid.shape, (200, 300))
        self.assertEqual(deltaE_grid.shape, (200, 300))
        self.assertEqual(hamilton.shape, (200, 300))
        # xy convention: time along axis 1, dE along axis 0
        np.testing.assert_allclose(
            copy_to_cpu(time_grid[0, :]), copy_to_cpu(time)
        )
        np.testing.assert_allclose(
            copy_to_cpu(time_grid[:, 0]), copy_to_cpu(time[0])
        )

    def test_hamiltonian_formula(self):
        time, well = self._single_bucket_well()
        _, deltaE_grid, hamilton = hamiltonian_grid(
            time, well, eom_factor_dE=EOM_FACTOR_DE
        )
        expected = EOM_FACTOR_DE * deltaE_grid**2 + well[np.newaxis, :]
        np.testing.assert_allclose(
            copy_to_cpu(hamilton), copy_to_cpu(expected)
        )

    def test_default_deltaE_frame_is_separatrix(self):
        time, well = self._single_bucket_well(amplitude=53.6)
        _, deltaE_grid, hamilton = hamiltonian_grid(
            time, well, eom_factor_dE=EOM_FACTOR_DE
        )
        potential_well_amplitude = float(well.max() - well.min())
        deltaE_max = np.sqrt(potential_well_amplitude / EOM_FACTOR_DE)
        np.testing.assert_allclose(float(deltaE_grid.max()), deltaE_max)
        np.testing.assert_allclose(float(deltaE_grid.min()), -deltaE_max)
        # In the well-minimum column at dE=dE_max: H = amplitude + V.min(),
        # i.e. exactly the well maximum.
        i_min = int(well.argmin())
        np.testing.assert_allclose(
            EOM_FACTOR_DE * deltaE_max**2, potential_well_amplitude
        )
        np.testing.assert_allclose(
            float(hamilton[:, i_min].max()), float(well.max()), rtol=1e-9
        )

    def test_min_of_hamiltonian_is_zero_at_center(self):
        # Odd n_time puts a sample exactly on the well minimum (V=0) and
        # an odd dE count includes the dE=0 row -> min H is exactly zero.
        time, well = self._single_bucket_well(n_time=401)
        _, _, hamilton = hamiltonian_grid(
            time, well, eom_factor_dE=EOM_FACTOR_DE, n_points_deltaE=401
        )
        np.testing.assert_allclose(float(hamilton.min()), 0.0, atol=1e-12)

    def test_custom_energy_range(self):
        time, well = self._single_bucket_well()
        _, deltaE_grid, _ = hamiltonian_grid(
            time, well, eom_factor_dE=EOM_FACTOR_DE, energy_range=(-1e8, 1e8)
        )
        np.testing.assert_allclose(float(deltaE_grid.min()), -1e8)
        np.testing.assert_allclose(float(deltaE_grid.max()), 1e8)

    def test_uncut_well_raises_with_default_energy_range(self):
        # Two-bucket well (interior maximum): the separatrix-based
        # default dE frame is meaningless, must raise.
        n = 401
        time = backend.linspace(0.0, 5e-9, n, dtype=backend.float)
        two_buckets = (
            50.0
            * backend.cos(
                2.0 * np.pi * backend.arange(n, dtype=backend.float) / (n - 1)
            )
            ** 2
        )
        with self.assertRaises(ValueError):
            hamiltonian_grid(time, two_buckets, eom_factor_dE=EOM_FACTOR_DE)

    def test_uncut_well_accepts_explicit_energy_range(self):
        n = 401
        time = backend.linspace(0.0, 5e-9, n, dtype=backend.float)
        two_buckets = (
            50.0
            * backend.cos(
                2.0 * np.pi * backend.arange(n, dtype=backend.float) / (n - 1)
            )
            ** 2
        )
        hamiltonian_grid(
            time,
            two_buckets,
            eom_factor_dE=EOM_FACTOR_DE,
            energy_range=(-1e8, 1e8),
        )

    def test_shape_mismatch_raises(self):
        time, well = self._single_bucket_well()
        with self.assertRaises(AssertionError):
            hamiltonian_grid(time, well[:-1], eom_factor_dE=EOM_FACTOR_DE)

    def test_energy_range_decreasing_raises(self):
        time, well = self._single_bucket_well()
        with self.assertRaises(AssertionError):
            hamiltonian_grid(
                time,
                well,
                eom_factor_dE=EOM_FACTOR_DE,
                energy_range=(1e8, -1e8),
            )

    def test_verbose_and_plot_smoke(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        time, well = self._single_bucket_well(n_time=64)
        hamiltonian_grid(
            time,
            well,
            eom_factor_dE=EOM_FACTOR_DE,
            n_points_deltaE=64,
            verbose=True,
            plot=True,
        )
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
