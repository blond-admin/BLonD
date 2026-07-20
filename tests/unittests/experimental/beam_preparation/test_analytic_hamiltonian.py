"""Tests for the analytic 2D Hamiltonian building blocks."""

import numpy as np

from blond.experimental.beam_preparation.analytic_hamiltonian import (
    eom_factor_dE,
    hamiltonian_grid,
)

# LHC-like reference (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9  # eV, ~ proton at 450 GeV/c


def _parabolic_well(n_time=400, barrier=53.6):
    # A simple symmetric well with a known barrier height, min at 0.
    time = np.linspace(0.0, 2.5e-9, n_time)
    well = barrier * (np.sin(np.pi * np.arange(n_time) / (n_time - 1))) ** 2
    return time, well


def test_eom_factor_dE_formula():
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    expected = abs(ETA_0) / (2.0 * BETA**2 * TOTAL_ENERGY)
    assert np.isclose(factor, expected)
    assert factor > 0.0


def test_grid_shape_and_convention():
    time, well = _parabolic_well(n_time=300)
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    time_grid, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=factor, n_points_deltaE=200
    )
    assert time_grid.shape == (200, 300)
    assert deltaE_grid.shape == (200, 300)
    assert hamilton.shape == (200, 300)
    # xy convention: time along axis 1, dE along axis 0
    assert np.allclose(time_grid[0, :], time)
    assert np.allclose(time_grid[:, 0], time[0])


def test_hamiltonian_formula():
    time, well = _parabolic_well()
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    time_grid, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=factor
    )
    expected = factor * deltaE_grid**2 + well[np.newaxis, :]
    np.testing.assert_allclose(hamilton, expected)


def test_default_deltaE_frame_is_separatrix():
    time, well = _parabolic_well(barrier=53.6)
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    _, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=factor
    )
    barrier = well.max() - well.min()
    deltaE_max = np.sqrt(barrier / factor)
    assert np.isclose(deltaE_grid.max(), deltaE_max)
    assert np.isclose(deltaE_grid.min(), -deltaE_max)
    # At the well minimum (V=0) and dE=dE_max, H equals the barrier height.
    i_min = int(well.argmin())
    assert np.isclose(factor * deltaE_max**2, barrier)
    assert np.isclose(hamilton[:, i_min].max(), barrier, rtol=1e-6)


def test_min_of_hamiltonian_is_zero_at_center():
    time, well = _parabolic_well()
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    _, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=factor, n_points_deltaE=401
    )
    # dE grid is symmetric with a zero row -> min H sits at (dE=0, V=0).
    assert np.isclose(hamilton.min(), 0.0)


def test_custom_energy_range():
    time, well = _parabolic_well()
    factor = eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    _, deltaE_grid, _ = hamiltonian_grid(
        time, well, eom_factor_dE=factor, energy_range=(-1e8, 1e8)
    )
    assert np.isclose(deltaE_grid.min(), -1e8)
    assert np.isclose(deltaE_grid.max(), 1e8)
