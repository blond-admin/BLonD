"""Tests for the analytic 2D Hamiltonian building blocks."""

import numpy as np
import pytest

from blond.experimental.beam_preparation.analytic_hamiltonian import (
    calc_eom_factor_dE,
    hamiltonian_grid,
)

# LHC-like reference (450 GeV protons).
ETA_0 = 3.172867586042721e-04
BETA = 0.9999978262922387
TOTAL_ENERGY = 450.00104432e9  # eV, ~ proton at 450 GeV/c

EOM_FACTOR_DE = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)


def _single_bucket_well(n_time=400, amplitude=53.6):
    # Symmetric single-bucket well: maxima at both edges, minimum at the
    # centre, known potential-well amplitude.
    time = np.linspace(0.0, 2.5e-9, n_time)
    well = amplitude * np.cos(np.pi * np.arange(n_time) / (n_time - 1)) ** 2
    return time, well


def test_calc_eom_factor_dE_formula():
    factor = calc_eom_factor_dE(ETA_0, BETA, TOTAL_ENERGY)
    expected = abs(ETA_0) / (2.0 * BETA**2 * TOTAL_ENERGY)
    assert np.isclose(factor, expected)
    assert factor > 0.0
    # Below transition the kinetic factor is identical (|eta_0|).
    assert calc_eom_factor_dE(-ETA_0, BETA, TOTAL_ENERGY) == factor


def test_grid_shape_and_convention():
    time, well = _single_bucket_well(n_time=300)
    time_grid, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=EOM_FACTOR_DE, n_points_deltaE=200
    )
    assert time_grid.shape == (200, 300)
    assert deltaE_grid.shape == (200, 300)
    assert hamilton.shape == (200, 300)
    # xy convention: time along axis 1, dE along axis 0
    assert np.allclose(time_grid[0, :], time)
    assert np.allclose(time_grid[:, 0], time[0])


def test_hamiltonian_formula():
    time, well = _single_bucket_well()
    time_grid, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=EOM_FACTOR_DE
    )
    expected = EOM_FACTOR_DE * deltaE_grid**2 + well[np.newaxis, :]
    np.testing.assert_allclose(hamilton, expected)


def test_default_deltaE_frame_is_separatrix():
    time, well = _single_bucket_well(amplitude=53.6)
    _, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=EOM_FACTOR_DE
    )
    potential_well_amplitude = well.max() - well.min()
    deltaE_max = np.sqrt(potential_well_amplitude / EOM_FACTOR_DE)
    assert np.isclose(deltaE_grid.max(), deltaE_max)
    assert np.isclose(deltaE_grid.min(), -deltaE_max)
    # In the well-minimum column at dE=dE_max: H = amplitude + V.min(),
    # i.e. exactly the well maximum.
    i_min = int(well.argmin())
    assert np.isclose(EOM_FACTOR_DE * deltaE_max**2, potential_well_amplitude)
    assert np.isclose(hamilton[:, i_min].max(), well.max(), rtol=1e-9)


def test_min_of_hamiltonian_is_zero_at_center():
    # Odd n_time puts a sample exactly on the well minimum (V=0) and an
    # odd dE count includes the dE=0 row -> min H is exactly zero.
    time, well = _single_bucket_well(n_time=401)
    _, deltaE_grid, hamilton = hamiltonian_grid(
        time, well, eom_factor_dE=EOM_FACTOR_DE, n_points_deltaE=401
    )
    assert np.isclose(hamilton.min(), 0.0)


def test_custom_energy_range():
    time, well = _single_bucket_well()
    _, deltaE_grid, _ = hamiltonian_grid(
        time, well, eom_factor_dE=EOM_FACTOR_DE, energy_range=(-1e8, 1e8)
    )
    assert np.isclose(deltaE_grid.min(), -1e8)
    assert np.isclose(deltaE_grid.max(), 1e8)


def test_uncut_well_raises_with_default_energy_range():
    # Two-bucket well (interior maximum): the separatrix-based default
    # dE frame is meaningless, must raise; explicit range is allowed.
    n = 401
    time = np.linspace(0.0, 5e-9, n)
    two_buckets = 50.0 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1)) ** 2
    with pytest.raises(ValueError):
        hamiltonian_grid(time, two_buckets, eom_factor_dE=EOM_FACTOR_DE)
    hamiltonian_grid(
        time,
        two_buckets,
        eom_factor_dE=EOM_FACTOR_DE,
        energy_range=(-1e8, 1e8),
    )


def test_shape_mismatch_raises():
    time, well = _single_bucket_well()
    with pytest.raises(AssertionError):
        hamiltonian_grid(time, well[:-1], eom_factor_dE=EOM_FACTOR_DE)


def test_energy_range_decreasing_raises():
    time, well = _single_bucket_well()
    with pytest.raises(AssertionError):
        hamiltonian_grid(
            time,
            well,
            eom_factor_dE=EOM_FACTOR_DE,
            energy_range=(1e8, -1e8),
        )


def test_verbose_and_plot_smoke():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time, well = _single_bucket_well(n_time=64)
    hamiltonian_grid(
        time,
        well,
        eom_factor_dE=EOM_FACTOR_DE,
        n_points_deltaE=64,
        verbose=True,
        plot=True,
    )
    plt.close("all")
