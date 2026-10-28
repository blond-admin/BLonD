"""Unit tests for the BLonD<->xsuite coordinate conversion helpers."""

import numpy as np
from scipy.constants import c

from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)


def _frame():
    return ReferenceFrame(beta0=0.999, energy0=450e9)


def test_zeta_to_dt_matches_formula():
    frame = _frame()
    zeta = np.array([-0.3, 0.0, 0.5])
    expected = -zeta / (frame.beta0 * c)
    np.testing.assert_allclose(zeta_to_dt(zeta, frame), expected, rtol=1e-15)


def test_dt_to_zeta_matches_formula():
    frame = _frame()
    dt = np.array([-1e-9, 0.0, 2e-9])
    expected = -dt * frame.beta0 * c
    np.testing.assert_allclose(dt_to_zeta(dt, frame), expected, rtol=1e-15)


def test_ptau_to_dE_matches_formula():
    frame = _frame()
    ptau = np.array([-1e-4, 0.0, 3e-4])
    expected = ptau * frame.beta0 * frame.energy0
    np.testing.assert_allclose(ptau_to_dE(ptau, frame), expected, rtol=1e-15)


def test_dE_to_ptau_matches_formula():
    frame = _frame()
    dE = np.array([-1e6, 0.0, 5e6])
    expected = dE / (frame.beta0 * frame.energy0)
    np.testing.assert_allclose(dE_to_ptau(dE, frame), expected, rtol=1e-15)


def test_zeta_dt_round_trip():
    frame = _frame()
    zeta = np.array([-0.7, -0.1, 0.0, 0.2, 0.9])
    np.testing.assert_allclose(
        dt_to_zeta(zeta_to_dt(zeta, frame), frame), zeta, rtol=1e-12
    )


def test_ptau_dE_round_trip():
    frame = _frame()
    ptau = np.array([-2e-4, 0.0, 1e-4, 4e-4])
    np.testing.assert_allclose(
        dE_to_ptau(ptau_to_dE(ptau, frame), frame), ptau, rtol=1e-12
    )
