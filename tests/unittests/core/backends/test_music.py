# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Correctness of the MuSiC kernel against the independent BLonD2 oracle.

Cross-backend agreement (python vs cpp) and the ``numba``/``cuda``
NotImplementedError are covered by ``test_backend.py::TestSpecials``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import elementary_charge as e

from blond.core.backends.backend import Numpy64Bit, backend
from blond.legacy.blond2.impedances.music import Music as LegacyMusic


@pytest.fixture(autouse=True)
def _numpy_backend():
    """Pin the numpy/python backend and restore state afterwards."""
    backend_org = backend.__class__
    specials_org = backend.specials_mode
    backend.change_backend(Numpy64Bit)
    backend.set_specials("python")
    yield
    backend.change_backend(backend_org)
    backend.set_specials(specials_org)


def _music_params(R_S, omega_R, Q):
    """Return (alpha, omega_bar, coeff1..4) for a resonator."""
    alpha = omega_R / (2 * Q)
    omega_bar = np.sqrt(omega_R**2 - alpha**2)
    coeff1 = -alpha / omega_bar
    coeff2 = -R_S * omega_R / (Q * omega_bar)
    coeff3 = omega_R * Q / (R_S * omega_bar)
    coeff4 = alpha / omega_bar
    return alpha, omega_bar, coeff1, coeff2, coeff3, coeff4


def _setup(seed=0, n=16):
    rng = np.random.default_rng(seed)
    dt = (rng.random(n) * 1e-9).astype(backend.float)
    dE = (rng.standard_normal(n) * 1e6).astype(backend.float)
    R_S, omega_R, Q = 1e6, 2 * np.pi * 1e9, 1.0
    n_particles, t_rev = 1e11, 2e-6
    const = -e * R_S * omega_R * n_particles / (n * Q)
    return dt, dE, R_S, omega_R, Q, n_particles, t_rev, const


def test_music_track_single_turn_matches_legacy():
    """Single-turn python kernel reproduces legacy ``track_py``."""
    dt, dE, R_S, omega_R, Q, n_particles, t_rev, const = _setup()
    n = len(dt)

    beam = SimpleNamespace(dt=dt.copy(), dE=dE.copy())
    legacy = LegacyMusic(beam, [R_S, omega_R, Q], n, n_particles, t_rev)
    legacy.track_py()

    alpha, omega_bar, c1, c2, c3, c4 = _music_params(R_S, omega_R, Q)
    idx = np.argsort(dt)
    dt_s = np.ascontiguousarray(dt[idx])
    dE_s = np.ascontiguousarray(dE[idx])
    iv = np.zeros(n, dtype=backend.float)
    ap = np.array([1.0, 0.0, t_rev, dt_s[-1]], dtype=backend.float)

    backend.specials.music_track(
        dt_s, dE_s, iv, ap, alpha, omega_bar, const, c1, c2, c3, c4, False
    )

    np.testing.assert_allclose(iv, legacy.induced_voltage, rtol=1e-12)
    np.testing.assert_allclose(dE_s, beam.dE, rtol=1e-12)


def test_music_track_multiturn_matches_legacy():
    """Multi-turn python kernel reproduces legacy ``track_py_multi_turn``."""
    dt, dE, R_S, omega_R, Q, n_particles, t_rev, const = _setup(seed=3)
    n = len(dt)

    beam = SimpleNamespace(dt=dt.copy(), dE=dE.copy())
    legacy = LegacyMusic(beam, [R_S, omega_R, Q], n, n_particles, t_rev)
    legacy.track_py()
    dt2 = (np.random.default_rng(7).random(n) * 1e-9).astype(backend.float)
    beam.dt = dt2.copy()
    legacy.track_py_multi_turn()

    alpha, omega_bar, c1, c2, c3, c4 = _music_params(R_S, omega_R, Q)
    idx = np.argsort(dt)
    dt_s = np.ascontiguousarray(dt[idx])
    dE_s = np.ascontiguousarray(dE[idx])
    iv = np.zeros(n, dtype=backend.float)
    ap = np.array([1.0, 0.0, t_rev, dt_s[-1]], dtype=backend.float)
    backend.specials.music_track(
        dt_s, dE_s, iv, ap, alpha, omega_bar, const, c1, c2, c3, c4, False
    )
    # turn 2: legacy keeps the turn-1 dE result and re-sorts by dt2,
    # carrying the running state via ``ap`` (set by the turn-1 kernel).
    idx2 = np.argsort(dt2)
    dt2_s = np.ascontiguousarray(dt2[idx2])
    dE2_s = np.ascontiguousarray(dE_s[idx2])
    iv2 = np.zeros(n, dtype=backend.float)
    ap[2] = t_rev
    backend.specials.music_track(
        dt2_s, dE2_s, iv2, ap, alpha, omega_bar, const, c1, c2, c3, c4, True
    )

    np.testing.assert_allclose(iv2, legacy.induced_voltage, rtol=1e-12)
    np.testing.assert_allclose(dE2_s, beam.dE, rtol=1e-12)
