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
from blond.handle_results.helpers import callers_relative_path
from blond.testing.helpers import save_golden_file

# BLonD 2 only runs to rewrite the golden file, see resources/README.md.
REWRITE_GOLDEN_FILE = False


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


def _run_legacy_music(dt, dE, R_S, omega_R, Q, n_particles, t_rev, method):
    """Run legacy MuSiC ``method`` once; return its voltage and ``dE``."""
    from blond.legacy.blond2.impedances.music import Music as LegacyMusic

    n = len(dt)
    beam = SimpleNamespace(dt=dt.copy(), dE=dE.copy())
    legacy = LegacyMusic(beam, [R_S, omega_R, Q], n, n_particles, t_rev)
    getattr(legacy, method)()
    return {
        "induced_voltage": np.asarray(legacy.induced_voltage),
        "dE": np.asarray(beam.dE),
    }


def _run_legacy_music_multiturn(
    dt, dt2, dE, R_S, omega_R, Q, n_particles, t_rev
):
    """Run legacy ``track_py`` on ``dt``, then ``track_py_multi_turn``."""
    from blond.legacy.blond2.impedances.music import Music as LegacyMusic

    n = len(dt)
    beam = SimpleNamespace(dt=dt.copy(), dE=dE.copy())
    legacy = LegacyMusic(beam, [R_S, omega_R, Q], n, n_particles, t_rev)
    legacy.track_py()
    beam.dt = dt2.copy()
    legacy.track_py_multi_turn()
    return {
        "induced_voltage": np.asarray(legacy.induced_voltage),
        "dE": np.asarray(beam.dE),
    }


def test_music_track_single_turn_matches_legacy():
    """Single-turn python kernel reproduces legacy ``track_py``."""
    dt, dE, R_S, omega_R, Q, n_particles, t_rev, const = _setup()
    n = len(dt)

    golden_path = callers_relative_path(
        "resources/music_track_single_turn_blond2.npz", stacklevel=1
    )
    if REWRITE_GOLDEN_FILE:
        save_golden_file(
            golden_path,
            **_run_legacy_music(
                dt, dE, R_S, omega_R, Q, n_particles, t_rev, "track_py"
            ),
        )
    with np.load(golden_path) as golden:
        induced_voltage_blond2 = golden["induced_voltage"]
        dE_blond2 = golden["dE"]

    alpha, omega_bar, c1, c2, c3, c4 = _music_params(R_S, omega_R, Q)
    idx = np.argsort(dt)
    dt_s = np.ascontiguousarray(dt[idx])
    dE_s = np.ascontiguousarray(dE[idx])
    iv = np.zeros(n, dtype=backend.float)
    ap = np.array([1.0, 0.0, dt_s[-1]], dtype=backend.float)

    backend.specials.music_track(
        dt_s,
        dE_s,
        iv,
        ap,
        alpha,
        omega_bar,
        const,
        c1,
        c2,
        c3,
        c4,
        t_rev,
        False,
    )

    np.testing.assert_allclose(iv, induced_voltage_blond2, rtol=1e-12)
    np.testing.assert_allclose(dE_s, dE_blond2, rtol=1e-12)


def test_music_track_matches_bruteforce_ground_truth():
    """Kernel matches the independent O(n^2) direct wake sum.

    `track_classic` evaluates the resonator wake by summing over all pairs,
    independent of the O(n) recurrence, so this validates the *physics*
    (not just agreement with legacy's recurrence).
    """
    dt, dE, R_S, omega_R, Q, n_particles, t_rev, const = _setup(seed=42, n=200)
    n = len(dt)

    golden_path = callers_relative_path(
        "resources/music_track_bruteforce_blond2.npz", stacklevel=1
    )
    if REWRITE_GOLDEN_FILE:
        save_golden_file(
            golden_path,
            # O(n^2) brute-force reference
            **_run_legacy_music(
                dt, dE, R_S, omega_R, Q, n_particles, t_rev, "track_classic"
            ),
        )
    with np.load(golden_path) as golden:
        induced_voltage_blond2 = golden["induced_voltage"]

    alpha, omega_bar, c1, c2, c3, c4 = _music_params(R_S, omega_R, Q)
    idx = np.argsort(dt)
    dt_s = np.ascontiguousarray(dt[idx])
    dE_s = np.ascontiguousarray(dE[idx])
    iv = np.zeros(n, dtype=backend.float)
    ap = np.array([1.0, 0.0, dt_s[-1]], dtype=backend.float)

    backend.specials.music_track(
        dt_s,
        dE_s,
        iv,
        ap,
        alpha,
        omega_bar,
        const,
        c1,
        c2,
        c3,
        c4,
        t_rev,
        False,
    )

    # rtol above the O(n) vs O(n^2) round-off (recurrence accumulates over
    # n steps); abs agreement is ~14 digits on the ~1e5-1e6 V signal.
    np.testing.assert_allclose(iv, induced_voltage_blond2, rtol=1e-6)


def test_music_track_multiturn_matches_legacy():
    """Multi-turn python kernel reproduces legacy ``track_py_multi_turn``."""
    dt, dE, R_S, omega_R, Q, n_particles, t_rev, const = _setup(seed=3)
    n = len(dt)

    dt2 = (np.random.default_rng(7).random(n) * 1e-9).astype(backend.float)
    golden_path = callers_relative_path(
        "resources/music_track_multiturn_blond2.npz", stacklevel=1
    )
    if REWRITE_GOLDEN_FILE:
        save_golden_file(
            golden_path,
            **_run_legacy_music_multiturn(
                dt, dt2, dE, R_S, omega_R, Q, n_particles, t_rev
            ),
        )
    with np.load(golden_path) as golden:
        induced_voltage_blond2 = golden["induced_voltage"]
        dE_blond2 = golden["dE"]

    alpha, omega_bar, c1, c2, c3, c4 = _music_params(R_S, omega_R, Q)
    idx = np.argsort(dt)
    dt_s = np.ascontiguousarray(dt[idx])
    dE_s = np.ascontiguousarray(dE[idx])
    iv = np.zeros(n, dtype=backend.float)
    ap = np.array([1.0, 0.0, dt_s[-1]], dtype=backend.float)
    backend.specials.music_track(
        dt_s,
        dE_s,
        iv,
        ap,
        alpha,
        omega_bar,
        const,
        c1,
        c2,
        c3,
        c4,
        t_rev,
        False,
    )
    # turn 2: legacy keeps the turn-1 dE result and re-sorts by dt2,
    # carrying the running state via ``ap`` (set by the turn-1 kernel).
    idx2 = np.argsort(dt2)
    dt2_s = np.ascontiguousarray(dt2[idx2])
    dE2_s = np.ascontiguousarray(dE_s[idx2])
    iv2 = np.zeros(n, dtype=backend.float)
    backend.specials.music_track(
        dt2_s,
        dE2_s,
        iv2,
        ap,
        alpha,
        omega_bar,
        const,
        c1,
        c2,
        c3,
        c4,
        t_rev,
        True,
    )

    np.testing.assert_allclose(iv2, induced_voltage_blond2, rtol=1e-12)
    np.testing.assert_allclose(dE2_s, dE_blond2, rtol=1e-12)
