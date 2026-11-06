"""Cross-check the coordinate helpers against xsuite's own physics.

The formula tests in ``test_helpers.py`` compare each helper to the same
expression it implements, so they only guard against edits to the code -- not
against the conversion being physically wrong. These tests instead use a real
``xtrack.Particles`` as an independent oracle: xsuite computes ``energy`` and
``beta0`` from ``ptau``/``p0c`` with its own internal definitions
(https://xsuite.readthedocs.io/), and we assert our conversions reproduce them.
"""

import math

import numpy as np
import pytest
from scipy.constants import c

pytest.importorskip("xtrack")

from xtrack import Particles  # noqa: E402

from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    dE_to_ptau,
    ptau_to_dE,
    zeta_to_dt,
)

# Proton at 2 GeV total energy; a regime where beta0 differs clearly from 1
# so a missing beta0 factor would show up.
_MASS = 938.272e6
_ENERGY0 = 2.0e9


def _particles(zeta, ptau):
    p0c = math.sqrt(_ENERGY0**2 - _MASS**2)
    return Particles(
        p0c=p0c,
        mass0=_MASS,
        q0=1.0,
        zeta=np.asarray(zeta, dtype=float),
        ptau=np.asarray(ptau, dtype=float),
    )


def test_ptau_to_dE_matches_xsuite_energy():
    # xsuite's per-particle total energy is the independent oracle: the energy
    # deviation dE it implies must equal ptau_to_dE(ptau).
    ptau = np.array([-2e-3, -1e-4, 0.0, 5e-4, 3e-3])
    particles = _particles(np.zeros_like(ptau), ptau)
    frame = ReferenceFrame(beta0=float(particles.beta0[0]), energy0=_ENERGY0)

    dE_from_xsuite = np.asarray(particles.energy) - _ENERGY0
    np.testing.assert_allclose(
        ptau_to_dE(ptau, frame), dE_from_xsuite, rtol=1e-9
    )


def test_dE_to_ptau_inverts_xsuite_energy():
    # Round-trip through xsuite: feed a known dE in, and recover the ptau that
    # xsuite assigned to a particle carrying exactly that energy deviation.
    dE = np.array([-3e6, -1e5, 0.0, 2e6])
    frame_beta0 = math.sqrt(_ENERGY0**2 - _MASS**2) / _ENERGY0
    frame = ReferenceFrame(beta0=frame_beta0, energy0=_ENERGY0)

    ptau = dE_to_ptau(dE, frame)
    particles = _particles(np.zeros_like(dE), ptau)
    np.testing.assert_allclose(
        np.asarray(particles.energy) - _ENERGY0, dE, rtol=1e-9
    )


def test_zeta_to_dt_uses_xsuite_beta0():
    # zeta -> dt is a pure time-of-flight: a particle at longitudinal offset
    # zeta arrives dt = -zeta / (beta0 c) relative to the reference. Anchor
    # beta0 to xsuite's own value so this is not a restatement of the formula.
    zeta = np.array([-0.1, -0.01, 0.0, 0.05, 0.2])
    particles = _particles(zeta, np.zeros_like(zeta))
    beta0 = float(particles.beta0[0])
    frame = ReferenceFrame(beta0=beta0, energy0=_ENERGY0)

    expected_dt = -zeta / (beta0 * c)
    np.testing.assert_allclose(
        zeta_to_dt(zeta, frame), expected_dt, rtol=1e-12
    )
