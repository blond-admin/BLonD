"""Tests for WrapXsuite4Blond: an xsuite element used inside a BLonD Ring."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, proton  # noqa: E402
from blond.core.base import UserDefinedElement  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_xsuite_elelemt import (  # noqa: E402
    WrapXsuite4Blond,
)


def _proton_p0c(total_energy):
    return float(np.sqrt(total_energy**2 - proton.mass**2))


def _blond_beam(dt, dE, total_energy):
    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.asarray(dt, dtype=float).copy(),
        dE=np.asarray(dE, dtype=float).copy(),
        reference_total_energy=total_energy,
    )
    return beam


def test_wrapper_is_a_blond_element():
    wrapper = WrapXsuite4Blond(xt.Drift(length=1.0))
    assert isinstance(wrapper, UserDefinedElement)


def test_wrapper_drift_matches_direct_xsuite_drift():
    energy0 = 1e9
    p0c = _proton_p0c(energy0)
    beta0 = p0c / energy0
    L = 1.0

    dt = np.array([-1e-9, 0.0, 2e-9])
    dE = np.array([-1e6, 0.0, 5e6])

    # 1. BLonD beam tracked via wrapper
    beam = _blond_beam(dt, dE, energy0)
    wrapper = WrapXsuite4Blond(xt.Drift(length=L))
    wrapper.track(beam)

    # 2. Direct xsuite drift on equivalent particles
    particles = xt.Particles(
        p0c=p0c,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=-dt * beta0 * c,
        ptau=dE / (beta0 * energy0),
    )
    xt.Drift(length=L).track(particles)
    expected_dt = -np.asarray(particles.zeta) / (beta0 * c)
    expected_dE = np.asarray(particles.ptau) * beta0 * energy0

    np.testing.assert_allclose(
        beam.read_partial_dt(), expected_dt, rtol=1e-10
    )
    np.testing.assert_allclose(
        beam.read_partial_dE(), expected_dE, rtol=1e-10
    )


def test_wrapper_zero_offset_particles_unchanged_by_drift():
    """A synchronous particle (dE=0) through any drift should keep dt=0."""
    energy0 = 1e9
    beam = _blond_beam([0.0], [0.0], energy0)
    wrapper = WrapXsuite4Blond(xt.Drift(length=2.5))
    wrapper.track(beam)
    np.testing.assert_allclose(beam.read_partial_dt(), [0.0], atol=1e-15)
    np.testing.assert_allclose(beam.read_partial_dE(), [0.0], atol=1e-15)
