"""Tests for WrapXsuite4Blond: an xsuite element used inside a BLonD Ring."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, proton  # noqa: E402
from blond.core.base import UserDefinedElement  # noqa: E402
from blond.core.beam.flags import BeamFlags  # noqa: E402
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


class _KillSlot:
    """Test guest that drops one slot to ``state=0`` to mimic an aperture cut."""

    def __init__(self, index: int):
        self._index = index

    def track(self, particles):
        particles.state[self._index] = 0


def test_wrapper_flags_lost_particles_from_xsuite():
    """Particles xsuite marks state<=0 must end up flagged LOST in the beam."""
    beam = _blond_beam([1e-9, 2e-9, 3e-9], [0.0, 0.0, 0.0], 1e9)
    wrapper = WrapXsuite4Blond(_KillSlot(index=1))
    wrapper.track(beam)

    flags = beam.read_partial_flags()
    assert flags[0] == BeamFlags.ACTIVE.value
    assert flags[1] == BeamFlags.LOST.value
    assert flags[2] == BeamFlags.ACTIVE.value


class _BumpReference:
    """Test guest that advances ``particles.energy0`` / ``beta0`` in place."""

    def __init__(self, target_energy0: float, mass0: float):
        self._target = target_energy0
        self._mass0 = mass0

    def track(self, particles):
        new_p0c = float(np.sqrt(self._target**2 - self._mass0**2))
        particles.energy0[:] = self._target
        particles.p0c[:] = new_p0c
        particles.beta0[:] = new_p0c / self._target


def test_wrapper_propagates_energy_program_to_beam_reference():
    """A guest that advances particles.energy0 must update beam.reference."""
    new_energy0 = 1.2e9
    beam = _blond_beam([0.0], [0.0], 1e9)
    wrapper = WrapXsuite4Blond(_BumpReference(new_energy0, proton.mass))
    wrapper.track(beam)

    assert beam.reference.total_energy == pytest.approx(new_energy0, rel=1e-10)
