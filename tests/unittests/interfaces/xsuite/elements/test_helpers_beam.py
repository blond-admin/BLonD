"""Tests for beam<->particle converters (require xsuite)."""

import unittest

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")

from blond import Beam, copy_to_cpu, proton  # noqa: E402
from blond.core.beam.flags import BeamFlags  # noqa: E402
from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    blond_beam_to_xsuite_particles,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    xsuite_particles_to_blond_beam,
    zeta_to_dt,
)


def _frame():
    return ReferenceFrame(beta0=0.999, energy0=450e9)


def _matched_context():
    """xobjects context matching BLonD's active backend.

    The converters assume the xsuite ``Particles`` and the BLonD beam share an
    array library (the production wrappers guarantee this — see
    ``WrapXsuite4Blond._resolve_context``). On a GPU backend the beam arrays
    are cupy, so the particles must live on a ``ContextCupy`` too; otherwise a
    ``cupy_array[:] = numpy_array`` write raises inside cupy's ``fill``. Reuse
    the shared harness context so these unit tests are backend agnostic.
    """
    from tests.integration.xsuite._lhc_harness import matched_context

    return matched_context()


def _beam(n):
    beam = Beam(intensity=1e9, particle_type=proton)
    beam.setup_beam(dt=np.zeros(n), dE=np.zeros(n))
    return beam


def test_xsuite_particles_to_blond_beam_converts_all_active():
    frame = _frame()
    zeta = np.array([-0.3, 0.0, 0.4, 0.9])
    ptau = np.array([-1e-4, 0.0, 2e-4, 1e-4])
    particles = xt.Particles(
        _context=_matched_context(),
        p0c=1e9,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta,
        ptau=ptau,
    )
    beam = _beam(len(zeta))

    active = xsuite_particles_to_blond_beam(particles, beam, frame)

    assert active.all()
    np.testing.assert_allclose(
        copy_to_cpu(beam.read_partial_dt()),
        zeta_to_dt(zeta, frame),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        copy_to_cpu(beam.read_partial_dE()),
        ptau_to_dE(ptau, frame),
        rtol=1e-12,
    )


def test_xsuite_particles_to_blond_beam_flags_lost():
    frame = _frame()
    zeta = np.array([0.0, 0.1, 0.2, 0.3])
    ptau = np.zeros(4)
    particles = xt.Particles(
        _context=_matched_context(),
        p0c=1e9,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta,
        ptau=ptau,
    )
    particles.state[1] = 0  # mark particle 1 lost
    particles.state[3] = 0  # mark particle 3 lost
    beam = _beam(len(zeta))

    active = xsuite_particles_to_blond_beam(particles, beam, frame)

    np.testing.assert_array_equal(
        copy_to_cpu(active), [True, False, True, False]
    )
    flags = copy_to_cpu(beam.read_partial_flags())
    assert flags[0] == BeamFlags.ACTIVE.value
    assert flags[1] == BeamFlags.LOST.value
    assert flags[2] == BeamFlags.ACTIVE.value
    assert flags[3] == BeamFlags.LOST.value


def test_round_trip_particles_beam_particles():
    frame = _frame()
    zeta = np.array([-0.5, -0.1, 0.0, 0.2, 0.7])
    ptau = np.array([-2e-4, -1e-4, 0.0, 1e-4, 3e-4])
    particles = xt.Particles(
        _context=_matched_context(),
        p0c=1e9,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta.copy(),
        ptau=ptau.copy(),
    )
    beam = _beam(len(zeta))

    active = xsuite_particles_to_blond_beam(particles, beam, frame)
    blond_beam_to_xsuite_particles(beam, particles, frame, active)

    np.testing.assert_allclose(copy_to_cpu(particles.zeta), zeta, rtol=1e-10)
    np.testing.assert_allclose(copy_to_cpu(particles.ptau), ptau, rtol=1e-10)


def test_blond_beam_to_xsuite_particles_leaves_lost_untouched():
    frame = _frame()
    zeta = np.array([0.0, 0.5, 1.0])
    ptau = np.zeros(3)
    particles = xt.Particles(
        _context=_matched_context(),
        p0c=1e9,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta.copy(),
        ptau=ptau.copy(),
    )
    particles.state[1] = 0
    beam = _beam(len(zeta))

    active = xsuite_particles_to_blond_beam(particles, beam, frame)
    # mutate the beam dt for all slots, then write back
    beam.write_partial_dt()[:] = 1e-9
    blond_beam_to_xsuite_particles(beam, particles, frame, active)

    # active particles updated, lost particle's zeta unchanged
    zeta_cpu = copy_to_cpu(particles.zeta)
    assert zeta_cpu[1] == 0.5
    np.testing.assert_allclose(
        zeta_cpu[[0, 2]],
        dt_to_zeta(np.array([1e-9, 1e-9]), frame),
        rtol=1e-12,
    )


def test_blond_beam_to_xsuite_particles_recomputes_active_when_none():
    """With ``active=None`` the mask is recomputed from ``state > 0``."""
    frame = _frame()
    zeta = np.array([0.0, 0.5, 1.0])
    ptau = np.zeros(3)
    particles = xt.Particles(
        _context=_matched_context(),
        p0c=1e9,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta.copy(),
        ptau=ptau.copy(),
    )
    particles.state[1] = 0
    beam = _beam(len(zeta))

    xsuite_particles_to_blond_beam(particles, beam, frame)
    beam.write_partial_dt()[:] = 1e-9
    # No ``active`` passed: the function must recompute it internally.
    blond_beam_to_xsuite_particles(beam, particles, frame)

    zeta_cpu = copy_to_cpu(particles.zeta)
    assert zeta_cpu[1] == 0.5  # lost particle untouched
    np.testing.assert_allclose(
        zeta_cpu[[0, 2]],
        dt_to_zeta(np.array([1e-9, 1e-9]), frame),
        rtol=1e-12,
    )


if __name__ == "__main__":
    unittest.main()
