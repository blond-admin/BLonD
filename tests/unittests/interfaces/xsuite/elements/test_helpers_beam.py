"""Tests for beam<->particle converters (require xsuite)."""
import unittest

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")

from blond import Beam, proton  # noqa: E402
from blond.core.beam.flags import BeamFlags  # noqa: E402
from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    beam_to_particles,
    dE_to_ptau,
    dt_to_zeta,
    particles_to_beam,
    ptau_to_dE,
    zeta_to_dt,
)


def _frame():
    return ReferenceFrame(beta0=0.999, energy0=450e9)


def _beam(n):
    beam = Beam(intensity=1e9, particle_type=proton)
    beam.setup_beam(dt=np.zeros(n), dE=np.zeros(n))
    return beam


def test_particles_to_beam_converts_all_active():
    frame = _frame()
    zeta = np.array([-0.3, 0.0, 0.4, 0.9])
    ptau = np.array([-1e-4, 0.0, 2e-4, 1e-4])
    particles = xt.Particles(p0c=1e9, mass0=proton.mass, q0=proton.charge,
                             zeta=zeta, ptau=ptau)
    beam = _beam(len(zeta))

    active = particles_to_beam(particles, beam, frame)

    assert active.all()
    np.testing.assert_allclose(
        beam.read_partial_dt(), zeta_to_dt(zeta, frame), rtol=1e-12
    )
    np.testing.assert_allclose(
        beam.read_partial_dE(), ptau_to_dE(ptau, frame), rtol=1e-12
    )


def test_particles_to_beam_flags_lost():
    frame = _frame()
    zeta = np.array([0.0, 0.1, 0.2, 0.3])
    ptau = np.zeros(4)
    particles = xt.Particles(p0c=1e9, mass0=proton.mass, q0=proton.charge,
                             zeta=zeta, ptau=ptau)
    particles.state[1] = 0  # mark particle 1 lost
    particles.state[3] = 0  # mark particle 3 lost
    beam = _beam(len(zeta))

    active = particles_to_beam(particles, beam, frame)

    np.testing.assert_array_equal(active, [True, False, True, False])
    flags = beam.read_partial_flags()
    assert flags[0] == BeamFlags.ACTIVE.value
    assert flags[1] == BeamFlags.LOST.value
    assert flags[2] == BeamFlags.ACTIVE.value
    assert flags[3] == BeamFlags.LOST.value


def test_round_trip_particles_beam_particles():
    frame = _frame()
    zeta = np.array([-0.5, -0.1, 0.0, 0.2, 0.7])
    ptau = np.array([-2e-4, -1e-4, 0.0, 1e-4, 3e-4])
    particles = xt.Particles(p0c=1e9, mass0=proton.mass, q0=proton.charge,
                             zeta=zeta.copy(), ptau=ptau.copy())
    beam = _beam(len(zeta))

    active = particles_to_beam(particles, beam, frame)
    beam_to_particles(beam, particles, frame, active)

    np.testing.assert_allclose(particles.zeta, zeta, rtol=1e-10)
    np.testing.assert_allclose(particles.ptau, ptau, rtol=1e-10)


def test_beam_to_particles_leaves_lost_untouched():
    frame = _frame()
    zeta = np.array([0.0, 0.5, 1.0])
    ptau = np.zeros(3)
    particles = xt.Particles(p0c=1e9, mass0=proton.mass, q0=proton.charge,
                             zeta=zeta.copy(), ptau=ptau.copy())
    particles.state[1] = 0
    beam = _beam(len(zeta))

    active = particles_to_beam(particles, beam, frame)
    # mutate the beam dt for all slots, then write back
    beam.write_partial_dt()[:] = 1e-9
    beam_to_particles(beam, particles, frame, active)

    # active particles updated, lost particle's zeta unchanged
    assert particles.zeta[1] == 0.5
    np.testing.assert_allclose(
        particles.zeta[[0, 2]], dt_to_zeta(np.array([1e-9, 1e-9]), frame),
        rtol=1e-12,
    )

if __name__ == "__main__":
    unittest.main()