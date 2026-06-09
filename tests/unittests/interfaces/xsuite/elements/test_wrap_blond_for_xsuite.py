"""Tests for WrapBlond4Xsuite: a BLonD element used inside an xsuite Line."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, SingleHarmonicRFStation, proton  # noqa: E402
from blond.core.base import UserDefinedElement  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_elelemt import (  # noqa: E402
    WrapBlond4Xsuite,
)


CIRCUMFERENCE = 26658.883
HARMONIC = 35640
VOLTAGE = 6e6


def _proton_p0c(total_energy):
    return float(np.sqrt(total_energy**2 - proton.mass**2))


def _headless_cavity(*, phi_rf, total_energy):
    p0c = _proton_p0c(total_energy)
    beta0 = p0c / total_energy
    return SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=VOLTAGE,
        phi_rf=phi_rf,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        beam_reference_beta=beta0,
        magnetic_cycle=None,
    )


def _xsuite_particles(zeta, ptau, total_energy):
    p0c = _proton_p0c(total_energy)
    return xt.Particles(
        p0c=p0c,
        mass0=proton.mass,
        q0=proton.charge,
        zeta=np.asarray(zeta, dtype=float),
        ptau=np.asarray(ptau, dtype=float),
    )


def test_wrapper_flat_energy_matches_direct_track():
    energy0 = 450e9
    beta0 = _proton_p0c(energy0) / energy0
    phi_rf = 0.1

    rf_w = _headless_cavity(phi_rf=phi_rf, total_energy=energy0)
    zeta_in = np.array([-0.5, 0.0, 0.4])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)

    wrapper = WrapBlond4Xsuite(element=rf_w)
    wrapper.track(particles)

    # Direct BLonD track for comparison.
    rf_d = _headless_cavity(phi_rf=phi_rf, total_energy=energy0)
    direct_beam = Beam(intensity=1.0, particle_type=proton)
    direct_beam.setup_beam(
        dt=-zeta_in / (beta0 * c),
        dE=np.zeros(3),
        reference_total_energy=energy0,
    )
    rf_d.track(direct_beam)
    expected_zeta = -direct_beam.read_partial_dt() * beta0 * c
    expected_ptau = direct_beam.read_partial_dE() / (beta0 * energy0)

    np.testing.assert_allclose(particles.zeta, expected_zeta, rtol=1e-10)
    np.testing.assert_allclose(particles.ptau, expected_ptau, rtol=1e-10)


class _ReferenceProbe(UserDefinedElement):
    """Records the reference energy the wrapped beam carries at track time."""

    def __init__(self):
        super().__init__()
        self.seen_energies: list[float] = []

    def _track(self, beam) -> None:
        self.seen_energies.append(float(beam.reference.total_energy))


def test_wrapper_pushes_xsuite_reference_into_beam():
    """The BLonD beam.reference follows xsuite's particles.energy0 each call."""
    probe = _ReferenceProbe()
    wrapper = WrapBlond4Xsuite(element=probe)

    e1, e2 = 450e9, 500e9
    wrapper.track(_xsuite_particles([0.0], [0.0], e1))
    wrapper.track(_xsuite_particles([0.0], [0.0], e2))

    assert probe.seen_energies[0] == pytest.approx(e1, rel=1e-12)
    assert probe.seen_energies[1] == pytest.approx(e2, rel=1e-12)


def test_wrapper_propagates_lost_particles_unchanged():
    energy0 = 450e9
    rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
    wrapper = WrapBlond4Xsuite(element=rf)

    zeta_in = np.array([0.1, 0.2, 0.3])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)
    particles.state[1] = 0  # mark particle 1 lost

    zeta_lost_before = float(particles.zeta[1])
    wrapper.track(particles)

    assert particles.zeta[1] == zeta_lost_before
    assert particles.ptau[1] == 0.0


class _BlondElementKillingMiddleSlot:
    """BLonD-side fake element that flags beam slot 1 LOST during its track.

    Stands in for any BLonD element that drops particles via the flag array
    (loss box, energy cut, ...); used to verify ``WrapBlond4Xsuite`` reflects
    those losses into the xsuite ``particles.state`` after the call.
    The wrapper only needs ``.track(beam)`` on the guest, so this is
    intentionally a duck-typed shim, not a real ``UserDefinedElement``.
    """

    def track(self, beam):
        from blond.core.beam.flags import BeamFlags as _Flags

        flags = beam.write_partial_flags()
        flags[1] = _Flags.LOST.value


def test_wrapper_propagates_blond_losses_back_to_xsuite():
    """A BLonD element that marks a slot LOST mid-track must update particles.state."""
    energy0 = 450e9
    wrapper = WrapBlond4Xsuite(element=_BlondElementKillingMiddleSlot())

    zeta_in = np.array([-0.2, 0.0, 0.3])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)
    wrapper.track(particles)

    assert int(particles.state[0]) > 0
    assert int(particles.state[1]) <= 0
    assert int(particles.state[2]) > 0
