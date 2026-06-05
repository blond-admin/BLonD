"""Tests for WrapBlond4Xsuite: a BLonD element used inside an xsuite Line."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, SingleHarmonicRFStation, proton  # noqa: E402
from blond.cycles.magnetic_cycle import ExternalReferenceCycle  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_elelemt import (  # noqa: E402
    WrapBlond4Xsuite,
)


CIRCUMFERENCE = 26658.883
HARMONIC = 35640
VOLTAGE = 6e6


def _proton_p0c(total_energy):
    return float(np.sqrt(total_energy**2 - proton.mass**2))


def _headless_cavity(*, phi_rf, total_energy, magnetic_cycle):
    p0c = _proton_p0c(total_energy)
    beta0 = p0c / total_energy
    return SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=VOLTAGE,
        phi_rf=phi_rf,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        total_energy=total_energy,
        beam_reference_beta=beta0,
        magnetic_cycle=magnetic_cycle,
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

    cycle_w = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=energy0
    )
    rf_w = _headless_cavity(
        phi_rf=phi_rf, total_energy=energy0, magnetic_cycle=cycle_w
    )

    zeta_in = np.array([-0.5, 0.0, 0.4])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)

    wrapper = WrapBlond4Xsuite(element=rf_w)
    wrapper.track(particles)

    # Direct BLonD track for comparison
    cycle_d = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=energy0
    )
    rf_d = _headless_cavity(
        phi_rf=phi_rf, total_energy=energy0, magnetic_cycle=cycle_d
    )
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


def test_wrapper_drives_external_reference_energy():
    """When particles.energy0 changes, the ExternalReferenceCycle follows."""
    energy0 = 450e9
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=energy0
    )
    rf = _headless_cavity(
        phi_rf=0.0, total_energy=energy0, magnetic_cycle=cycle
    )
    wrapper = WrapBlond4Xsuite(element=rf)

    new_energy = 500e9
    particles = _xsuite_particles(
        np.array([0.0, 0.1]), np.zeros(2), new_energy
    )
    wrapper.track(particles)

    assert (
        cycle.get_target_total_energy(0, 0, 0.0, proton) == pytest.approx(
            new_energy, rel=1e-12
        )
    )


def test_wrapper_propagates_lost_particles_unchanged():
    energy0 = 450e9
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=energy0
    )
    rf = _headless_cavity(
        phi_rf=0.0, total_energy=energy0, magnetic_cycle=cycle
    )
    wrapper = WrapBlond4Xsuite(element=rf)

    zeta_in = np.array([0.1, 0.2, 0.3])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)
    particles.state[1] = 0  # mark particle 1 lost

    zeta_lost_before = float(particles.zeta[1])
    wrapper.track(particles)

    assert particles.zeta[1] == zeta_lost_before
    assert particles.ptau[1] == 0.0
