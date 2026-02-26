from unittest.mock import Mock

import numpy as np
import pytest
import xpart as xp
import xtrack as xt

from blond import SingleHarmonicRFStation, proton
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonD3Cavity,
    blond_to_xsuite_transform,
    particle_xsuite_to_blond,
    xsuite_to_blond_transform,
)
from legacy.unittests.trackers.test_drift import relative_tolerance


def test_forward_backward_transform_consistency():
    """
    Test that xsuite -> BLonD -> xsuite returns the original coordinates.
    """

    rng = np.random.default_rng(12345)

    # Reference parameters
    beta0 = 1
    energy0 = 450e9  # eV
    omega_rf = 2 * np.pi * 400e6  # rad/s
    phi_s = 0.0  # rad
    n_particles = 100

    # Initial xsuite coordinates
    if n_particles == 1:
        zeta = rng.uniform(-1e-2, 1e-2)
        ptau = rng.uniform(-1e-4, 1e-4)
    else:
        zeta = rng.uniform(-1e-2, 1e-2, n_particles)
        ptau = rng.uniform(-1e-4, 1e-4, n_particles)

    # Forward transform: xsuite -> BLonD
    dt, dE = xsuite_to_blond_transform(
        zeta=zeta,
        ptau=ptau,
        beta0=beta0,
        energy0=energy0,
        omega_rf=omega_rf,
        phi_s=phi_s,
    )

    # Backward transform: BLonD -> xsuite
    zeta_back, ptau_back = blond_to_xsuite_transform(
        dt=dt,
        de=dE,
        beta0=beta0,
        energy0=energy0,
        omega_rf=omega_rf,
        phi_s=phi_s,
    )

    np.testing.assert_allclose(
        zeta_back,
        zeta,
        rtol=0,
        atol=1e-14,
        err_msg="zeta not preserved by forward/backward transform",
    )

    np.testing.assert_allclose(
        ptau_back,
        ptau,
        rtol=0,
        atol=1e-14,
        err_msg="ptau not preserved by forward/backward transform",
    )


def test_proton_mass_and_charge_consistency():
    """
    Test that the proton definition in Xsuite and BLonD are consistent,
    and that particle_xsuite_to_blond preserves mass and charge.
    """

    xsuite_particle = xp.Particles(
        p0c=450e9,
        mass0=xp.PROTON_MASS_EV,
        q0=1.0,
    )

    blond_particle = particle_xsuite_to_blond(xsuite_particle)

    assert blond_particle.mass == pytest.approx(
        proton.mass,
        rel=1e-6,
        abs=1e-5,  # eV-level tolerance
    ), "mass mismatch between Xsuite and BLonD for proton"

    assert blond_particle.charge == pytest.approx(
        proton.charge,
        rel=0,
        abs=1e-5,
    ), "charge mismatch between Xsuite and BLonD for proton"


def test_reference_energy_matches_magnetic_cycle_target():
    """
    Check that the BLonD beam reference energy matches the magnetic
    cycle target total energy after tracking.
    """

    C = 26658.8832  # circumference [m]
    p0c = 450e9  # eV
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    h = 35640
    matrix = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0,
        frequency_rf=0,
        lag_rf=0,
        momentum_compaction_factor=alpha,
        length=C,
    )

    line = xt.Line(elements=[matrix], element_names={"matrix"})
    line.particle_ref = xp.Particles(
        p0c=p0c,
        mass0=xp.PROTON_MASS_EV,
        q0=1.0,
    )

    particles = xp.Particles(
        p0c=p0c,
        mass0=xp.PROTON_MASS_EV,
        q0=1.0,
        zeta=[0.0],
        ptau=[0.0],
    )
    mass0 = float(particles.mass0)  #

    # --- BLonD cavity ---
    cavity = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=5e6,
        harmonic=h,
        phi_rf=0.0,
        circumference=C,
        total_energy=float(np.sqrt(p0c**2 + mass0**2)),
        is_below_transition=False,
        beam_reference_beta=float(line.particle_ref.beta0[0]),
    )

    # --- Mock magnetic cycle ---
    E0_expected = float(np.sqrt(p0c**2 + mass0**2))

    magnetic_cycle = Mock()
    magnetic_cycle.get_target_total_energy.return_value = E0_expected
    cavity._magnetic_cycle = magnetic_cycle

    blond_cavity = BLonD3Cavity(
        cavity=cavity,
        particles=particles,
        line=line,
        initial_intensity=1,
    )

    blond_cavity.track(particles)

    assert blond_cavity.beam.reference.total_energy == pytest.approx(
        E0_expected,
        rel=1e-12,
    )

    magnetic_cycle.get_target_total_energy.assert_called()
