import numpy as np
import pytest
import xpart as xp

from blond import proton
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    blond_to_xsuite_transform,
    particle_xsuite_to_blond,
    xsuite_to_blond_transform,
)


@pytest.mark.parametrize("n_particles", [1, 10, 1000])
def test_forward_backward_transform_consistency(n_particles):
    """
    Test that xsuite -> BLonD -> xsuite returns the original coordinates.
    """

    rng = np.random.default_rng(12345)

    # Reference parameters
    beta0 = 0.999
    energy0 = 450e9  # eV
    omega_rf = 2 * np.pi * 400e6  # rad/s
    phi_s = 0.1  # rad

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


