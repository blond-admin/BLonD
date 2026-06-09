"""Tests for WrapXsuite4Blond: an xsuite element used inside a BLonD Ring."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, proton  # noqa: E402
from blond.core.base import BeamPhysicsRelevant  # noqa: E402
from blond.core.beam.flags import BeamFlags  # noqa: E402
from blond.physics.cavities import RFStationBaseClass  # noqa: E402
from blond.physics.drifts import DriftBaseClass  # noqa: E402
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
    """The wrapper is a real BLonD element via the RF + drift base classes."""
    wrapper = WrapXsuite4Blond(xt.Drift(length=1.0))
    assert isinstance(wrapper, BeamPhysicsRelevant)
    assert isinstance(wrapper, RFStationBaseClass)
    assert isinstance(wrapper, DriftBaseClass)


def test_wrapper_orbit_length_inherits_from_guest():
    wrapper = WrapXsuite4Blond(xt.Drift(length=2.5))
    assert wrapper.orbit_length == pytest.approx(2.5)


def test_wrapper_orbit_length_override():
    wrapper = WrapXsuite4Blond(xt.Drift(length=2.5), orbit_length=7.0)
    assert wrapper.orbit_length == pytest.approx(7.0)


def test_wrapper_rejects_guest_without_length():
    class _NoLength:
        def track(self, particles):
            pass

    with pytest.raises(TypeError, match="orbit length"):
        WrapXsuite4Blond(_NoLength())


def test_wrapper_track_reference_advances_clock_time():
    """`track_reference` advances reference.time by orbit_length / velocity."""
    energy0 = 1e9
    beam = _blond_beam([0.0], [0.0], energy0)
    initial_time = float(beam.reference.time)

    wrapper = WrapXsuite4Blond(xt.Drift(length=10.0))
    energy_delta = wrapper.track_reference(beam.reference)

    expected_dt = 10.0 / beam.reference.velocity
    # Drift: time advances, energy doesn't (probe sees no change).
    assert energy_delta == pytest.approx(0.0, abs=1e-9)
    assert beam.reference.total_energy == pytest.approx(energy0, rel=1e-12)
    assert beam.reference.time == pytest.approx(
        initial_time + expected_dt, rel=1e-12
    )


def test_wrapper_track_reference_probe_picks_up_ReferenceEnergyIncrease():
    """Probe detects an energy advance baked into the wrapped guest."""
    energy0 = 1e9
    delta_p0c = 5e6
    beam = _blond_beam([0.0], [0.0], energy0)

    wrapper = WrapXsuite4Blond(
        xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c), orbit_length=0.0
    )
    energy_delta = wrapper.track_reference(beam.reference)

    expected_p0c = _proton_p0c(energy0) + delta_p0c
    expected_e0 = float(np.sqrt(expected_p0c**2 + proton.mass**2))
    assert beam.reference.total_energy == pytest.approx(expected_e0, rel=1e-12)
    assert energy_delta == pytest.approx(
        expected_e0 - energy0, rel=1e-12
    )


def test_wrapper_track_reference_probe_lost_particle_returns_zero():
    """If the probe is killed by the guest, track_reference returns 0 cleanly."""

    class _KillEverything:
        length = 1.0

        def track(self, particles):
            particles.state[:] = 0

    energy0 = 1e9
    beam = _blond_beam([0.0], [0.0], energy0)
    wrapper = WrapXsuite4Blond(_KillEverything())

    energy_delta = wrapper.track_reference(beam.reference)
    assert energy_delta == 0.0
    assert beam.reference.total_energy == pytest.approx(energy0, rel=1e-12)


def test_wrapper_track_reference_rejects_counter_rotating():
    """Counter-rotating beams are unsupported and must raise, not silently lie."""
    beam = _blond_beam([0.0], [0.0], 1e9)
    wrapper = WrapXsuite4Blond(xt.Drift(length=1.0))
    with pytest.raises(NotImplementedError, match="counter-rotating"):
        wrapper.track_reference(beam.reference, is_counter_rotating=True)


def test_wrapper_refuses_line_with_energy_program():
    """Item K guard: an xt.Line with its own energy_program must be refused."""
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    import xpart as _xp

    line.particle_ref = _xp.Particles(
        p0c=1e9, mass0=_xp.PROTON_MASS_EV, q0=1.0
    )
    line.energy_program = xt.EnergyProgram(
        t_s=np.array([0.0, 1.0]), p0c=np.array([1e9, 1.001e9])
    )
    with pytest.raises(ValueError, match="energy_program"):
        WrapXsuite4Blond(line)


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
    wrapper = WrapXsuite4Blond(_KillSlot(index=1), orbit_length=0.0)
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
    wrapper = WrapXsuite4Blond(
        _BumpReference(new_energy0, proton.mass), orbit_length=0.0
    )
    wrapper.track(beam)

    assert beam.reference.total_energy == pytest.approx(new_energy0, rel=1e-10)


def test_wrapper_reuses_cached_buffers_across_turns():
    """Two same-N calls must not allocate a fresh Particles or _p0c_buf."""
    energy0 = 1e9
    beam = _blond_beam([1e-10, 0.0, -1e-10], [0.0, 0.0, 0.0], energy0)
    wrapper = WrapXsuite4Blond(xt.Drift(length=1.0))
    wrapper.track(beam)
    particles_after_first = wrapper._particles
    p0c_buf_after_first = wrapper._p0c_buf

    wrapper.track(beam)
    # Same object, not rebuilt — proves the build/refresh branch is taken
    # and we're not reallocating a fresh ``Particles`` every turn.
    assert wrapper._particles is particles_after_first
    assert wrapper._p0c_buf is p0c_buf_after_first


def test_wrapper_rebuilds_particles_on_size_change():
    """If the beam slot count changes, the cached buffers must follow."""
    wrapper = WrapXsuite4Blond(xt.Drift(length=1.0))
    wrapper.track(_blond_beam([0.0, 0.0], [0.0, 0.0], 1e9))
    particles_n2 = wrapper._particles

    wrapper.track(_blond_beam([0.0, 0.0, 0.0, 0.0], [0.0] * 4, 1e9))
    assert wrapper._particles is not particles_n2
    assert wrapper._particles.zeta.shape == (4,)
    assert wrapper._p0c_buf.shape == (4,)


def test_wrapper_propagates_reference_energy_increase():
    """An xsuite ReferenceEnergyIncrease guest must push the new reference into beam."""
    import xpart as _xp

    energy0 = 1e9
    delta_p0c = 5e6

    # Wrap a real xsuite element that advances the reference; the wrapper
    # should detect the new energy0 in its post-track check and update
    # beam.reference accordingly.
    line = xt.Line(
        elements=[xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c)],
        element_names=["accel"],
    )
    line.particle_ref = _xp.Particles(
        p0c=_proton_p0c(energy0), mass0=proton.mass, q0=proton.charge
    )
    line.build_tracker()

    beam = _blond_beam([0.0], [0.0], energy0)
    # Line has no energy_program, so the construction guard does not fire.
    wrapper = WrapXsuite4Blond(line)
    wrapper.track(beam)

    expected_p0c = _proton_p0c(energy0) + delta_p0c
    expected_e0 = float(np.sqrt(expected_p0c**2 + proton.mass**2))
    assert beam.reference.total_energy == pytest.approx(expected_e0, rel=1e-10)


# ---------------------------------------------------------------------------
# Capability claims — the wrapper inherits RFStationBaseClass + DriftBaseClass
# so BLonD's ring accounting (orbit length, reference advance, eta_0 / RF
# filters) treats it as a first-class element. The RF / drift *physics* is not
# mapped yet: those abstract members raise NotImplementedError until the
# richer features land.
# ---------------------------------------------------------------------------


def test_wrapper_claims_rf_and_drift_capabilities():
    """The wrapper is recognised as both a drift and an RF station."""
    w = WrapXsuite4Blond(xt.Drift(length=1.0))
    assert isinstance(w, DriftBaseClass)
    assert isinstance(w, RFStationBaseClass)


def test_wrapper_unmapped_physics_raises_not_implemented():
    """Every feature beyond track / track_reference / orbit_length is stubbed."""
    w = WrapXsuite4Blond(xt.Drift(length=1.0))
    with pytest.raises(NotImplementedError):
        w.eta_0(gamma=2.0)
    with pytest.raises(NotImplementedError):
        w.get_main_harmonic()
    with pytest.raises(NotImplementedError):
        w.get_main_harmonic_voltage()
    with pytest.raises(NotImplementedError):
        w.get_main_harmonic_phi_rf()
    with pytest.raises(NotImplementedError):
        w.calc_main_harmonic_omega_rf_design(
            beam_beta=0.9, ring_circumference=100.0
        )
    with pytest.raises(NotImplementedError):
        w.get_main_harmonic_omega_rf()


def test_wrapper_accepts_arbitrary_xsuite_guest_with_length():
    """No isinstance dispatch — anything with .length (or `orbit_length=` kwarg) works."""
    # Drift, LineSegmentMap, Cavity, … all just work.
    for guest in (
        xt.Drift(length=1.0),
        xt.LineSegmentMap(
            longitudinal_mode="nonlinear",
            qx=1.1, qy=1.2, betx=1.0, bety=1.0,
            voltage_rf=0.0, frequency_rf=0.0, lag_rf=0.0,
            momentum_compaction_factor=1e-3, length=100.0,
        ),
    ):
        WrapXsuite4Blond(guest)  # must not raise


def test_wrapper_accepts_zero_length_guest_via_override():
    """xt.Cavity has no `.length` attribute we can rely on — explicit override works."""
    cav = xt.Cavity(voltage=1e6, frequency=400e6, lag=0.0)
    # Cavity has a .length attribute equal to 0 by default in xsuite,
    # so extraction succeeds. But the override path is the supported
    # contract for guests where we don't want to trust the attribute.
    w = WrapXsuite4Blond(cav, orbit_length=0.0)
    assert w.orbit_length == 0.0
