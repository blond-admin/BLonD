"""Tests for WrapBlond4Xsuite: a BLonD element used inside an xsuite Line."""

from unittest import mock

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
xo = pytest.importorskip("xobjects")

from blond import (  # noqa: E402
    Beam,
    SingleHarmonicRFStation,
    copy_to_cpu,
    proton,
)
from blond.core.backends.backend import backend  # noqa: E402
from blond.core.base import UserDefinedElement  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_element import (  # noqa: E402
    WrapBlond4Xsuite,
)
from blond.testing.backend_testing import BLonDTestCase  # noqa: E402

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


def _matched_context():
    """Return the xobjects context matching the active BLonD backend.

    Tests must pass under whatever process-global backend an earlier
    (randomly ordered) test left behind. Contexts are shared so xtrack's
    per-context kernel caches are reused across tests.
    """
    from tests.integration.xsuite._lhc_harness import matched_context

    return matched_context()


def _xsuite_particles(zeta, ptau, total_energy, ctx=None):
    p0c = _proton_p0c(total_energy)
    return xt.Particles(
        _context=ctx if ctx is not None else _matched_context(),
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
    expected_zeta = -copy_to_cpu(direct_beam.read_partial_dt()) * beta0 * c
    expected_ptau = copy_to_cpu(direct_beam.read_partial_dE()) / (
        beta0 * energy0
    )

    np.testing.assert_allclose(
        copy_to_cpu(particles.zeta), copy_to_cpu(expected_zeta), rtol=1e-10
    )
    np.testing.assert_allclose(
        copy_to_cpu(particles.ptau), copy_to_cpu(expected_ptau), rtol=1e-10
    )


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

    assert float(particles.zeta[1]) == zeta_lost_before
    assert float(particles.ptau[1]) == 0.0


class _BlondElementRemovingMiddleSlot:
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


def test_wrapper_rejects_element_with_magnetic_cycle():
    """An element driving the reference itself must be refused at wrap time.

    Inside an xsuite line, xsuite owns the reference energy; a wrapped
    station with its own magnetic cycle would relabel ``dE`` every turn
    while xsuite's ``p0c`` never moves — silent double counting. Mirror of
    ``WrapXsuite4Blond``'s ``energy_program`` guard.
    """
    from blond import ConstantMagneticCycle

    total_energy = 450e9
    cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=_proton_p0c(total_energy),
        in_unit="momentum",
    )
    p0c = _proton_p0c(total_energy)
    rf = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=VOLTAGE,
        phi_rf=0.0,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        beam_reference_beta=p0c / total_energy,
        magnetic_cycle=cycle,
    )
    with pytest.raises(ValueError, match="magnetic_cycle"):
        WrapBlond4Xsuite(element=rf)


def test_turn_keyed_schedule_follows_xsuite_turns():
    """A per-turn (ScheduledArray) schedule advances with xsuite's turns.

    The wrapper derives the turn index from ``particles.at_turn`` — xsuite's
    own bookkeeping — so the wrapped element's schedule machinery sees the
    correct turn without any user-managed counter.
    """
    energy0 = 450e9
    rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
    voltages = VOLTAGE * np.array([1.0, 2.0, 3.0, 4.0])
    rf.schedule(attribute="voltage", value=voltages)

    ctx = _matched_context()
    line = xt.Line(
        elements=[WrapBlond4Xsuite(element=rf)], element_names=["blond"]
    )
    line.build_tracker(_context=ctx)
    particles = _xsuite_particles([0.1, 0.0, -0.2], np.zeros(3), energy0, ctx)

    for turn in range(3):
        line.track(particles, num_turns=1)
        # apply_schedules ran inside the element's track at this turn.
        assert rf.voltage == pytest.approx(voltages[turn]), f"{turn=}"


def test_turn_keyed_schedule_ignores_lost_particles():
    """The turn is read from an *active* particle (lost ones freeze at_turn)."""
    energy0 = 450e9
    rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
    voltages = VOLTAGE * np.array([1.0, 2.0, 3.0, 4.0])
    rf.schedule(attribute="voltage", value=voltages)

    ctx = _matched_context()
    line = xt.Line(
        elements=[WrapBlond4Xsuite(element=rf)], element_names=["blond"]
    )
    line.build_tracker(_context=ctx)
    particles = _xsuite_particles([0.1, 0.0, -0.2], np.zeros(3), energy0, ctx)
    particles.state[0] = 0  # lost before turn 0: its at_turn stays 0

    for turn in range(3):
        line.track(particles, num_turns=1)
        assert rf.voltage == pytest.approx(voltages[turn]), f"{turn=}"


def test_wrapper_rejects_time_keyed_schedules():
    """Time-keyed (ScheduledInterpolation) schedules are still refused.

    ``reference_time`` is not driven inside an xsuite-owned loop (yet), so a
    time-interpolated schedule would silently evaluate at t=0 forever —
    refuse it upfront and point to turn-keyed arrays.
    """
    rf = _headless_cavity(phi_rf=0.0, total_energy=450e9)
    rf.schedule(
        attribute="voltage",
        value=(
            np.linspace(0.0, 1.0, 10),
            np.linspace(VOLTAGE, 2 * VOLTAGE, 10),
        ),
    )
    with pytest.raises(NotImplementedError, match="[Tt]ime"):
        WrapBlond4Xsuite(element=rf)


def test_wrapper_rejects_particles_with_charge_ratio_not_one():
    """Per-particle charge ratios are not mapped to BLonD's uniform charge.

    BLonD kicks every macroparticle with the same charge; silently dropping
    ``charge_ratio`` would give partially-stripped-ion lines wrong kicks.
    """
    energy0 = 450e9
    rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
    wrapper = WrapBlond4Xsuite(element=rf)

    particles = xt.Particles(
        _context=_matched_context(),
        p0c=_proton_p0c(energy0),
        mass0=proton.mass,
        q0=proton.charge,
        zeta=np.zeros(3),
        ptau=np.zeros(3),
        charge_ratio=np.array([1.0, 0.5, 1.0]),
        mass_ratio=np.ones(3),
    )
    with pytest.raises(NotImplementedError, match="charge_ratio"):
        wrapper.track(particles)


def test_wrapper_rejects_non_uniform_weights():
    """Per-particle weight spreads are not mapped to BLonD's uniform ratio.

    BLonD models the bunch as ``intensity / n_macroparticles`` per slot;
    silently summing non-uniform xsuite weights would distort any wrapped
    intensity effect.
    """
    energy0 = 450e9
    rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
    wrapper = WrapBlond4Xsuite(element=rf)

    particles = xt.Particles(
        _context=_matched_context(),
        p0c=_proton_p0c(energy0),
        mass0=proton.mass,
        q0=proton.charge,
        zeta=np.zeros(3),
        ptau=np.zeros(3),
        weight=np.array([1e5, 2e5, 1e5]),
    )
    with pytest.raises(NotImplementedError, match="weight"):
        wrapper.track(particles)


def test_wrapper_rejects_length_bearing_elements():
    """A BLonD drift must stay on the xsuite side.

    xsuite owns the geometry: the wrapper is a thin element with no
    ``length``, so a wrapped element occupying orbit length would change
    ``dt`` while xsuite's ``s``/survey bookkeeping never moves.
    """
    from blond import DriftSimple

    drift = DriftSimple.headless(
        momentum_compaction_factor=1e-3, orbit_length=10.0
    )
    with pytest.raises(ValueError, match="orbit_length"):
        WrapBlond4Xsuite(element=drift)


def test_wrapper_accepts_length_bearing_rf_station():
    """An RF station carrying a bookkeeping ``orbit_length`` is wrappable.

    Real cavities occupy metres of the ring, so an RF element may carry an
    ``orbit_length`` for BLonD's ring-layout accounting — but its track is a
    thin kick: it never moves ``dt`` by a drift map nor advances the
    reference clock by a time of flight. Wrapping it therefore cannot
    disturb xsuite's geometry, and it must track exactly like the same
    station without a length. Only drift *physics* stays on the xsuite side.
    """
    from blond.core.base import HasOrbitLength

    energy0 = 450e9

    class _LengthBearingStation(SingleHarmonicRFStation, HasOrbitLength):
        """A thin-kick RF station that occupies orbit length (bookkeeping)."""

    thick = _headless_cavity(phi_rf=0.3, total_energy=energy0)
    # Re-class the headless station into the length-bearing test double —
    # the factory hardcodes ``SingleHarmonicRFStation``, and the RF-station
    # constructor chain does not forward ``orbit_length``.
    thick.__class__ = _LengthBearingStation
    thick.orbit_length = 10.0
    assert isinstance(thick, HasOrbitLength)

    wrapper = WrapBlond4Xsuite(element=thick)  # must not raise

    zeta = np.array([0.0, 0.5, -0.5])
    ptau = np.zeros(3)
    particles = _xsuite_particles(zeta, ptau, energy0)
    wrapper.track(particles)

    thin_reference = _xsuite_particles(zeta, ptau, energy0)
    WrapBlond4Xsuite(
        element=_headless_cavity(phi_rf=0.3, total_energy=energy0)
    ).track(thin_reference)

    np.testing.assert_allclose(
        copy_to_cpu(particles.ptau),
        copy_to_cpu(thin_reference.ptau),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        copy_to_cpu(particles.zeta), copy_to_cpu(zeta), rtol=1e-12
    )


def test_wrapper_propagates_blond_losses_back_to_xsuite():
    """A BLonD element that marks a slot LOST mid-track must update particles.state."""
    energy0 = 450e9
    wrapper = WrapBlond4Xsuite(element=_BlondElementRemovingMiddleSlot())

    zeta_in = np.array([-0.2, 0.0, 0.3])
    particles = _xsuite_particles(zeta_in, np.zeros(3), energy0)
    wrapper.track(particles)

    assert int(particles.state[0]) > 0
    assert int(particles.state[1]) <= 0
    assert int(particles.state[2]) > 0


def test_wrapper_rejects_unsupported_context():
    """Particles on an unsupported context type raise NotImplementedError."""
    energy0 = 450e9
    wrapper = WrapBlond4Xsuite(
        element=_headless_cavity(phi_rf=0.1, total_energy=energy0)
    )

    # Create a mock particles object with an unsupported context.
    class _UnsupportedContext:
        pass

    class _MockParticles:
        def __init__(self):
            self._context = _UnsupportedContext()

    particles = _MockParticles()

    with pytest.raises(
        NotImplementedError, match="ContextCpu and ContextCupy"
    ):
        wrapper.track(particles)


class TestTurnCounterWithAllParticlesLost(BLonDTestCase):
    """With every slot lost there is no live ``at_turn`` to read."""

    def test_turn_counter_keeps_last_value(self):
        # Lost particles freeze ``at_turn`` at their loss turn, so reading
        # one would jump the schedule; the counter must hold instead.
        energy0 = 450e9
        rf = _headless_cavity(phi_rf=0.0, total_energy=energy0)
        voltages = VOLTAGE * np.array([1.0, 2.0, 3.0, 4.0])
        rf.schedule(attribute="voltage", value=voltages)
        wrapper = WrapBlond4Xsuite(element=rf)

        ctx = _matched_context()
        line = xt.Line(elements=[wrapper], element_names=["blond"])
        line.build_tracker(_context=ctx)
        particles = _xsuite_particles(
            [0.1, 0.0, -0.2], np.zeros(3), energy0, ctx
        )
        line.track(particles, num_turns=2)  # last tracked turn is 1
        particles.state[:] = 0  # all lost, every at_turn is now 2

        wrapper.track(particles)

        self.assertEqual(float(rf.voltage), voltages[1])


class TestCpuParticlesSwitchGpuBackendToCpu(BLonDTestCase):
    """CPU particles move a GPU BLonD backend back to a CPU backend.

    ``backend.is_gpu`` is patched and ``setup_backend`` replaced by a
    recorder, so the switching logic is tested on any machine without
    touching the process-global backend.
    """

    def test_falls_back_to_numba_when_cpp_cannot_load(self):
        requested_modes = []

        def _setup_backend_without_cpp(mode):
            requested_modes.append(mode)
            if mode == "cpp":
                raise OSError("libblond is not compiled")

        energy0 = 450e9
        wrapper = WrapBlond4Xsuite(
            element=_headless_cavity(phi_rf=0.1, total_energy=energy0)
        )
        particles = _xsuite_particles(
            [0.0], [0.0], energy0, ctx=xo.ContextCpu()
        )

        with (
            mock.patch.object(backend, "_is_gpu", True),
            mock.patch(
                "blond.core.backends.helpers.setup_backend",
                _setup_backend_without_cpp,
            ),
            self.assertWarns(UserWarning),
        ):
            wrapper._sync_backend_with_context(particles)

        self.assertEqual(requested_modes, ["cpp", "numba"])
        self.assertIsNone(wrapper._beam)
