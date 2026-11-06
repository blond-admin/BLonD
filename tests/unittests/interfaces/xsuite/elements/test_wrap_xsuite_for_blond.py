"""Tests for WrapXsuite4Blond: an xsuite element used inside a BLonD Ring."""

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")

from blond import Beam, copy_to_cpu, proton  # noqa: E402
from blond.core.base import BeamPhysicsRelevant  # noqa: E402
from blond.core.base import AltersReference, HasOrbitLength
from blond.core.beam.flags import BeamFlags  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_xsuite_element import (  # noqa: E402
    WrapXsuite4Blond,
)
from blond.physics.drifts import DriftBaseClass  # noqa: E402
from blond.physics.rf_station import RFStationBaseClass  # noqa: E402
from blond.testing.backend_testing import BLonDTestCase  # noqa: E402


def _proton_p0c(total_energy):
    return float(np.sqrt(total_energy**2 - proton.mass**2))


def _matched_context():
    """xobjects context matching BLonD's active backend.

    The wrapper runs its guest line on the context dictated by BLonD's backend
    (cupy on a GPU backend), so a line whose tracker is pre-built here must use
    the matching context — otherwise ``_resolve_context`` refuses the mismatch.
    Reuse the shared harness context so these tests are backend agnostic.
    """
    from tests.integration.xsuite._lhc_harness import matched_context

    return matched_context()


def _blond_beam(dt, dE, total_energy):
    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.asarray(dt, dtype=float).copy(),
        dE=np.asarray(dE, dtype=float).copy(),
        reference_total_energy=total_energy,
    )
    return beam


def _one_element_line(element, name="e"):
    """Wrap a single xsuite element in a one-element line.

    ``WrapXsuite4Blond`` takes an ``xt.Line``; this is the canonical way to
    feed it a lone element in the tests.
    """
    return xt.Line(elements=[element], element_names=[name])


def test_wrapper_is_a_black_box_element():
    """The wrapper is a length-bearing, reference-altering BLonD element —
    and deliberately NOT a drift or RF station by type, so drift physics
    (eta_0 averaging, synchrotron radiation) and RF accounting (n_rf_stations,
    separatrix, feedbacks) never see it.
    """

    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))
    assert isinstance(wrapper, BeamPhysicsRelevant)
    assert isinstance(wrapper, AltersReference)
    assert isinstance(wrapper, HasOrbitLength)
    assert not isinstance(wrapper, RFStationBaseClass)
    assert not isinstance(wrapper, DriftBaseClass)


def test_wrapper_orbit_length_inherits_from_guest():
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=2.5)))
    assert wrapper.orbit_length == pytest.approx(2.5)


def test_wrapper_rejects_non_line_guest():
    """The wrapper takes an xt.Line; arbitrary objects are refused."""

    class _NotALine:
        def track(self, particles):
            pass

    with pytest.raises(TypeError, match="xt.Line"):
        WrapXsuite4Blond(_NotALine())


def test_wrapper_rejects_bare_beam_element():
    """A bare BeamElement (not a Line) is refused — wrap it in a line first."""
    with pytest.raises(TypeError, match="xt.Line"):
        WrapXsuite4Blond(xt.Drift(length=1.0))


def test_wrapper_track_reference_advances_clock_time():
    """`track_reference` advances reference.time by orbit_length / velocity."""
    energy0 = 1e9
    beam = _blond_beam([0.0], [0.0], energy0)
    initial_time = float(beam.reference.time)

    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=10.0)))
    energy_delta = wrapper.track_reference(beam.reference)

    expected_dt = 10.0 / beam.reference.velocity
    # Drift: time advances, energy doesn't (probe sees no change).
    assert energy_delta == pytest.approx(0.0, abs=1e-9)
    assert beam.reference.total_energy == pytest.approx(energy0, rel=1e-12)
    assert beam.reference.time == pytest.approx(
        initial_time + expected_dt, rel=1e-12
    )


def _expected_energy_after_kick(energy0: float, delta_p0c: float) -> float:
    """Total energy [eV] after one ``ReferenceEnergyIncrease`` pass."""
    p0c = float(np.sqrt(energy0**2 - proton.mass**2)) + delta_p0c
    return float(np.sqrt(p0c**2 + proton.mass**2))


def test_track_reference_does_not_track_the_guest_line():
    """``track_reference`` must never run the wrapped line itself.

    The guest line may be stateful (time-dependent variables, monitors,
    wakefields); only the real beam pass in ``track`` may execute it. The
    reference probe runs a single particle through a throwaway copy
    instead — the guest's own ``track`` stays untouched.
    """
    line = _one_element_line(xt.Drift(length=1.0))
    calls = []
    original_track = line.track
    line.track = lambda *args, **kwargs: (
        calls.append(1),
        original_track(*args, **kwargs),
    )[-1]
    beam = _blond_beam([0.0], [0.0], 1e9)
    wrapper = WrapXsuite4Blond(line)

    wrapper.track_reference(beam.reference)

    assert calls == []


def test_track_reference_forecasts_guest_advance_without_beam_pass():
    """The reference probe reports a guest ramp before any beam pass.

    ``track_reference`` tracks a single probe particle through a copy of
    the guest line, so a guest-driven ramp (``xt.ReferenceEnergyIncrease``)
    is forecast exactly — no beam pass needed, no one-turn lag, and no
    state shared with ``track``.
    """
    energy0 = 1e9
    delta_p0c = 5e6
    beam = _blond_beam([0.0], [0.0], energy0)
    wrapper = WrapXsuite4Blond(
        _one_element_line(xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c))
    )

    energy_delta = wrapper.track_reference(beam.reference)

    expected = _expected_energy_after_kick(energy0, delta_p0c)
    assert energy_delta == pytest.approx(expected - energy0, rel=1e-12)
    assert beam.reference.total_energy == pytest.approx(expected, rel=1e-12)


def test_track_reference_repeated_calls_simulate_successive_passes():
    """N ``track_reference`` calls forecast N successive guest passes.

    Like every other ``AltersReference`` element, the wrapper's reference
    advance is a pure function of the incoming reference state — scans that
    iterate turns (``n_turns_max``) get the correct cumulative ramp.
    """
    energy0 = 1e9
    delta_p0c = 5e6
    reference = _blond_beam([0.0], [0.0], energy0).reference
    wrapper = WrapXsuite4Blond(
        _one_element_line(xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c))
    )

    wrapper.track_reference(reference)
    wrapper.track_reference(reference)

    energy_1 = _expected_energy_after_kick(energy0, delta_p0c)
    energy_2 = _expected_energy_after_kick(energy_1, delta_p0c)
    assert reference.total_energy == pytest.approx(energy_2, rel=1e-12)


def test_track_reference_probes_passive_line_only_once():
    """A guest that never moves ``p0c`` is probed once, then skipped.

    The probe (line copy + one-particle track) costs milliseconds; paying
    it every turn for a plain wrapped drift would slow the per-turn
    forecast for nothing. The first probe classifies the line as passive,
    after which ``track_reference`` is a pure time-of-flight advance.
    """
    line = _one_element_line(xt.Drift(length=10.0))
    wrapper = WrapXsuite4Blond(line)
    reference = _blond_beam([0.0], [0.0], 1e9).reference

    assert wrapper.track_reference(reference) == 0.0

    def _no_more_copies(*args, **kwargs):
        raise AssertionError("passive line must not be re-probed")

    line.copy = _no_more_copies
    assert wrapper.track_reference(reference) == 0.0


def _wrapper_after_guest_advance(energy0: float, delta_p0c: float):
    """Build a wrapper that observed one guest-driven energy advance.

    Returns
    -------
    tuple[WrapXsuite4Blond, Beam, float]
        The wrapper, the tracked beam, and the post-pass total energy [eV].
    """
    import xpart as _xp

    ctx = _matched_context()
    line = xt.Line(
        elements=[xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c)],
        element_names=["accel"],
    )
    line.particle_ref = _xp.Particles(
        _context=ctx,
        p0c=_proton_p0c(energy0),
        mass0=proton.mass,
        q0=proton.charge,
    )
    line.build_tracker(_context=ctx)
    wrapper = WrapXsuite4Blond(line)

    beam = _blond_beam([0.0], [0.0], energy0)
    wrapper.track(beam)

    energy_after = float(beam.reference.total_energy)
    assert energy_after > energy0  # the guest really advanced the reference
    return wrapper, beam, energy_after


def test_track_reference_simulates_a_pass_from_any_reference_state():
    """The forecast is exact from any incoming reference state.

    Because the probe simulates a pass on a copy of the guest line's
    *current* state, the forecast neither lags one turn behind the live
    ramp (forecast from the live state = the next pass's advance) nor
    pollutes rewound scans (``get_t_rev_init`` re-tracks from the cycle
    start and gets the advance a pass from *that* state would produce).
    """
    energy0 = 1e9
    delta_p0c = 5e6
    wrapper, beam, energy_after = _wrapper_after_guest_advance(
        energy0, delta_p0c
    )

    # Live path: the guest advance reached beam.reference during track.
    assert energy_after == pytest.approx(
        _expected_energy_after_kick(energy0, delta_p0c), rel=1e-12
    )

    from copy import copy

    # Forecast from the live post-pass state = the *next* pass's advance.
    forecast = copy(beam.reference)
    expected_next = _expected_energy_after_kick(energy_after, delta_p0c)
    assert wrapper.track_reference(forecast) == pytest.approx(
        expected_next - energy_after, rel=1e-12
    )

    # A rewound scan simulates a pass from the rewound state.
    rewound_energy = 0.99e9  # above the proton rest mass, below energy0
    rewound = _blond_beam([0.0], [0.0], rewound_energy).reference
    expected_rewound = _expected_energy_after_kick(rewound_energy, delta_p0c)
    assert wrapper.track_reference(rewound) == pytest.approx(
        expected_rewound - rewound_energy, rel=1e-12
    )


def test_track_advances_live_reference_time():
    """``track`` must advance ``beam.reference.time`` like every drift does.

    Time-based magnetic cycles and schedules read the live reference clock;
    if the wrapper replaces the BLonD drift, it owns the time advance.
    """
    energy0 = 1e9
    length = 10.0
    beam = _blond_beam([0.0], [0.0], energy0)
    initial_time = float(beam.reference.time)
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=length)))

    wrapper.track(beam)

    expected_dt = length / beam.reference.velocity
    assert beam.reference.time == pytest.approx(
        initial_time + expected_dt, rel=1e-12
    )


def test_wrapper_track_reference_rejects_counter_rotating():
    """Counter-rotating beams are unsupported and must raise, not silently lie."""
    beam = _blond_beam([0.0], [0.0], 1e9)
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))
    with pytest.raises(NotImplementedError, match="counter-rotating"):
        wrapper.track_reference(beam.reference, is_counter_rotating=True)


def test_wrapper_refuses_line_with_energy_program():
    """A wrapped xsuite Line cannot have its own energy_program.

    BLonD owns the reference energy inside the Ring; a guest line with its own
    energy_program would update the reference independently, causing
    double-counting of acceleration. This guards against silent misuse.
    """
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
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=L)))
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
        copy_to_cpu(beam.read_partial_dt()), expected_dt, rtol=1e-10
    )
    np.testing.assert_allclose(
        copy_to_cpu(beam.read_partial_dE()), expected_dE, rtol=1e-10
    )


def test_wrapper_zero_offset_particles_unchanged_by_drift():
    """A synchronous particle (dE=0) through any drift should keep dt=0."""
    energy0 = 1e9
    beam = _blond_beam([0.0], [0.0], energy0)
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=2.5)))
    wrapper.track(beam)
    np.testing.assert_allclose(
        copy_to_cpu(beam.read_partial_dt()), [0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        copy_to_cpu(beam.read_partial_dE()), [0.0], atol=1e-15
    )


class _DeactivateSlot(xt.Line):
    """One-element line that drops one slot to ``state=0`` (mimics an aperture cut)."""

    def __init__(self, index: int):
        super().__init__(elements=[xt.Drift(length=0.0)], element_names=["d"])
        self._index = index

    def track(self, particles, **kwargs):
        particles.state[self._index] = 0


def test_wrapper_flags_lost_particles_from_xsuite():
    """Particles xsuite marks state<=0 must end up flagged LOST in the beam."""
    beam = _blond_beam([1e-9, 2e-9, 3e-9], [0.0, 0.0, 0.0], 1e9)
    wrapper = WrapXsuite4Blond(_DeactivateSlot(index=1))
    wrapper.track(beam)

    flags = beam.read_partial_flags()
    assert flags[0] == BeamFlags.ACTIVE.value
    assert flags[1] == BeamFlags.LOST.value
    assert flags[2] == BeamFlags.ACTIVE.value


class _BumpReference(xt.Line):
    """One-element line that advances ``particles.energy0`` / ``beta0`` in place."""

    def __init__(self, target_energy0: float, mass0: float):
        super().__init__(elements=[xt.Drift(length=0.0)], element_names=["d"])
        self._target = target_energy0
        self._mass0 = mass0

    def track(self, particles, **kwargs):
        new_p0c = float(np.sqrt(self._target**2 - self._mass0**2))
        particles.energy0[:] = self._target
        particles.p0c[:] = new_p0c
        particles.beta0[:] = new_p0c / self._target


def test_guest_particles_carry_unit_weight():
    """The guest xsuite Particles carry weight 1, not BLonD's bunch weight.

    One macroparticle in the guest line represents one real particle, so
    intensity effects *inside* the wrapped line (space charge, wakefields)
    do not see BLonD's bunch population — model them on the BLonD side of
    the ring instead.
    """
    n = 4
    beam = Beam(intensity=5e10, particle_type=proton)
    beam.setup_beam(dt=np.zeros(n), dE=np.zeros(n), reference_total_energy=1e9)

    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))
    wrapper.track(beam)

    np.testing.assert_allclose(
        copy_to_cpu(wrapper._particles.weight), np.ones(n), rtol=1e-12
    )


def test_fresh_beam_is_not_corrupted_by_previous_losses():
    """Losses from an earlier beam must not leak into a fresh same-size beam.

    The wrapper caches its xsuite ``Particles`` between calls; slots marked
    lost while tracking one beam must be revived when the next beam (same
    size, e.g. a re-run or a parameter scan) comes through.
    """
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))

    beam_a = _blond_beam([0.0] * 4, [0.0] * 4, 1e9)
    beam_a.flags.array_local[2] = BeamFlags.LOST.value
    wrapper.track(beam_a)
    assert beam_a.read_partial_flags()[2] == BeamFlags.LOST.value  # sanity

    beam_b = _blond_beam([0.0] * 4, [0.0] * 4, 1e9)
    wrapper.track(beam_b)

    np.testing.assert_array_equal(
        copy_to_cpu(beam_b.read_partial_flags()),
        np.full(4, BeamFlags.ACTIVE.value),
    )


def test_wrapper_propagates_energy_program_to_beam_reference():
    """A guest that advances particles.energy0 must update beam.reference."""
    new_energy0 = 1.2e9
    beam = _blond_beam([0.0], [0.0], 1e9)
    wrapper = WrapXsuite4Blond(_BumpReference(new_energy0, proton.mass))
    wrapper.track(beam)

    assert beam.reference.total_energy == pytest.approx(new_energy0, rel=1e-10)


def test_wrapper_follows_beam_size_change():
    """The guest Particles must always match the beam's slot count."""
    wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))
    wrapper.track(_blond_beam([0.0, 0.0], [0.0, 0.0], 1e9))
    assert wrapper._particles.zeta.shape == (2,)

    wrapper.track(_blond_beam([0.0, 0.0, 0.0, 0.0], [0.0] * 4, 1e9))
    assert wrapper._particles.zeta.shape == (4,)


def test_wrapper_propagates_reference_energy_increase():
    """An xsuite ReferenceEnergyIncrease guest must push the new reference into beam."""
    import xpart as _xp

    energy0 = 1e9
    delta_p0c = 5e6

    # Wrap a real xsuite element that advances the reference; the wrapper
    # should detect the new energy0 in its post-track check and update
    # beam.reference accordingly.
    ctx = _matched_context()
    line = xt.Line(
        elements=[xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c)],
        element_names=["accel"],
    )
    line.particle_ref = _xp.Particles(
        _context=ctx,
        p0c=_proton_p0c(energy0),
        mass0=proton.mass,
        q0=proton.charge,
    )
    line.build_tracker(_context=ctx)

    beam = _blond_beam([0.0], [0.0], energy0)
    # Line has no energy_program, so the construction guard does not fire.
    wrapper = WrapXsuite4Blond(line)
    wrapper.track(beam)

    expected_p0c = _proton_p0c(energy0) + delta_p0c
    expected_e0 = float(np.sqrt(expected_p0c**2 + proton.mass**2))
    assert beam.reference.total_energy == pytest.approx(expected_e0, rel=1e-10)


def test_wrapper_accepts_any_line_contents():
    """No introspection — a line wrapping any element just works."""
    # Drift, LineSegmentMap, … all just work once wrapped in a line.
    for element in (
        xt.Drift(length=1.0),
        xt.LineSegmentMap(
            longitudinal_mode="nonlinear",
            qx=1.1,
            qy=1.2,
            betx=1.0,
            bety=1.0,
            voltage_rf=0.0,
            frequency_rf=0.0,
            lag_rf=0.0,
            momentum_compaction_factor=1e-3,
            length=100.0,
        ),
    ):
        WrapXsuite4Blond(_one_element_line(element))  # must not raise


def test_wrapper_zero_length_line_has_zero_orbit_length():
    """A line with no spatial extent (a lone cavity) contributes zero orbit length."""
    cav = xt.Cavity(voltage=1e6, frequency=400e6, lag=0.0)
    w = WrapXsuite4Blond(_one_element_line(cav))
    assert w.orbit_length == 0.0


class TestGuestParticlesAreRebuiltEachPass(BLonDTestCase):
    """The wrapper must hand the guest line a fresh ``Particles`` each pass.

    Reusing one cached ``Particles`` leaks xsuite's internal bookkeeping
    across passes: writing ``state`` directly does not update xsuite's
    ``_num_active_particles`` / ``_num_lost_particles`` counters, so a slot
    lost on one pass stays outside the tracked range on every later pass —
    it is silently never tracked again, even once BLonD revives it.
    """

    def test_revived_slot_is_tracked_again_on_the_next_pass(self):
        # A drift moves ``dt`` only for particles it actually tracks, so a
        # changed ``dt`` is proof the guest saw the slot.
        energy0 = 1e9
        wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=2.5)))

        # Pass 1: BLonD declares slot 2 lost, so the wrapper kills that
        # xsuite slot before tracking.
        beam_a = _blond_beam([0.0] * 4, [1e6] * 4, energy0)
        beam_a.flags.array_local[2] = BeamFlags.LOST.value
        wrapper.track(beam_a)

        # Pass 2: a fresh, fully active beam of the same size.
        beam_b = _blond_beam([0.0] * 4, [1e6] * 4, energy0)
        wrapper.track(beam_b)

        dt_after = copy_to_cpu(beam_b.read_partial_dt())
        moved = dt_after != 0.0
        self.assertTrue(
            bool(moved.all()),
            f"slots {np.flatnonzero(~moved).tolist()} were not tracked by "
            f"the guest line; dt={dt_after.tolist()}",
        )

    def test_particles_are_not_reused_between_passes(self):
        beam = _blond_beam([0.0] * 3, [0.0] * 3, 1e9)
        wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))

        wrapper.track(beam)
        particles_after_first = wrapper._particles
        wrapper.track(beam)

        self.assertIsNot(wrapper._particles, particles_after_first)

    def test_guest_turn_counter_does_not_accumulate(self):
        """``at_turn`` must not creep up pass after pass.

        BLonD owns the turn loop; each pass through the guest is a single
        fresh track, so every pass must start the guest's own turn
        bookkeeping from zero.
        """
        beam = _blond_beam([0.0] * 3, [0.0] * 3, 1e9)
        wrapper = WrapXsuite4Blond(_one_element_line(xt.Drift(length=1.0)))

        wrapper.track(beam)
        at_turn_first = copy_to_cpu(wrapper._particles.at_turn).copy()
        wrapper.track(beam)
        at_turn_second = copy_to_cpu(wrapper._particles.at_turn)

        np.testing.assert_array_equal(at_turn_second, at_turn_first)


class TestReferenceProbeLostInGuest(BLonDTestCase):
    """A guest that loses the on-reference probe reports no advance."""

    def test_lost_probe_gives_zero_energy_advance(self):
        # The aperture excludes the reference orbit (x = 0), so the probe
        # is lost on its first element.
        energy0 = 1e9
        reference = _blond_beam([0.0], [0.0], energy0).reference
        aperture = xt.LimitRect(min_x=1e-3, max_x=2e-3)
        wrapper = WrapXsuite4Blond(_one_element_line(aperture))

        energy_delta = wrapper.track_reference(reference)

        self.assertEqual(energy_delta, 0.0)
        self.assertEqual(reference.total_energy, energy0)
