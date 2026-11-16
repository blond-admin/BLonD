# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration test for :class:`WrapXsuite4Blond` under a BLonD-owned energy ramp.

The harness here is the opposite of the BLonD-in-xsuite test: BLonD owns the
main loop and the energy program. The simulation is driven through
:class:`blond.Simulation` with a real :class:`MagneticCyclePerTurn` ramp; a
zero-voltage :class:`SingleHarmonicRFStation` is the BLonD-canonical way to
advance the reference each turn (its RF kick is null, but its
``track_reference`` step relabels the beam exactly as ``ReferenceEnergyChange``
would). The longitudinal drift physics comes from ``WrapXsuite4Blond(line)``
where ``line`` holds a single ``xt.LineSegmentMap``.

Reference (X): pure xsuite — a line holding a ``ReferenceEnergyIncrease``
followed by an ``xt.LineSegmentMap``, raising ``p0c`` by the ramp's constant
per-turn step at the start of every turn. An ``EnergyProgram`` is not used:
since xtrack 0.117 it ramps the reference continuously *within* a turn and
shifts ``zeta`` at each turn end by ``beta0 * c * t_sim - C``, which differs
from BLonD's turn-wise constant reference by ~3e-9 m per turn and also feeds
xtrack's floating-point clock into ``zeta``. ``ReferenceEnergyIncrease`` keeps
the conventional update (``zeta / beta0`` preserved), like BLonD's relabel.

The wrapper must push BLonD's *current* reference into its cached
``Particles`` on every call (see :meth:`WrapXsuite4Blond._track`); without
that, the xsuite guest reads stale ``beta0`` / ``p0c`` from build time and the
synchrotron amplitude drifts within a few turns.

Timing convention used here
---------------------------
* xsuite ``record_last_track.zeta[:, t]`` is the state *before* turn ``t``'s
  elements fire (turn 0 = initial state). After turn ``t``'s elements, the
  state appears at index ``t + 1``.
* BLonD's :class:`BeamObservationOncePerTurn` records *after* all elements of
  the turn have fired, so its index ``i`` corresponds to the same physical
  instant as xsuite's index ``i + 1``.

Element order
-------------
BLonD ring order is ``(rf_station, wrapper)``: the zero-voltage RF station runs
first to advance the reference at the *start* of the turn (mirroring the
``ReferenceEnergyIncrease`` at the start of the xsuite line), then the wrapper
drifts the beam at the new reference. Both runs start at ``p0c[0]`` and drift
turn ``i`` at ``p0c[i + 1]``.

Set ``DEV_PLOT = True`` to render comparison plots; ``plt.show()`` is gated on
the matplotlib backend so a headless CI run is a no-op.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
pytest.importorskip("xpart")

from blond import (  # noqa: E402
    Beam,
    BeamObservationOncePerTurn,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    copy_to_cpu,
    proton,
)
from blond.core.beam.flags import BeamFlags  # noqa: E402
from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)
from blond.interfaces.xsuite.elements.wrap_xsuite_element import (  # noqa: E402
    WrapXsuite4Blond,
)
from tests.integration.xsuite._lhc_harness import (  # noqa: E402
    CIRCUMFERENCE,
    HARMONIC,
    N_PART,
    assert_phase_space_match,
    build_common_line,
    build_line_segment_map,
    initial_distribution,
    plot_three,
    smooth_ramp,
)

N_TURNS = 2000

DEV_PLOT = False


def _build_wrapped_line() -> xt.Line:
    """One-element line holding the segment map — the wrapper's Line input."""
    return xt.Line(
        elements=[build_line_segment_map()], element_names=["matrix"]
    )


def _seed_blond_coords(p0c_ramp: np.ndarray, dist):
    """
    Build the initial BLonD ``(dt, dE)`` from xsuite's deterministic particles.

    Seeding from xsuite's own ``build_particles`` output at the initial
    reference makes the BLonD initial state match the pure-xsuite run bit for
    bit.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        ``(dt_init, dE_init, reference_total_energy)``.
    """
    seed = build_common_line(p0c_init=float(p0c_ramp[0])).build_particles(
        **dist
    )
    zeta_init = np.asarray(seed.zeta).copy()
    ptau_init = np.asarray(seed.ptau).copy()

    mass = float(proton.mass)
    e0 = float(np.sqrt(float(p0c_ramp[0]) ** 2 + mass**2))
    beta0 = float(p0c_ramp[0]) / e0
    frame = ReferenceFrame(beta0=beta0, energy0=e0)
    return zeta_to_dt(zeta_init, frame), ptau_to_dE(ptau_init, frame), e0


def _build_blond_owned_ring(n_turns: int, p0c_ramp: np.ndarray):
    """
    Assemble the BLonD ring + energy cycle for the wrapped-xsuite drift.

    A zero-voltage RF station advances the reference at the start of each turn
    (mirroring xsuite's between-turn ``ReferenceEnergyIncrease``); the wrapper
    then drifts the beam at the new reference.

    Returns
    -------
    tuple[Ring, MagneticCyclePerTurn]
        The assembled ring and the per-turn energy cycle.
    """
    # The wrapper is a black-box length-bearing element (not an RF station by
    # type), so it shares section 0 with the zero-voltage RF station without
    # affecting RF accounting: the energy ramp is not split between them and
    # the one-station-per-section check ignores it.
    ring = Ring(circumference=CIRCUMFERENCE)
    # Turn ``i`` advances the reference to ``p0c[i + 1]``, as the
    # ``ReferenceEnergyIncrease`` at the start of the pure-xsuite line does.
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p0c_ramp[0]),
        values_after_turn=np.asarray(p0c_ramp[1 : n_turns + 2], dtype=float),
        in_unit="momentum",
    )
    rf_station = SingleHarmonicRFStation(
        voltage=0.0, phi_rf=0.0, harmonic=HARMONIC
    )
    wrapper = WrapXsuite4Blond(_build_wrapped_line())
    ring.add_elements((rf_station, wrapper), reorder=False, section_index=0)
    return ring, energy_cycle


def _run_pure_xsuite(p0c_ramp: np.ndarray, n_turns: int, dist):
    """Track ``n_turns + 1`` so every BLonD turn has a matching xsuite snapshot."""
    line = build_common_line(p0c_init=float(p0c_ramp[0]))
    p0c_steps = np.diff(p0c_ramp)
    # One constant step per turn: ``ReferenceEnergyIncrease`` is static.
    np.testing.assert_allclose(p0c_steps, p0c_steps[0], rtol=1e-6)
    line.insert(
        "energy_increase",
        obj=xt.ReferenceEnergyIncrease(Delta_p0c=float(p0c_steps[0])),
        at=0,
    )
    line.build_tracker()

    particles = line.build_particles(**dist)
    line.track(
        particles=particles,
        num_turns=n_turns + 1,
        turn_by_turn_monitor=True,
    )
    # rec[:, t] = state before xsuite turn t. End-of-turn N for N in
    # 0..n_turns-1 lives at rec[:, 1 .. n_turns].
    rec = line.record_last_track
    return (
        rec.zeta[:, 1 : n_turns + 1].copy(),
        rec.ptau[:, 1 : n_turns + 1].copy(),
    )


def _run_blond_simulation(p0c_ramp: np.ndarray, n_turns: int, dist):
    """BLonD-owned ramp: voltage-0 RF station advances reference, wrapper drifts."""
    dt_init, dE_init, e0 = _seed_blond_coords(p0c_ramp, dist)
    ring, energy_cycle = _build_blond_owned_ring(n_turns, p0c_ramp)

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=dt_init.copy(),
        dE=dE_init.copy(),
        reference_total_energy=e0,
    )

    bunch_obs = BeamObservationOncePerTurn(each_turn_i=1, warn=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.run_simulation(beams=(beam,), n_turns=n_turns, observe=(bunch_obs,))

    # Recorded shape is (n_turns, n_part). Convert each row at its own
    # reference using the shared helpers.
    dts = copy_to_cpu(bunch_obs.dts)
    dEs = copy_to_cpu(bunch_obs.dEs)
    ref_E = copy_to_cpu(bunch_obs.reference_total_energy)
    mass = float(proton.mass)
    ref_p0c = np.sqrt(ref_E**2 - mass**2)
    ref_beta = ref_p0c / ref_E

    zeta = np.empty_like(dts)
    ptau = np.empty_like(dEs)
    for turn in range(n_turns):
        frame_t = ReferenceFrame(
            beta0=float(ref_beta[turn]), energy0=float(ref_E[turn])
        )
        zeta[turn] = dt_to_zeta(dts[turn], frame_t)
        ptau[turn] = dE_to_ptau(dEs[turn], frame_t)
    return zeta.T, ptau.T


class TestBlondOwnedRamp(unittest.TestCase):
    """An xsuite drift wrapped in a BLonD-owned ramp must match pure xsuite."""

    def test_matches_pure_xsuite_drift(self):
        """BLonD-owned ramp through WrapXsuite4Blond(line) matches pure xsuite."""
        dist = initial_distribution()
        p0c_ramp = smooth_ramp(N_TURNS)

        zeta_x, ptau_x = _run_pure_xsuite(p0c_ramp, N_TURNS, dist)
        zeta_b, ptau_b = _run_blond_simulation(p0c_ramp, N_TURNS, dist)

        if DEV_PLOT:
            plot_three(zeta_x, ptau_x, zeta_b, ptau_b, "wrapped xsuite")

        # With the reference stepped once per turn on both sides, the
        # BLonD-owned ramp should match pure xsuite to floating-point
        # round-off. The tight tolerance
        # guards against reintroducing a per-turn reference offset -- e.g. an
        # ultra-relativistic ``C/c`` program axis (~1e-6) or an off-by-one in
        # ``values_after_turn`` (a one-step ~5.5e-8 ptau jump).
        assert_phase_space_match(
            zeta_b, ptau_b, zeta_x, ptau_x, N_TURNS, rtol=1e-7
        )

    def test_lost_particles_survive_multi_turn_track(self):
        """Particles flagged LOST at setup stay untouched across many turns.

        Multi-turn extension of ``test_xsuite_particles_to_blond_beam_flags_lost`` (unit
        test): the wrapper must keep honoring the LOST flag across a full BLonD
        ``Simulation.run_simulation``, not just once. Otherwise a lost slot
        could be silently re-activated by the wrapper's coordinate write-back or
        by a stale-cache push of ``update_p0c``.
        """
        n_turns = 50
        p0c_ramp = smooth_ramp(n_turns)
        dist = initial_distribution()

        # Mark every other slot lost up front.
        lost_mask = np.zeros(N_PART, dtype=bool)
        lost_mask[::2] = True

        dt_init, dE_init, e0 = _seed_blond_coords(p0c_ramp, dist)
        flags = np.where(
            lost_mask,
            np.int32(BeamFlags.LOST.value),
            np.int32(BeamFlags.ACTIVE.value),
        )

        ring, energy_cycle = _build_blond_owned_ring(n_turns, p0c_ramp)

        beam = Beam(intensity=1.0, particle_type=proton)
        beam.setup_beam(
            dt=dt_init.copy(),
            dE=dE_init.copy(),
            flags=flags.copy(),
            reference_total_energy=e0,
        )

        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
        sim.run_simulation(beams=(beam,), n_turns=n_turns)

        final_flags = copy_to_cpu(beam.read_partial_flags())
        final_dt = copy_to_cpu(beam.read_partial_dt())

        # Lost slots: flag preserved + dt bit-identical (the wrapper only writes
        # active slots back into the beam). dE is NOT checked because BLonD's
        # zero-voltage RF station applies the per-turn ``acceleration_kick`` to
        # every slot in ``beam.dE`` regardless of flag — a BLonD-core behavior
        # the wrapper has no jurisdiction over (see discussion note item N).
        np.testing.assert_array_equal(
            final_flags[lost_mask], BeamFlags.LOST.value
        )
        np.testing.assert_array_equal(final_dt[lost_mask], dt_init[lost_mask])
        # Active slots: still active and at least some moved (so we know the
        # wrapper actually ran).
        np.testing.assert_array_equal(
            final_flags[~lost_mask], BeamFlags.ACTIVE.value
        )
        self.assertFalse(
            np.array_equal(final_dt[~lost_mask], dt_init[~lost_mask])
        )

    def test_all_particles_lost_does_not_crash(self):
        """A beam that is entirely lost must track through the wrapper safely.

        The wrapper's ``_track`` gathers/scatters through the active mask; with
        no active slots that mask is empty, which must not raise (no empty
        ``argmax``, no zero-length write going wrong) and must leave every slot
        flagged LOST.
        """
        n_turns = 5
        p0c_ramp = smooth_ramp(n_turns)
        dist = initial_distribution()

        dt_init, dE_init, e0 = _seed_blond_coords(p0c_ramp, dist)
        flags = np.full(N_PART, np.int32(BeamFlags.LOST.value))

        ring, energy_cycle = _build_blond_owned_ring(n_turns, p0c_ramp)

        beam = Beam(intensity=1.0, particle_type=proton)
        beam.setup_beam(
            dt=dt_init.copy(),
            dE=dE_init.copy(),
            flags=flags.copy(),
            reference_total_energy=e0,
        )

        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
        sim.run_simulation(beams=(beam,), n_turns=n_turns)  # must not raise

        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_flags()), BeamFlags.LOST.value
        )


if __name__ == "__main__":
    unittest.main()
