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
``track_reference`` step relabels the beam exactly as
``ReferenceEnergyChange`` would). The longitudinal drift physics comes from
``WrapXsuite4Blond(xt.LineSegmentMap)``.

Reference (X): pure xsuite — a line containing only an ``xt.LineSegmentMap``
plus an ``EnergyProgram`` that ramps ``p0c`` turn-by-turn.

The wrapper must push BLonD's *current* reference into its cached
``Particles`` on every call (see :meth:`WrapXsuite4Blond._track`); without
that, the xsuite guest reads stale ``β₀`` / ``p0c`` from build time and the
synchrotron amplitude drifts within a few turns.

Timing convention used here
---------------------------
* xsuite ``record_last_track.zeta[:, t]`` is the state *before* turn ``t``'s
  elements fire (turn 0 = initial state). After turn ``t``'s elements, the
  state appears at index ``t + 1``.
* BLonD's :class:`BeamObservationOncePerTurn` records *after* all elements
  of the turn have fired, so its index ``i`` corresponds to the same
  physical instant as xsuite's index ``i + 1``.

Element order
-------------
BLonD ring order is ``(rf_station, wrapper)``: the zero-voltage RF station
runs first to advance the reference at the *start* of the turn (mirroring
xsuite's between-turn ``ReferenceEnergyIncrease``), then the wrapper drifts
the beam at the new reference. To keep turn 0's reference equal to
``p0c[0]`` (matching xsuite), the magnetic cycle's ``value_init`` is the
same as ``values_after_turn[0]``, which makes turn 0's relabel a no-op.

Set ``DEV_PLOT = True`` to render comparison plots; ``plt.show()`` is
gated on the matplotlib backend so a headless CI run is a no-op.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
xp = pytest.importorskip("xpart")

from blond import (  # noqa: E402
    Beam,
    BeamObservationOncePerTurn,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.interfaces.xsuite.elements.helpers import (  # noqa: E402
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)
from blond.interfaces.xsuite.elements.wrap_xsuite_elelemt import (  # noqa: E402
    WrapXsuite4Blond,
)

# LHC-like harness; mirrors test_ramp_wrapped_blond_vs_xsuite.py.
CIRCUMFERENCE = 26658.8832
ALPHA = 0.00034849575112251314
HARMONIC = 35640
P0C_INIT = 450e9
P0C_FINAL = 450.05e9
N_TURNS = 2000
N_PART = 16
SEED = 0

DEV_PLOT = False


def _build_line_segment_map() -> xt.LineSegmentMap:
    return xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0.0,
        frequency_rf=0.0,
        lag_rf=0.0,
        momentum_compaction_factor=ALPHA,
        length=CIRCUMFERENCE,
    )


def _build_common_line(p0c_init: float) -> xt.Line:
    line = xt.Line(
        elements=[_build_line_segment_map()], element_names=["matrix"]
    )
    line.particle_ref = xp.Particles(
        p0c=p0c_init, mass0=xp.PROTON_MASS_EV, q0=1.0
    )
    return line


def _attach_energy_program(line: xt.Line, p0c_ramp: np.ndarray) -> None:
    t_rev = CIRCUMFERENCE / c
    n_program = len(p0c_ramp)
    t_s = np.linspace(0.0, t_rev * (n_program - 1), n_program)
    line.energy_program = xt.EnergyProgram(
        t_s=t_s, p0c=np.asarray(p0c_ramp, dtype=float)
    )


def _initial_distribution() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        "x": rng.uniform(-1e-3, 1e-3, N_PART),
        "px": rng.uniform(-1e-5, 1e-5, N_PART),
        "y": rng.uniform(-2e-3, 2e-3, N_PART),
        "py": rng.uniform(-3e-5, 3e-5, N_PART),
        "zeta": np.linspace(-0.5, 0.5, N_PART),
        "delta": np.linspace(-1e-4, 1e-4, N_PART),
    }


def _smooth_ramp(n_turns: int) -> np.ndarray:
    # n_turns + 2 entries: turn-0 reference + one node per simulated turn-end
    # + a margin so the program never falls off the right edge.
    return np.linspace(P0C_INIT, P0C_FINAL, n_turns + 2)


def _run_pure_xsuite(p0c_ramp: np.ndarray, n_turns: int, dist):
    """Track ``n_turns + 1`` so every BLonD turn has a matching xsuite snapshot."""
    line = _build_common_line(p0c_init=float(p0c_ramp[0]))
    _attach_energy_program(line, p0c_ramp)
    line.enable_time_dependent_vars = True
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
    # Seed initial BLonD coords from xsuite's deterministic build_particles
    # output at the initial reference so the initial state matches the
    # pure-xsuite run bit for bit.
    seed_line = _build_common_line(p0c_init=float(p0c_ramp[0]))
    seed = seed_line.build_particles(**dist)
    zeta_init = np.asarray(seed.zeta).copy()
    ptau_init = np.asarray(seed.ptau).copy()

    mass = float(proton.mass)
    e0 = float(np.sqrt(float(p0c_ramp[0]) ** 2 + mass**2))
    beta0_0 = float(p0c_ramp[0]) / e0
    init_frame = ReferenceFrame(beta0=beta0_0, energy0=e0)
    dt_init = zeta_to_dt(zeta_init, init_frame)
    dE_init = ptau_to_dE(ptau_init, init_frame)

    # The wrapper is an RFStationBaseClass by type, so it shares section 0
    # with the real zero-voltage RF station. It opts out of RF-station
    # *accounting* (``counts_as_rf_station = False``), so the energy ramp is
    # not split between them; but the element container's section-indexing
    # check still flags two RF-typed elements in one section, so we disable
    # it for this black-box setup.
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    # ``values_after_turn`` is one entry per simulated turn. We re-use
    # ``p0c[0]`` for the slot the cycle reads on turn 0 — that makes turn
    # 0's reference advance a no-op, leaving the beam at the same reference
    # as the pure-xsuite run sees at the start of its turn 0.
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p0c_ramp[0]),
        values_after_turn=np.asarray(p0c_ramp[: n_turns + 1], dtype=float),
        in_unit="momentum",
    )

    # rf_station first — it advances the reference at the start of the turn,
    # mirroring xsuite's between-turn ``ReferenceEnergyIncrease`` timing.
    rf_station = SingleHarmonicRFStation(
        voltage=0.0, phi_rf=0.0, harmonic=HARMONIC
    )
    wrapper = WrapXsuite4Blond(_build_line_segment_map())
    ring.add_elements((rf_station, wrapper), reorder=False, section_index=0)

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
    dts = np.asarray(bunch_obs.dts)
    dEs = np.asarray(bunch_obs.dEs)
    ref_E = np.asarray(bunch_obs.reference_total_energy)
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


def _plot_comparison(zeta_x, ptau_x, zeta_b, ptau_b) -> None:
    """Render comparison plots (DEV_PLOT only). No-op under headless backends."""
    import matplotlib
    import matplotlib.pyplot as plt

    n_turns = zeta_x.shape[1]
    snapshots = sorted({0, n_turns // 2, n_turns - 1})
    fig, axes = plt.subplots(
        1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False
    )
    for ax, turn in zip(axes[0], snapshots):
        ax.scatter(
            zeta_x[:, turn], ptau_x[:, turn], label="xsuite", marker="x"
        )
        ax.scatter(
            zeta_b[:, turn],
            ptau_b[:, turn],
            label="BLonD-owned + wrapper",
            marker="o",
            facecolors="none",
            edgecolors="C1",
        )
        ax.set_xlabel(r"$\zeta$ [m]")
        ax.set_ylabel(r"$p_\tau$")
        ax.set_title(f"turn {turn}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig2, (ax_z, ax_p) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    turns = np.arange(n_turns)
    dz_max = np.max(np.abs(zeta_b - zeta_x), axis=0)
    dp_max = np.max(np.abs(ptau_b - ptau_x), axis=0)
    ax_z.semilogy(turns, dz_max + 1e-300)
    ax_z.set_ylabel(r"max $|\Delta \zeta|$ [m]")
    ax_z.grid(True, which="both", alpha=0.3)
    ax_p.semilogy(turns, dp_max + 1e-300)
    ax_p.set_ylabel(r"max $|\Delta p_\tau|$")
    ax_p.set_xlabel("turn")
    ax_p.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()

    # Only block on show() under an interactive backend; under "agg" (CI)
    # the figures stay in memory and are GC'd at end of test.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def _assert_phase_space_match(
    zeta_b, ptau_b, zeta_x, ptau_x, n_turns, rtol=1e-5
):
    assert zeta_b.shape == zeta_x.shape
    assert ptau_b.shape == ptau_x.shape
    for turn in range(n_turns):
        zeta_amp = float(np.max(np.abs(zeta_x[:, turn])))
        ptau_amp = float(np.max(np.abs(ptau_x[:, turn])))
        np.testing.assert_allclose(
            zeta_b[:, turn],
            zeta_x[:, turn],
            atol=max(zeta_amp * rtol, 1e-12),
            rtol=0.0,
            err_msg=f"zeta mismatch on turn {turn}",
        )
        np.testing.assert_allclose(
            ptau_b[:, turn],
            ptau_x[:, turn],
            atol=max(ptau_amp * rtol, 1e-15),
            rtol=0.0,
            err_msg=f"ptau mismatch on turn {turn}",
        )


def test_blond_owned_ramp_matches_pure_xsuite_drift():
    """BLonD-owned ramp through WrapXsuite4Blond(LineSegmentMap) matches pure xsuite."""
    dist = _initial_distribution()
    p0c_ramp = _smooth_ramp(N_TURNS)

    zeta_x, ptau_x = _run_pure_xsuite(p0c_ramp, N_TURNS, dist)
    zeta_b, ptau_b = _run_blond_simulation(p0c_ramp, N_TURNS, dist)

    if DEV_PLOT:
        _plot_comparison(zeta_x, ptau_x, zeta_b, ptau_b)

    _assert_phase_space_match(zeta_b, ptau_b, zeta_x, ptau_x, N_TURNS)


def test_lost_particles_survive_multi_turn_wrapper_track():
    """Particles flagged LOST at setup must stay untouched across many turns.

    This is the multi-turn extension of
    ``test_particles_to_beam_flags_lost`` (unit test): the wrapper must not
    only honor the LOST flag once, it must keep doing so across a full
    BLonD ``Simulation.run_simulation``. Otherwise a lost slot could be
    silently re-activated by the wrapper's coordinate write-back or by a
    stale-cache push of ``update_p0c``.
    """
    from blond.core.beam.flags import BeamFlags

    p0c_ramp = _smooth_ramp(50)
    dist = _initial_distribution()

    # Mark every other slot lost up front.
    lost_mask = np.zeros(N_PART, dtype=bool)
    lost_mask[::2] = True

    seed_line = _build_common_line(p0c_init=float(p0c_ramp[0]))
    seed = seed_line.build_particles(**dist)
    zeta_init = np.asarray(seed.zeta).copy()
    ptau_init = np.asarray(seed.ptau).copy()

    mass = float(proton.mass)
    e0 = float(np.sqrt(float(p0c_ramp[0]) ** 2 + mass**2))
    beta0_0 = float(p0c_ramp[0]) / e0
    frame_init = ReferenceFrame(beta0=beta0_0, energy0=e0)
    dt_init = zeta_to_dt(zeta_init, frame_init)
    dE_init = ptau_to_dE(ptau_init, frame_init)

    flags = np.where(
        lost_mask,
        np.int32(BeamFlags.LOST.value),
        np.int32(BeamFlags.ACTIVE.value),
    )

    # The wrapper is an RFStationBaseClass by type, so it shares section 0
    # with the real zero-voltage RF station. It opts out of RF-station
    # *accounting* (``counts_as_rf_station = False``), so the energy ramp is
    # not split between them; but the element container's section-indexing
    # check still flags two RF-typed elements in one section, so we disable
    # it for this black-box setup.
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p0c_ramp[0]),
        values_after_turn=np.asarray(p0c_ramp[:51], dtype=float),
        in_unit="momentum",
    )
    rf_station = SingleHarmonicRFStation(
        voltage=0.0, phi_rf=0.0, harmonic=HARMONIC
    )
    wrapper = WrapXsuite4Blond(_build_line_segment_map())
    ring.add_elements((rf_station, wrapper), reorder=False, section_index=0)

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=dt_init.copy(),
        dE=dE_init.copy(),
        flags=flags.copy(),
        reference_total_energy=e0,
    )

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.run_simulation(beams=(beam,), n_turns=50)

    final_flags = beam.read_partial_flags()
    final_dt = beam.read_partial_dt()
    final_dE = beam.read_partial_dE()

    # Lost slots: flag preserved + dt bit-identical (the wrapper only writes
    # active slots back into the beam). dE is NOT checked because BLonD's
    # zero-voltage RF station applies the per-turn ``acceleration_kick`` to
    # every slot in ``beam.dE`` regardless of flag — a BLonD-core behavior
    # the wrapper has no jurisdiction over (see discussion note item N).
    np.testing.assert_array_equal(final_flags[lost_mask], BeamFlags.LOST.value)
    np.testing.assert_array_equal(final_dt[lost_mask], dt_init[lost_mask])
    # Active slots: still active and at least some moved (so we know the
    # wrapper actually ran).
    np.testing.assert_array_equal(
        final_flags[~lost_mask], BeamFlags.ACTIVE.value
    )
    assert not np.array_equal(final_dt[~lost_mask], dt_init[~lost_mask])
