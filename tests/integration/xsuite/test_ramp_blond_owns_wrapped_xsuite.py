# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration test for :class:`WrapXsuite4Blond` under a BLonD-owned energy ramp.

The harness in this direction is the opposite of the BLonD-in-xsuite test:
BLonD owns the main loop and the energy program. The simulation is driven
through :class:`blond.Simulation` with a real :class:`MagneticCyclePerTurn`
ramp; a zero-voltage :class:`SingleHarmonicRFStation` is the BLonD-canonical
way to advance the reference each turn (its RF kick is null, but its
``track_reference`` step relabels the beam exactly as
``ReferenceEnergyChange`` would). The longitudinal drift physics comes from
``WrapXsuite4Blond(xt.LineSegmentMap)``.

Reference (X): pure xsuite, a line containing only an ``xt.LineSegmentMap``
plus an ``EnergyProgram`` that ramps ``p0c`` turn-by-turn.

The wrapper must push BLonD's *current* reference into its cached
``Particles`` on every call. Without that the xsuite guest reads stale
``beta0`` / ``p0c`` from build time and the synchrotron amplitude drifts
within a few turns.

Timing note: xsuite's ``record_last_track.zeta[:, t]`` is the state at the
*start* of turn ``t``; BLonD's ``BeamObservationOncePerTurn`` records at the
*end* of turn ``t``. So BLonD obs index ``i`` is compared against xsuite
record index ``i + 1``.

Set ``DEV_PLOT = True`` to render comparison plots when running directly.
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
from blond.interfaces.xsuite.elements.wrap_xsuite_elelemt import (  # noqa: E402
    WrapXsuite4Blond,
)


# LHC-like harness; mirrors test_ramp_wrapped_blond_vs_xsuite.py.
CIRCUMFERENCE = 26658.8832
ALPHA = 0.00034849575112251314
HARMONIC = 35640
P0C_INIT = 450e9
P0C_FINAL = 450.05e9
N_TURNS = 200
N_PART = 16
SEED = 0

DEV_PLOT = True  # set True to render comparison plots


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
    # n_turns + 2 entries: turn-0 reference + one node per simulated turn-end +
    # a margin so the program never falls off the right edge.
    return np.linspace(P0C_INIT, P0C_FINAL, n_turns + 2)


def _run_pure_xsuite(p0c_ramp: np.ndarray, n_turns: int, dist):
    """Track ``n_turns + 1`` so we have an end-of-turn snapshot for every BLonD turn."""
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
    # rec[:, t] = state at start of xsuite turn t; we want end-of-turn N for
    # N in 0..n_turns-1, which corresponds to rec[:, 1..n_turns].
    rec = line.record_last_track
    return rec.zeta[:, 1 : n_turns + 1].copy(), rec.ptau[:, 1 : n_turns + 1].copy()


def _run_blond_simulation(p0c_ramp: np.ndarray, n_turns: int, dist):
    """Run BLonD-owned simulation: voltage-0 RF station advances reference, wrapper drifts."""
    # Seed initial BLonD coords from xsuite's deterministic build_particles
    # output at the initial reference, so the initial state matches the
    # pure-xsuite run bit for bit.
    seed_line = _build_common_line(p0c_init=float(p0c_ramp[0]))
    seed = seed_line.build_particles(**dist)
    zeta_init = np.asarray(seed.zeta).copy()
    ptau_init = np.asarray(seed.ptau).copy()

    mass = float(proton.mass)
    e0 = float(np.sqrt(float(p0c_ramp[0]) ** 2 + mass**2))
    beta0_0 = float(p0c_ramp[0]) / e0
    dt_init = -zeta_init / (beta0_0 * c)
    dE_init = ptau_init * beta0_0 * e0

    ring = Ring(circumference=CIRCUMFERENCE)
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=float(p0c_ramp[0]),
        values_after_turn=np.asarray(p0c_ramp[1:], dtype=float),
        in_unit="momentum",
    )

    # Drift first (matches xsuite's "advance reference between turns" order),
    # then a zero-voltage RF station to advance the BLonD reference.
    wrapper = WrapXsuite4Blond(_build_line_segment_map())
    rf_station = SingleHarmonicRFStation(
        voltage=0.0, phi_rf=0.0, harmonic=HARMONIC
    )
    ring.add_elements((wrapper, rf_station), reorder=False, section_index=0)

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=dt_init.copy(), dE=dE_init.copy(), reference_total_energy=e0
    )

    bunch_obs = BeamObservationOncePerTurn(each_turn_i=1, warn=False)

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.run_simulation(
        beams=(beam,), n_turns=n_turns, observe=(bunch_obs,)
    )

    dts = np.asarray(bunch_obs.dts)  # shape (n_turns, n_part)
    dEs = np.asarray(bunch_obs.dEs)
    ref_E = np.asarray(bunch_obs.reference_total_energy)  # shape (n_turns,)
    ref_p0c = np.sqrt(ref_E**2 - mass**2)
    ref_beta = ref_p0c / ref_E

    # Convert (dt, dE) at each turn's reference back to (zeta, ptau) and shape
    # (n_part, n_turns) to match the xsuite record layout.
    zeta = -dts * ref_beta[:, None] * c
    ptau = dEs / (ref_beta[:, None] * ref_E[:, None])
    return zeta.T, ptau.T


def _plot_comparison(zeta_x, ptau_x, zeta_b, ptau_b) -> None:
    """Render comparison plots (DEV_PLOT only). Side-effect heavy, opens windows."""
    import matplotlib.pyplot as plt

    n_turns = zeta_x.shape[1]
    snapshots = sorted({0, n_turns // 2, n_turns - 1})
    fig, axes = plt.subplots(
        1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False
    )
    for ax, turn in zip(axes[0], snapshots):
        ax.scatter(zeta_x[:, turn], ptau_x[:, turn], label="xsuite", marker="x")
        ax.scatter(
            zeta_b[:, turn], ptau_b[:, turn], label="BLonD-owned + wrapper",
            marker="o", facecolors="none", edgecolors="C1",
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
