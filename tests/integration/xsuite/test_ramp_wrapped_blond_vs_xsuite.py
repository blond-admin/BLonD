# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Cross-framework integration test for the BLonD↔xsuite element wrapper.

Same LHC-like lattice and same energy ramp are tracked two ways:

* Reference (X): a pure ``xt.Cavity`` whose frequency is locked to the ramp's
  per-turn revolution frequency.
* Under test (W): the same physical cavity expressed as a
  ``SingleHarmonicRFStation`` (with ``magnetic_cycle=None`` — xsuite owns the
  reference) wrapped by :class:`WrapBlond4Xsuite` and inserted at the same
  position in the line.

The two runs must agree on the longitudinal phase space turn-by-turn. The
phase conventions reconcile to ``phi_rf [rad] = lag [deg] · π/180``; here we
deliberately pick a non-trivial ``phi_rf = π/4`` (= ``lag = 45°``) so a
sign-flip or unit-confusion bug cannot accidentally pass.

Set ``DEV_PLOT = True`` to render comparison plots when running the file
directly (no effect inside pytest's headless run).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
xp = pytest.importorskip("xpart")

from blond import SingleHarmonicRFStation  # noqa: E402
from blond.interfaces.xsuite.elements.wrap_blond_elelemt import (  # noqa: E402
    WrapBlond4Xsuite,
)


# LHC-like ring & program parameters.
CIRCUMFERENCE = 26658.8832
ALPHA = 0.00034849575112251314
HARMONIC = 35640
RF_VOLTAGE = 5e6
P0C_INIT = 450e9
P0C_FINAL = 450.05e9
N_TURNS = 200
N_PART = 16
SEED = 0

# Non-trivial phase pair: an off-by-π/2 or deg-vs-rad mistake would visibly
# break the comparison. phi_rf [rad] = lag [deg] · π/180.
PHI_RF = np.pi / 4
LAG_DEG = 45.0

DEV_PLOT = True # TODO false  # set True to render comparison plots; pytest
# leaves this off


def _build_common_line(p0c_init: float = P0C_INIT) -> xt.Line:
    matrix = xt.LineSegmentMap(
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
    line = xt.Line(elements=[matrix], element_names=["matrix"])
    line.particle_ref = xp.Particles(
        p0c=p0c_init, mass0=xp.PROTON_MASS_EV, q0=1.0
    )
    return line


def _attach_energy_program(line: xt.Line, n_turns: int) -> None:
    n_program = n_turns + 2  # margin past the last tracked turn
    t_rev = CIRCUMFERENCE / c
    p0c_ramp = np.linspace(P0C_INIT, P0C_FINAL, n_program)
    t_s = np.linspace(0.0, t_rev * (n_program - 1), n_program)
    line.energy_program = xt.EnergyProgram(t_s=t_s, p0c=p0c_ramp)


def _initial_distribution(line: xt.Line) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        "x": rng.uniform(-1e-3, 1e-3, N_PART),
        "px": rng.uniform(-1e-5, 1e-5, N_PART),
        "y": rng.uniform(-2e-3, 2e-3, N_PART),
        "py": rng.uniform(-3e-5, 3e-5, N_PART),
        "zeta": np.linspace(-0.5, 0.5, N_PART),
        "delta": np.linspace(-1e-4, 1e-4, N_PART),
    }


def _build_particles(line: xt.Line, dist: dict[str, np.ndarray]):
    return line.build_particles(**dist)


def _run_pure_xsuite(dist: dict[str, np.ndarray]):
    line = _build_common_line()
    _attach_energy_program(line, N_TURNS)

    # Cavity frequency tracks the ramp's per-turn revolution frequency.
    t_rev = CIRCUMFERENCE / c
    n_program = N_TURNS + 2
    t_rf = np.linspace(0.0, t_rev * (n_program - 1), n_program)
    f_rev = line.energy_program.get_frev_at_t_s(t_rf)
    f_rf = HARMONIC * f_rev

    cavity = xt.Cavity(
        voltage=RF_VOLTAGE, frequency=float(f_rf[0]), lag=LAG_DEG
    )
    line.insert("rf_cavity", obj=cavity, at=0)

    line.functions["fun_f_rf"] = xt.FunctionPieceWiseLinear(x=t_rf, y=f_rf)
    line["rf_cavity"].frequency = line.functions["fun_f_rf"](
        line.ref["t_turn_s"]
    )

    line.enable_time_dependent_vars = True
    line.build_tracker()

    particles = _build_particles(line, dist)
    line.track(
        particles=particles, num_turns=N_TURNS, turn_by_turn_monitor=True
    )
    return (
        line.record_last_track.zeta.copy(),
        line.record_last_track.ptau.copy(),
    )


def _run_wrapped_blond(dist: dict[str, np.ndarray]):
    line = _build_common_line()
    _attach_energy_program(line, N_TURNS)

    beta0_init = float(line.particle_ref.beta0[0])
    blond_cavity = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=RF_VOLTAGE,
        phi_rf=PHI_RF,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        beam_reference_beta=beta0_init,
        magnetic_cycle=None,
    )
    wrapped = WrapBlond4Xsuite(element=blond_cavity)
    line.insert("rf_cavity", obj=wrapped, at=0)

    line.enable_time_dependent_vars = True
    line.build_tracker()

    particles = _build_particles(line, dist)
    line.track(
        particles=particles, num_turns=N_TURNS, turn_by_turn_monitor=True
    )
    return (
        line.record_last_track.zeta.copy(),
        line.record_last_track.ptau.copy(),
    )


def _plot_comparison(zeta_x, ptau_x, zeta_w, ptau_w) -> None:
    """Render comparison plots (DEV_PLOT only). Side-effect heavy, opens windows."""
    import matplotlib.pyplot as plt

    snapshots = [0, N_TURNS // 2, N_TURNS - 1]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    for ax, turn in zip(axes, snapshots):
        ax.scatter(zeta_x[:, turn], ptau_x[:, turn], label="xsuite", marker="x")
        ax.scatter(
            zeta_w[:, turn], ptau_w[:, turn], label="wrapped BLonD", marker="o",
            facecolors="none", edgecolors="C1",
        )
        ax.set_xlabel(r"$\zeta$ [m]")
        ax.set_ylabel(r"$p_\tau$")
        ax.set_title(f"turn {turn}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig2, (ax_z, ax_p) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    turns = np.arange(N_TURNS)
    dz_max = np.max(np.abs(zeta_w - zeta_x), axis=0)
    dp_max = np.max(np.abs(ptau_w - ptau_x), axis=0)
    ax_z.semilogy(turns, dz_max + 1e-300)
    ax_z.set_ylabel(r"max $|\Delta \zeta|$ [m]")
    ax_z.grid(True, which="both", alpha=0.3)
    ax_p.semilogy(turns, dp_max + 1e-300)
    ax_p.set_ylabel(r"max $|\Delta p_\tau|$")
    ax_p.set_xlabel("turn")
    ax_p.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    plt.show()


def test_wrapped_blond_matches_pure_xsuite_under_ramp():
    """Wrapped BLonD cavity tracks identically to a pure xt.Cavity under a ramp."""
    dist = _initial_distribution(_build_common_line())

    zeta_x, ptau_x = _run_pure_xsuite(dist)
    zeta_w, ptau_w = _run_wrapped_blond(dist)

    if DEV_PLOT:
        _plot_comparison(zeta_x, ptau_x, zeta_w, ptau_w)

    assert zeta_w.shape == zeta_x.shape
    assert ptau_w.shape == ptau_x.shape

    # Compare each turn against the xsuite reference. zeta crosses zero so use
    # an amplitude-scaled atol; the synchrotron amplitude grows during the
    # ramp, hence the per-turn rescaling.
    for turn in range(N_TURNS):
        zeta_amp = float(np.max(np.abs(zeta_x[:, turn])))
        ptau_amp = float(np.max(np.abs(ptau_x[:, turn])))
        np.testing.assert_allclose(
            zeta_w[:, turn],
            zeta_x[:, turn],
            atol=max(zeta_amp * 1e-5, 1e-12),
            rtol=0.0,
            err_msg=f"zeta mismatch on turn {turn}",
        )
        np.testing.assert_allclose(
            ptau_w[:, turn],
            ptau_x[:, turn],
            atol=max(ptau_amp * 1e-5, 1e-15),
            rtol=0.0,
            err_msg=f"ptau mismatch on turn {turn}",
        )
