# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Shared LHC-like harness for the BLonD<->xsuite ramp integration tests.

Both ramp tests track the same LHC-like lattice through the same energy
program, just from opposite sides of the wrapper (BLonD inside an xsuite
``Line`` vs. an xsuite ``Line`` inside a BLonD ring). This module holds the
pieces they have in common — the ring/program constants, the line and
energy-program builders, the deterministic initial distribution, and the
turn-by-turn phase-space comparison — so each test file only carries what is
unique to its tracking direction.

The leading underscore keeps pytest from collecting this module as a test.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

xt = pytest.importorskip("xtrack")
xp = pytest.importorskip("xpart")
xo = pytest.importorskip("xobjects")

from blond import copy_to_cpu  # noqa: E402

_GPU_CONTEXT = None


def matched_context():
    """
    Return an xobjects context matching BLonD's *currently active* backend.

    Tests must pass under whatever process-global backend an earlier test
    left behind (the suite runs in random order); building trackers and
    particles on the matching context keeps the wrappers' context guards
    satisfied on CPU and GPU alike. Contexts are shared (the xobjects
    default on CPU, one process-wide ``ContextCupy`` on GPU) so xtrack's
    per-context kernel caches are reused across tests.
    """
    from blond import backend

    if backend.is_gpu:
        global _GPU_CONTEXT
        if _GPU_CONTEXT is None:
            _GPU_CONTEXT = xo.ContextCupy()
        return _GPU_CONTEXT
    return xo.context_default


# LHC-like ring & energy-program parameters shared by both ramp tests.
CIRCUMFERENCE = 26658.8832
ALPHA = 0.00034849575112251314
HARMONIC = 35640
P0C_INIT = 450e9
P0C_FINAL = 450.05e9
N_PART = 16
SEED = 0


def build_line_segment_map() -> xt.LineSegmentMap:
    """Return the LHC-like longitudinal map with its RF disabled (drift only)."""
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


def build_common_line(p0c_init: float = P0C_INIT) -> xt.Line:
    """
    Return a one-element line (the segment map) with a proton reference.

    Parameters
    ----------
    p0c_init
        Reference momentum [eV/c] of the line's ``particle_ref``.

    Returns
    -------
    xtrack.Line
        A line holding a single ``LineSegmentMap`` named ``"matrix"``.
    """
    line = xt.Line(
        elements=[build_line_segment_map()], element_names=["matrix"]
    )
    line.particle_ref = xp.Particles(
        p0c=p0c_init, mass0=xp.PROTON_MASS_EV, q0=1.0
    )
    return line


def attach_energy_program(line: xt.Line, p0c_ramp: np.ndarray) -> None:
    """
    Attach an ``EnergyProgram`` with one node per turn at the true turn times.

    ``xt.EnergyProgram`` is specified in *real time* (``p0c`` vs ``t_s``) and
    xsuite samples it at the beam's accumulated revolution time, which advances
    by the velocity-dependent period :math:`T_\\mathrm{rev} = C / (\\beta c)`.
    Placing node ``i`` at the cumulative real turn time therefore makes
    ``p0c_ramp[i]`` the reference momentum used *during turn i*, matching a BLonD
    ``MagneticCyclePerTurn`` indexed by turn. The ultra-relativistic spacing
    ``C / c`` instead leaves a per-turn timing error of order ``1 - beta``
    (~2e-6 at LHC injection) that accumulates into the longitudinal coordinates
    over a ramp -- see ``test_ramp_blond_owns_wrapped_xsuite``.

    Parameters
    ----------
    line
        The line to attach the program to (modified in place).
    p0c_ramp
        Reference momentum [eV/c] per program node, one node per turn.
    """
    p0c_ramp = np.asarray(p0c_ramp, dtype=float)
    energy = np.sqrt(p0c_ramp**2 + xp.PROTON_MASS_EV**2)
    beta = p0c_ramp / energy
    # Node i sits at the start of turn i: the cumulative real revolution period
    # of the preceding turns.
    t_rev = CIRCUMFERENCE / (beta * c)
    t_s = np.concatenate(([0.0], np.cumsum(t_rev[:-1])))
    line.energy_program = xt.EnergyProgram(t_s=t_s, p0c=p0c_ramp)


def smooth_ramp(n_turns: int) -> np.ndarray:
    """
    Return a linear p0c ramp with a two-entry margin past the last turn.

    Parameters
    ----------
    n_turns
        Number of tracked turns.

    Returns
    -------
    numpy.ndarray
        ``n_turns + 2`` momentum nodes from ``P0C_INIT`` to ``P0C_FINAL``;
        the margin keeps the energy program from falling off its right edge.
    """
    return np.linspace(P0C_INIT, P0C_FINAL, n_turns + 2)


def initial_distribution() -> dict[str, np.ndarray]:
    """
    Return a deterministic seeded 6-D distribution for ``build_particles``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per-coordinate arrays (``x``/``px``/``y``/``py``/``zeta``/``delta``)
        of length ``N_PART``, reproducible via the fixed ``SEED``.
    """
    rng = np.random.default_rng(SEED)
    return {
        "x": rng.uniform(-1e-3, 1e-3, N_PART),
        "px": rng.uniform(-1e-5, 1e-5, N_PART),
        "y": rng.uniform(-2e-3, 2e-3, N_PART),
        "py": rng.uniform(-3e-5, 3e-5, N_PART),
        "zeta": np.linspace(-0.5, 0.5, N_PART),
        "delta": np.linspace(-1e-4, 1e-4, N_PART),
    }


def assert_phase_space_match(
    zeta_test, ptau_test, zeta_ref, ptau_ref, n_turns, rtol=1e-5
) -> None:
    """
    Assert that a run matches the reference run turn-by-turn.

    Tolerance is amplitude-scaled per turn (``rtol`` times the reference
    amplitude, floored at a tiny absolute value) so near-zero coordinates do
    not trip a pure relative check.

    Parameters
    ----------
    zeta_test, ptau_test
        Coordinates from the run under test, shape ``(n_part, n_turns)``.
    zeta_ref, ptau_ref
        Coordinates from the reference run, same shape.
    n_turns
        Number of turns to compare.
    rtol
        Relative tolerance applied to the per-turn amplitude.
    """
    # Device arrays (GPU backend/context) are compared on the host.
    zeta_test = copy_to_cpu(zeta_test)
    ptau_test = copy_to_cpu(ptau_test)
    zeta_ref = copy_to_cpu(zeta_ref)
    ptau_ref = copy_to_cpu(ptau_ref)
    assert zeta_test.shape == zeta_ref.shape
    assert ptau_test.shape == ptau_ref.shape
    for turn in range(n_turns):
        zeta_amp = float(np.max(np.abs(zeta_ref[:, turn])))
        ptau_amp = float(np.max(np.abs(ptau_ref[:, turn])))
        np.testing.assert_allclose(
            zeta_test[:, turn],
            zeta_ref[:, turn],
            atol=max(zeta_amp * rtol, 1e-12),
            rtol=0.0,
            err_msg=f"zeta mismatch on turn {turn}",
        )
        np.testing.assert_allclose(
            ptau_test[:, turn],
            ptau_ref[:, turn],
            atol=max(ptau_amp * rtol, 1e-15),
            rtol=0.0,
            err_msg=f"ptau mismatch on turn {turn}",
        )


def plot_three(
    zeta_ref, ptau_ref, zeta_test, ptau_test, test_label: str
) -> None:
    """
    Render the three diagnostic figures used across the xsuite cross-checks.

    Same format as ``test_psb_pure_cavity_1e6turns``: the input distribution,
    the output distribution (reference vs run-under-test overlaid), and the
    per-particle normalised phase-space distance vs turn.

    Parameters
    ----------
    zeta_ref, ptau_ref
        Reference-run coordinates, shape ``(n_part, n_turns)``.
    zeta_test, ptau_test
        Run-under-test coordinates, same shape.
    test_label
        Legend label for the run under test.
    """

    zeta_ref = copy_to_cpu(zeta_ref)
    ptau_ref = copy_to_cpu(ptau_ref)
    zeta_test = copy_to_cpu(zeta_test)
    ptau_test = copy_to_cpu(ptau_test)
    n_part = zeta_ref.shape[0]

    # 1) input coordinates (first recorded turn).
    plt.figure("input coordinates")
    plt.scatter(
        zeta_ref[:, 0], ptau_ref[:, 0], c=range(n_part), cmap="viridis"
    )
    plt.xlabel(r"$\zeta$ [m]")
    plt.ylabel(r"$p_\tau$")
    plt.title("Input coordinates")
    plt.grid(True, alpha=0.3)

    # 2) output coordinates (last recorded turn), both runs overlaid.
    plt.figure("output coordinates")
    plt.scatter(
        zeta_ref[:, -1], ptau_ref[:, -1], marker="x", c="C0", label="xsuite"
    )
    plt.scatter(
        zeta_test[:, -1],
        ptau_test[:, -1],
        marker="o",
        facecolors="none",
        edgecolors="C1",
        label=test_label,
    )
    plt.xlabel(r"$\zeta$ [m]")
    plt.ylabel(r"$p_\tau$")
    plt.title("Output coordinates")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3) per-particle normalised distance vs turn.
    zamp = float(np.max(np.abs(zeta_ref))) or 1.0
    pamp = float(np.max(np.abs(ptau_ref))) or 1.0
    dist = np.sqrt(
        ((zeta_test - zeta_ref) / zamp) ** 2
        + ((ptau_test - ptau_ref) / pamp) ** 2
    )
    turns = np.arange(zeta_ref.shape[1])
    plt.figure("distance per particle")
    for k in range(n_part):
        plt.semilogy(turns, dist[k] + 1e-300, label=f"particle {k}")
    plt.xlabel("turn")
    plt.ylabel("normalised phase-space distance")
    plt.title(f"{test_label} vs xsuite: distance per particle")
    plt.ylim(1e-17, 1e-2)
    plt.grid(True, which="both", alpha=0.3)

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def plot_comparison(
    zeta_ref, ptau_ref, zeta_test, ptau_test, test_label: str
) -> None:
    """
    Render phase-space + per-turn error plots for manual inspection.

    Dev aid only (the tests gate it behind a ``DEV_PLOT`` flag). No-op under a
    headless backend: figures are drawn but ``show()`` is skipped under "agg".

    Parameters
    ----------
    zeta_ref, ptau_ref
        Reference-run coordinates, shape ``(n_part, n_turns)``.
    zeta_test, ptau_test
        Run-under-test coordinates, same shape.
    test_label
        Legend label for the run under test.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    n_turns = zeta_ref.shape[1]
    snapshots = sorted({0, n_turns // 2, n_turns - 1})
    fig, axes = plt.subplots(
        1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False
    )
    for ax, turn in zip(axes[0], snapshots, strict=False):
        ax.scatter(
            zeta_ref[:, turn], ptau_ref[:, turn], label="xsuite", marker="x"
        )
        ax.scatter(
            zeta_test[:, turn],
            ptau_test[:, turn],
            label=test_label,
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
    dz_max = np.max(np.abs(zeta_test - zeta_ref), axis=0)
    dp_max = np.max(np.abs(ptau_test - ptau_ref), axis=0)
    ax_z.semilogy(turns, dz_max + 1e-300)
    ax_z.set_ylabel(r"max $|\Delta \zeta|$ [m]")
    ax_z.grid(True, which="both", alpha=0.3)
    ax_p.semilogy(turns, dp_max + 1e-300)
    ax_p.set_ylabel(r"max $|\Delta p_\tau|$")
    ax_p.set_xlabel("turn")
    ax_p.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
