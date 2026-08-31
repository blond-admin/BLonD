"""Regression check: is the sparse solver's unresolvable-pole blow-up fixed?

Background
----------
``MultiPoleSparseSolve`` (BLonD's pole-residue sparse solver) advances each
pole's wake with a per-bin recursion. That recursion is only accurate if a
pole doesn't decay/oscillate faster than the profile's bin resolution can
represent (``|pole| * bin_dt`` well below 1). SPS's own pole-residue fitter
(``simulations/machines/sps/pole_residue.py``) documents having measured
exactly this failure mode: "the induced voltage comes out orders of
magnitude wrong (measured: 3700x)".

We hit the same bug on the LHC model: its broadband resonator term was
fit with an unconstrained resonance frequency and converged to
``Rs=1.108e5 Ohm, Q=0.55 (near-critically-damped), fr=26.319 GHz`` --
since ``|pole| = 2*pi*fr`` for an underdamped resonator (independent of
``Q``), that's a pole at ``|p| = 1.65e11 rad/s``. Against a realistic LHC
bin width (``N_BINS=128`` profile bins per ``t_rf ~ 2.5 ns`` bucket,
``bin_dt ~ 1.95e-11 s``), that gives ``|p| * bin_dt ~ 3.2`` -- over 3x past
the "well below 1" resolvability limit -- and made
``simulations.machines.lhc.debug_simulation.fig_waterfall`` several
orders of magnitude too strong and unstable on the sparse solver (see
``simulations/machines/lhc/pole_generator/lhc_own_fitter.py``,
``apply_pole_cap``, which works around it by keeping ``fr`` under a 5 GHz
cap during fitting).

BLonD has since landed a fix for a related low-Q-resonator aliasing bug
(commit ``b4d2d9e3``, "Fixed InducedVoltageTime/InducedVoltageFreq
mismatch for low-Q resonators"): instead of point-sampling/naively
recursing a pole's wake, ``MultiPoleSparseSolve`` bin-averages it
analytically. That first fix averaged over the source bin only
(``sinh(p*dt/2)/(p*dt/2)`` residue scaling) and left ~1.18x peak /
~6.7% rms here; averaging over the observation bin as well (the residue
scaling squared -- see ``solvers.py``'s
``MultiPoleSparseSolve._finalize_solver`` and
``Resonators._wake_bin_average``) removes the first-order aliasing of the
above-Nyquist pole onto the impedance's inductive flank and brings this
down to ~0.94x peak / ~1.5% rms.

This test reproduces the *exact* runaway resonator above directly against
``blond.physics.impedances.sources.Resonators`` (BLonD's own source, not
this project's fitter) and checks that ``MultiPoleSparseSolve`` tracks the
frequency-domain reference for it.

Set ``DEV_DRAW=true`` in the environment to plot the induced-voltage
comparison and residual for visual inspection.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    PeriodicFreqSolver,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.impedances.sources import Resonators

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"

# ---- LHC injection parameters (real values, hardcoded here so this test
# has no dependency on the rest of the repo -- see simulations/machines/lhc)
CIRCUMFERENCE = 26658.883  # m
TRANSITION_GAMMA = 55.759505
HARMONIC = 35640
RF_VOLTAGE = 6e6  # V
SYNC_MOMENTUM = 450e9  # eV/c
N_BINS = 128  # profile bins per RF bucket, matches fig_waterfall's N_BINS
INTENSITY = 1.15e11  # protons

# ---- the runaway broadband resonator the (unconstrained) LHC fitter
# converged to -- see lhc_own_fitter.fit_broadband_resonator's docstring
RS = 1.108e5  # Ohm
QUALITY_FACTOR = 0.55  # just above the Q=0.5 critically-damped boundary
FR = 26.319e9  # Hz


def _build_ring() -> tuple:
    """LHC-like ring/RF/drift/cycle (flat momentum, single harmonic)."""
    ring = Ring(circumference=CIRCUMFERENCE)
    cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=SYNC_MOMENTUM,
        in_unit="momentum",
    )
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=TRANSITION_GAMMA
        ),
        orbit_length=CIRCUMFERENCE,
    )
    rf = SingleHarmonicRFStation(
        harmonic=HARMONIC, voltage=RF_VOLTAGE, phi_rf=0.0
    )
    return ring, cycle, drift, rf


def _run_one_turn(
    solver, t_rf: float, sigma_dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a fresh ring/beam/profile and return (t, hist, induced_voltage)."""
    ring, cycle, drift, rf = _build_ring()
    profile = StaticProfile(cut_left=0.0, cut_right=t_rf, n_bins=N_BINS)
    resonator = Resonators(RS, FR, QUALITY_FACTOR)
    wakefield = WakeField(sources=(resonator,), solver=solver, profile=profile)
    ring.add_elements([drift, rf, wakefield, profile], reorder=True)

    sim = Simulation(ring=ring, magnetic_cycle=cycle)
    beam = Beam(intensity=INTENSITY, particle_type=proton)
    sim.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            sigma_dt=sigma_dt, n_macroparticles=200_000, seed=42
        ),
    )
    sim.run_simulation(beams=(beam,), n_turns=1)
    return (
        np.asarray(copy_to_cpu(profile.hist_x)),
        np.asarray(copy_to_cpu(profile.hist_y)),
        np.asarray(copy_to_cpu(wakefield.induced_voltage)),
    )


def test_sparse_solver_matches_freq_domain_for_unresolvable_pole():
    """``MultiPoleSparseSolve`` must not blow up on a fast, low-Q pole.

    Compares the sparse pole-residue solver against the frequency-domain
    reference (``PeriodicFreqSolver``) for a resonator whose pole is ~3x
    past the "well below 1" resolvability limit relative to the profile's
    bin width (see module docstring). Before the fix this diverged by
    orders of magnitude (measured up to 3700x); with the double
    bin-average it is down to ~0.94x peak / ~1.5% rms.
    """
    t_rev = ConstantMagneticCycle(
        reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
    ).get_t_rev_init(CIRCUMFERENCE, particle_type=proton)
    t_rf = t_rev / HARMONIC
    bin_dt = t_rf / N_BINS
    sigma_dt = t_rf / 10.0

    resonator = Resonators(RS, FR, QUALITY_FACTOR)
    poles, _residues, _cr = resonator.get_vectorfit()
    pole_rate = float(np.max(np.abs(poles)))
    resolvability = pole_rate * bin_dt
    # Sanity check on the scenario itself: this resonator must actually be
    # in the unresolvable regime, or the test would pass for the wrong
    # reason (a well-resolved pole is trivially easy for any solver).
    assert resolvability > 1.0, (
        "test setup no longer reproduces an unresolvable pole "
        f"(|p|*bin_dt = {resolvability:.3g}); update RS/FR/QUALITY_FACTOR/N_BINS"
    )

    t_ns, hist, v_freq = _run_one_turn(PeriodicFreqSolver(), t_rf, sigma_dt)
    _, _, v_sparse = _run_one_turn(MultiPoleSparseSolve(), t_rf, sigma_dt)
    t_ns = t_ns * 1e9

    peak_freq = float(np.max(np.abs(v_freq)))
    peak_sparse = float(np.max(np.abs(v_sparse)))
    peak_ratio = peak_sparse / peak_freq if peak_freq else np.inf
    rms_err = (
        float(np.sqrt(np.mean((v_sparse - v_freq) ** 2)) / peak_freq)
        if peak_freq
        else np.inf
    )

    if _DEV_DRAW:
        with AllowPlotting():
            fig, (ax_v, ax_res, ax_hist) = plt.subplots(
                3,
                1,
                sharex=True,
                figsize=(7.0, 8.0),
                constrained_layout=True,
                height_ratios=(2, 1, 1),
            )
            ax_v.plot(
                t_ns,
                v_freq,
                color="#2a78d6",
                lw=1.5,
                label="PeriodicFreqSolver (reference)",
            )
            ax_v.plot(
                t_ns,
                v_sparse,
                color="#eb6834",
                lw=1.2,
                ls="--",
                label="MultiPoleSparseSolve",
            )
            ax_v.set_ylabel("induced voltage / V")
            ax_v.legend(frameon=False, loc="upper right")
            ax_v.set_title(
                f"unresolvable-pole regression check: Rs={RS:.3g} Ohm, "
                f"Q={QUALITY_FACTOR}, fr={FR / 1e9:.2f} GHz  "
                f"(|p|*bin_dt={resolvability:.2g}, "
                f"peak ratio={peak_ratio:.3g}x)",
                fontsize=10,
            )
            ax_res.plot(t_ns, v_sparse - v_freq, color="#52514e", lw=1.0)
            ax_res.axhline(0.0, color="#52514e", lw=0.5, alpha=0.5)
            ax_res.set_ylabel("sparse $-$ freq / V")
            ax_hist.plot(t_ns, hist, color="#2a78d6", lw=1.0)
            ax_hist.set_ylabel("beam profile")
            ax_hist.set_xlabel("time / ns")
            plt.show()

    # Before the low-Q-resonator fix this diverged by orders of magnitude
    # (measured up to 3700x); the double bin-average brings it to ~0.94x
    # peak / ~1.5% rms (see module docstring). The bounds keep a ~3x margin
    # on that, so they track the physics rather than the exact numbers.
    assert rms_err < 0.05, (
        f"MultiPoleSparseSolve disagrees with the frequency-domain "
        f"reference by {100 * rms_err:.2f}% rms (peak ratio "
        f"{peak_ratio:.3g}x) for a pole with |p|*bin_dt = "
        f"{resolvability:.3g}"
    )
    assert 0.8 < peak_ratio < 1.25, (
        f"MultiPoleSparseSolve peak induced voltage is {peak_ratio:.3g}x "
        f"the frequency-domain reference for a pole with |p|*bin_dt = "
        f"{resolvability:.3g}"
    )
