"""Regression check: is the sparse solver's unresolvable-pole blow-up fixed?

Background
----------
``MultiPoleSparseSolve`` (BLonD's pole-residue sparse solver) advances each
pole's wake with a per-bin recursion. That recursion is only accurate if a
pole doesn't decay/oscillate faster than the profile's bin resolution can
represent (``|pole| * bin_dt`` well below 1). A pole-residue fit that is
not constrained to stay in that regime is a known way to make the induced
voltage come out orders of magnitude wrong.

That is what happened on the LHC model: its broadband resonator term was
fit with an unconstrained resonance frequency and converged to
``Rs=1.108e5 Ohm, Q=0.55 (near-critically-damped), fr=26.319 GHz`` --
since ``|pole| = 2*pi*fr`` for an underdamped resonator (independent of
``Q``), that's a pole at ``|p| = 1.65e11 rad/s``. Against a realistic LHC
bin width (``N_BINS=128`` profile bins per ``t_rf ~ 2.5 ns`` bucket,
``bin_dt ~ 1.95e-11 s``), that gives ``|p| * bin_dt ~ 3.2`` -- over 3x past
the "well below 1" resolvability limit -- and made the sparse solver's
induced voltage several orders of magnitude too strong and unstable. The
usual workaround is to cap ``fr`` (at a few GHz) while fitting; this test
checks that the solver no longer needs that crutch.

BLonD has since landed a fix for a related low-Q-resonator aliasing bug
(commit ``b4d2d9e3``, "Fixed InducedVoltageTime/InducedVoltageFreq
mismatch for low-Q resonators"): instead of point-sampling/naively
recursing a pole's wake, the wake is bin-averaged analytically. The
shipped kernel averages over *three* boxes -- the source bin, the
observation bin, and a third box that reconstructs the line density as
piecewise linear instead of as a staircase -- i.e. it weights the wake
with the quadratic B-spline ``box * box * box`` (``sinc^3(pi f dt)``).
Two boxes already remove the first-order aliasing of the above-Nyquist
pole onto the impedance's inductive flank; the third removes the half-bin
lag the staircase beam model leaves behind. In pole-residue form the
third box scales each residue by
``((exp(p*dt) - 1) / (p*dt))**3 * exp(p*dt/2)``; the recursion then covers
lags of two bins and more (its state is referenced two bins back) and
``MultiPoleSparseSolve`` adds the three near lags -- the previous bin, the
bin itself and the *next* one, reached by the kernel's non-causal tap --
in closed form. See ``Resonators._wake_bin_average`` and
``MultiPoleSparseSolve._finalize_solver``.

This test reproduces the *exact* runaway resonator above directly against
``blond.physics.impedances.sources.Resonators`` (BLonD's own source, not
any external fitter) and checks that ``MultiPoleSparseSolve`` tracks the
frequency-domain reference for it. Measured at the time of writing:
``|p| * bin_dt = 3.22``, sparse peak = 0.84x the frequency-domain peak,
4.6 % rms deviation.

Set ``DEV_DRAW=true`` in the environment to plot the induced-voltage
comparison and residual for visual inspection.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
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
# stands on its own)
CIRCUMFERENCE = 26658.883  # m
TRANSITION_GAMMA = 55.759505
HARMONIC = 35640
RF_VOLTAGE = 6e6  # V
SYNC_MOMENTUM = 450e9  # eV/c
N_BINS = 128  # profile bins per RF bucket, as used in LHC studies
INTENSITY = 1.15e11  # protons

# ---- the runaway broadband resonator an unconstrained LHC broadband
# fit converged to (see module docstring)
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
    """Build a fresh ring/beam/profile.

    Returns ``(t, hist, induced_voltage)``.
    """
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


@pytest.mark.integration
class TestUnresolvablePole(unittest.TestCase):
    """The sparse solver must not blow up on an unresolvable pole."""

    def test_sparse_solver_matches_freq_domain_for_unresolvable_pole(
        self,
    ):
        """``MultiPoleSparseSolve`` must not blow up on a fast, low-Q pole.

        Compares the sparse pole-residue solver against the
        frequency-domain reference (``PeriodicFreqSolver``) for a
        resonator whose pole is ~3x past the "well below 1" resolvability
        limit relative to the profile's bin width (see module docstring).
        Before the fix this diverged by orders of magnitude; with the
        triple bin-average it is at 0.84x peak / 4.6 % rms.
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
        self.assertGreater(
            resolvability,
            1.0,
            msg=(
                "test setup no longer reproduces an unresolvable pole "
                f"(|p|*bin_dt = {resolvability:.3g}); update "
                "RS/FR/QUALITY_FACTOR/N_BINS"
            ),
        )

        t_ns, hist, v_freq = _run_one_turn(
            PeriodicFreqSolver(), t_rf, sigma_dt
        )
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

        # Before the low-Q-resonator fix this diverged by orders of
        # magnitude; the triple bin-average brings it to 0.84x peak /
        # 4.6 % rms (see module docstring).
        self.assertLess(
            rms_err,
            0.05,
            msg=(
                f"MultiPoleSparseSolve disagrees with the frequency-domain "
                f"reference by {100 * rms_err:.2f}% rms (peak ratio "
                f"{peak_ratio:.3g}x) for a pole with |p|*bin_dt = "
                f"{resolvability:.3g}"
            ),
        )
        self.assertTrue(
            0.8 < peak_ratio < 1.25,
            msg=(
                f"MultiPoleSparseSolve peak induced voltage is "
                f"{peak_ratio:.3g}x the frequency-domain reference for a "
                f"pole with |p|*bin_dt = {resolvability:.3g}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
