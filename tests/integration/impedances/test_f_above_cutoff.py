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
observation bin and a third one -- i.e. it weights the wake with the
quadratic B-spline ``box * box * box`` (``sinc(f dt)**3``, with NumPy's
``sinc(x) = sin(pi x) / (pi x)``). Sampling the wake every ``dt`` folds
the impedance and each box suppresses the alias images one order further.
Two boxes already keep the above-Nyquist pole from swamping the
impedance's inductive flank, but their tail decays as ``1 / j**2`` alone
and leaks the pole's *resistive* part into this reactive band; the third
box takes the tail to ``1 / |j|**3``. It is a bandwidth argument, not a
timing one. In pole-residue form the third box scales each residue by
``((exp(p*dt) - 1) / (p*dt))**3 * exp(p*dt/2)``; the recursion then covers
lags of two bins and more (its state is referenced two bins back) and
``MultiPoleSparseSolve`` adds the three near lags -- the previous bin, the
bin itself and the *next* one, reached by the kernel's non-causal tap --
in closed form. See ``Resonators._wake_bin_average`` and
``MultiPoleSparseSolve``.

This test reproduces the *exact* runaway resonator above directly against
``blond.physics.impedances.sources.Resonators`` (BLonD's own source, not
any external fitter) and checks that ``MultiPoleSparseSolve`` tracks the
frequency-domain reference for it.

Why the line density is analytic
--------------------------------
The profile is *not* histogrammed from macroparticles here: it is set to
the exact bin-integrated Gaussian, i.e. the infinite-statistics limit of
the histogram (the same trick as in
``tests/unittests/physics/impedances/comparisons/
test_induced_voltage_resonator_mtw_counterrotation_analytical.py``).
An earlier version of this test sampled 200k macroparticles into the 128
bins and then compared the two solvers; that measured *shot noise*, not
solver accuracy. This resonator's impedance is inductive over the whole
resolved band, so the induced voltage differentiates the line density and
amplifies bin-to-bin histogram noise; the time-domain kernel low-passes
that noise with its ``sinc^3`` bin average (up to 4x at Nyquist) while
``PeriodicFreqSolver`` does not, so the two solvers disagreed about the
noise (peak ratio 0.84, 4.6 % rms at 200k particles, converging to 1 only
as ``1/sqrt(N)``: 0.96 at 2e6, 0.99 at 2e7 macroparticles).

With the analytic line density the same comparison at the same
``|p| * bin_dt = 3.22`` measures the solvers instead, and they agree to
peak ratio 0.9986 with 0.069 % rms deviation; refining the profile to 256
bins shrinks both deviations by ~3.9x (second order in ``bin_dt``), which
is what identifies the remaining difference as discretisation error
rather than a solver defect.

Set ``DEV_DRAW=true`` in the environment to plot the induced-voltage
comparison and residual for visual inspection.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.special import erf

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
    backend,
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
N_BINS_REFINED = 2 * N_BINS  # for the discretisation-convergence check
INTENSITY = 1.15e11  # protons
# The line density is analytic (see module docstring), so the macro-
# particles only receive the induced-voltage kick and never enter the
# measurement; a small seeded beam keeps the turn cheap and deterministic.
N_MACROPARTICLES = 1_000

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


def _bin_integrated_gaussian(
    hist_x: np.ndarray,
    hist_step: float,
    center: float,
    sigma_dt: float,
) -> np.ndarray:
    """Fraction of a Gaussian bunch falling into each profile bin.

    This is what ``StaticProfile`` would histogram from an infinite
    number of macroparticles, so it carries no shot noise while keeping
    the ``hist_y * hist_y_to_density_factor`` semantics of a real
    profile (the entries sum to one).
    """
    edges_left = hist_x - 0.5 * hist_step
    edges_right = hist_x + 0.5 * hist_step
    scale = np.sqrt(2.0) * sigma_dt
    return 0.5 * (
        erf((edges_right - center) / scale)
        - erf((edges_left - center) / scale)
    )


def _run_one_turn(
    solver, t_rf: float, sigma_dt: float, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one turn with an analytic Gaussian line density.

    The profile is filled with the bin-integrated Gaussian instead of a
    macroparticle histogram, and the wakefield is told not to re-track
    the profile, so the induced voltage is a deterministic, noise-free
    function of the solver.

    Returns ``(hist_x, hist_y, induced_voltage)``.
    """
    ring, cycle, drift, rf = _build_ring()
    profile = StaticProfile(cut_left=0.0, cut_right=t_rf, n_bins=n_bins)
    resonator = Resonators(RS, FR, QUALITY_FACTOR)
    wakefield = WakeField(sources=(resonator,), solver=solver, profile=profile)
    # keep the analytic line density: neither the wakefield nor the ring
    # may overwrite it with a macroparticle histogram
    wakefield.track_profile = False
    ring.add_elements([drift, rf, wakefield], reorder=True)

    sim = Simulation(ring=ring, magnetic_cycle=cycle)
    beam = Beam(intensity=INTENSITY, particle_type=proton)
    sim.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            sigma_dt=sigma_dt, n_macroparticles=N_MACROPARTICLES, seed=42
        ),
    )
    hist_x = np.asarray(copy_to_cpu(profile.hist_x))
    bin_fractions = _bin_integrated_gaussian(
        hist_x, profile.hist_step, center=0.5 * t_rf, sigma_dt=sigma_dt
    )
    profile._hist_y[:] = backend.array(bin_fractions, dtype=backend.float)
    # `hist_y` already holds the per-bin probability, i.e. what a real
    # histogram of `n` macroparticles would give after scaling by `1/n`
    profile.hist_y_to_density_factor = 1.0

    sim.run_simulation(beams=(beam,), n_turns=1)
    return (
        hist_x,
        np.asarray(copy_to_cpu(profile.hist_y)),
        np.asarray(copy_to_cpu(wakefield.induced_voltage)),
    )


def _peak_ratio_and_rms(
    v_sparse: np.ndarray, v_freq: np.ndarray
) -> tuple[float, float]:
    """Peak ratio and peak-normalised rms deviation of the two solvers."""
    peak_freq = float(np.max(np.abs(v_freq)))
    if not peak_freq:
        return np.inf, np.inf
    peak_ratio = float(np.max(np.abs(v_sparse))) / peak_freq
    rms_err = float(np.sqrt(np.mean((v_sparse - v_freq) ** 2)) / peak_freq)
    return peak_ratio, rms_err


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
        limit relative to the profile's bin width (see module docstring),
        on an analytic (noise-free) Gaussian line density. Before the fix
        this diverged by orders of magnitude; with the triple bin-average
        the measured agreement is peak ratio 0.9986 / 0.069 % rms at 128
        bins, improving to 0.9996 / 0.018 % at 256 bins.
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

        peak_ratio, rms_err = _peak_ratio_and_rms(v_sparse, v_freq)

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
                    f"peak ratio={peak_ratio:.6g}x)",
                    fontsize=10,
                )
                ax_res.plot(t_ns, v_sparse - v_freq, color="#52514e", lw=1.0)
                ax_res.axhline(0.0, color="#52514e", lw=0.5, alpha=0.5)
                ax_res.set_ylabel("sparse $-$ freq / V")
                ax_hist.plot(t_ns, hist, color="#2a78d6", lw=1.0)
                ax_hist.set_ylabel("analytic line density")
                ax_hist.set_xlabel("time / ns")
                plt.show()

        # Before the low-Q-resonator fix this diverged by orders of
        # magnitude; the triple bin-average brings it to 0.9986x peak /
        # 0.069 % rms on the analytic line density (see module
        # docstring). The bounds keep ~7x margin on both measured
        # deviations.
        self.assertLess(
            rms_err,
            0.005,
            msg=(
                f"MultiPoleSparseSolve disagrees with the frequency-domain "
                f"reference by {100 * rms_err:.3f}% rms (peak ratio "
                f"{peak_ratio:.6g}x) for a pole with |p|*bin_dt = "
                f"{resolvability:.3g}"
            ),
        )
        self.assertTrue(
            0.99 < peak_ratio < 1.01,
            msg=(
                f"MultiPoleSparseSolve peak induced voltage is "
                f"{peak_ratio:.6g}x the frequency-domain reference for a "
                f"pole with |p|*bin_dt = {resolvability:.3g}"
            ),
        )

    def test_solver_agreement_improves_with_finer_bins(self):
        """The residual deviation must be discretisation error.

        Halving the bin width leaves the pole unresolvable
        (``|p|*bin_dt = 1.6`` at 256 bins) but must shrink both the peak
        deviation and the rms deviation, because what is left after the
        bin-average fix is a second-order discretisation error and not a
        solver defect. Measured shrink factors are ~3.9x on both, as
        expected for second order; the test only demands 2x.
        """
        t_rev = ConstantMagneticCycle(
            reference_particle=proton, value=SYNC_MOMENTUM, in_unit="momentum"
        ).get_t_rev_init(CIRCUMFERENCE, particle_type=proton)
        t_rf = t_rev / HARMONIC
        sigma_dt = t_rf / 10.0

        deviations = {}
        for n_bins in (N_BINS, N_BINS_REFINED):
            _, _, v_freq = _run_one_turn(
                PeriodicFreqSolver(), t_rf, sigma_dt, n_bins
            )
            _, _, v_sparse = _run_one_turn(
                MultiPoleSparseSolve(), t_rf, sigma_dt, n_bins
            )
            peak_ratio, rms_err = _peak_ratio_and_rms(v_sparse, v_freq)
            deviations[n_bins] = (abs(peak_ratio - 1.0), rms_err)

        peak_deviation_coarse, rms_coarse = deviations[N_BINS]
        peak_deviation_fine, rms_fine = deviations[N_BINS_REFINED]
        self.assertLess(
            peak_deviation_fine,
            0.5 * peak_deviation_coarse,
            msg=(
                f"peak deviation did not halve when refining "
                f"{N_BINS} -> {N_BINS_REFINED} bins: "
                f"{peak_deviation_coarse:.3g} -> {peak_deviation_fine:.3g}"
            ),
        )
        self.assertLess(
            rms_fine,
            0.5 * rms_coarse,
            msg=(
                f"rms deviation did not halve when refining "
                f"{N_BINS} -> {N_BINS_REFINED} bins: "
                f"{rms_coarse:.3g} -> {rms_fine:.3g}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
