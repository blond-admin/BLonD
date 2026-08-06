"""
Closed-loop (Robinson) dynamical stability of the driven PI cavity feedback.

Every other feedback test verifies the loop *open*: a non-driven cavity
response, or an ``<=8``-turn ``|V_ant|`` recovery. None checks that the
*coupled* system -- bunch synchrotron motion <-> beam loading <-> PI voltage
regulation -- is dynamically stable, i.e. that the coherent longitudinal
dipole (the Robinson mode) stays bounded rather than growing turn over turn.
A growth of a few ``1e-3`` per turn is invisible over the ``3``-``8`` turns the
existing PI tests run, because the synchrotron period is tens of turns.

This test closes the loop and tracks a real matched bunch, kicked into a
coherent dipole, through the driven
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
(with a
:class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController`)
plus strong beam loading, for ``300`` turns (~``30`` synchrotron periods at
``Q_s ~ 0.10``).  It records the bunch centroid ``<dt>`` each turn -- the
coherent-dipole amplitude -- and fits the growth rate of its oscillation
envelope.  Two configurations are compared, differing *only* in the sign of
the cavity resonance detuning ``delta_omega``:

* **nominal** (``delta_omega = -DELTA_OMEGA``): the empirically stabilizing
  sign at this above-transition operating point -- the dipole envelope stays
  bounded (net decay, growth rate ``< 0``);
* **perturbed** (``delta_omega = +DELTA_OMEGA``): the flipped, destabilizing
  sign -- the dipole envelope grows (growth rate ``> 0``), measurably faster
  than the nominal.

**Machine point.**  ``E0 = 4 GeV`` constant (just above transition,
``gamma_t = 1/sqrt(alpha_p) ~ 31``), so the RF frame does *not* slip and the
bunch stays in the profile window over the whole long run -- unlike the
accelerating fast-ramp regime, where the frame slips ~``0.09 t_rf`` per turn
and the bunch would leave the window within tens of turns.  At ``4 GeV`` the
beam loading couples strongly to the (soft) longitudinal motion; at the
``63 GeV`` operating point the bunch is too stiff for a detuning-dependent
dipole signal to rise above the filamentation floor.

**What this test establishes, and what it does not.**

* It is a *differential* regression test: it pins that the nominal (stabilizing
  detuning) closed loop keeps the coherent dipole bounded while the perturbed
  (destabilizing detuning) closed loop lets it grow, with a robust growth-rate
  gap (~``2.5e-3`` per turn, stable across RNG seed and fit window).  It does
  **not** cross-check an analytic Robinson growth rate.
* The instability here is finite-amplitude / detuning driven and entangled
  with filamentation: a *near-linear* bunch shows no measurable growth for any
  detuning (the ultra-fast loop cancels linear Robinson entirely), so a
  realistic (filamenting) bunch is required to expose the differential.
* The loop runs at ~``harmonic`` coarse samples per turn, orders of magnitude
  faster than the synchrotron motion, so the loop gain and loop delay are
  dynamically *inert* on the dipole (verified in exploration): the operative
  destabilization knob is the detuning sign, not gain/delay.
* The differential is measured over turns ``[FIT_START, N_TURNS]`` (skipping
  the initial filamentation transient).  Beyond ~``350`` turns a slow secular
  drift common to *both* configurations (the documented bounded-secular-drift
  limitation) erodes the detuning differential, so the horizon is kept at
  ``300`` turns.

The run is deterministic (fixed RNG seed, deterministic tracking); the
assertions are inequalities with wide margins rather than hard numeric pins,
because the absolute growth rate is fit-window sensitive while the ordering
(perturbed grows, nominal bounded) and the gap are robust.
"""

import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_plus,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)

# --- machine (RCS1-like single cavity, just above transition) --------------
R_OVER_Q = 518.0
Q_L = 1.29e6
V_DESIGN = 30.0e6
HARMONIC = 25900
CIRCUMFERENCE = 5990.0
ALPHA_P = 10.395e-4
E0 = 4.0e9  # constant energy, above transition (no frame slip)

# --- beam / tracking -------------------------------------------------------
INTENSITY = 5.0e12  # strong beam loading (drives the coherent dipole)
N_SLICES = 512
N_MACROPARTICLES = 20_000
SEED = 7
N_TURNS = 300
SIGMA_DT_FRAC = 0.03  # matched bunch length as a fraction of t_rf
DIPOLE_KICK_FRAC = 0.05  # initial coherent-dipole offset as a fraction of t_rf

# --- feedback / controller (loop tuning as in the PI tracking tests) -------
I_GEN_BIAS = V_DESIGN / (2.0 * R_OVER_Q * Q_L)
GAIN_P = 0.1 / (R_OVER_Q * 2.0 * np.pi)
N_DELAY = 2

# --- Robinson knob: the cavity resonance detuning sign ---------------------
# Nominal uses -DELTA_OMEGA (empirically stabilizing here); perturbed flips it
# to +DELTA_OMEGA (destabilizing). |delta_omega| ~ 2e5 rad/s is small vs the
# cavity bandwidth omega_rf/(2 Q_L) ~ 3e3 rad/s * ... (well within the
# forward-Euler step-size cap: delta_omega * t_rf ~ 1.5e-4 rad per step).
DELTA_OMEGA = 200_000.0

# Fit the envelope growth over turns [FIT_START, N_TURNS], skipping the first
# FIT_START_PERIODS synchrotron periods of filamentation transient.
FIT_START_PERIODS = 6

# --- assertion thresholds (measured: nominal ~-0.85e-3, perturbed ~+1.68e-3,
#     gap ~+2.55e-3 per turn; seed- and window-robust) --------------------
NOMINAL_BOUND = 5.0e-4  # nominal growth must stay below this (bounded)
PERTURBED_GROWS = 5.0e-4  # perturbed growth must exceed this (grows)
DIFFERENTIAL_MARGIN = 1.5e-3  # perturbed - nominal must exceed this


def _sliding_dipole_amplitude(centroid: np.ndarray, period: int) -> np.ndarray:
    """
    Coherent-dipole envelope: sliding RMS of the centroid over one period.

    The standard deviation over a ``+/- period`` window removes the local mean,
    so the envelope is insensitive to the slow drift of the synchronous
    (equilibrium) phase and reflects only the coherent oscillation amplitude.

    Parameters
    ----------
    centroid
        Per-turn bunch centroid ``<dt>`` [s].
    period
        Synchrotron period in turns, the half-width of the sliding window.

    Returns
    -------
    numpy.ndarray
        Per-turn coherent-dipole amplitude [s].
    """
    n = len(centroid)
    amplitude = np.empty(n)
    for i in range(n):
        window = centroid[max(0, i - period) : min(n, i + period + 1)]
        amplitude[i] = float(np.std(window))
    return amplitude


def _growth_rate_per_turn(
    centroid: np.ndarray, period: int, fit_start: int
) -> float:
    """
    Exponential growth rate of the coherent-dipole envelope, per turn.

    Fits ``log(amplitude)`` linearly over ``[fit_start, len(centroid))``; the
    slope is the per-turn growth (``> 0`` unstable, ``< 0`` damped).

    Parameters
    ----------
    centroid
        Per-turn bunch centroid ``<dt>`` [s].
    period
        Synchrotron period in turns (envelope window half-width).
    fit_start
        First turn of the fit window (skips the filamentation transient).

    Returns
    -------
    float
        Growth rate per turn.
    """
    amplitude = _sliding_dipole_amplitude(centroid, period)
    turns = np.arange(len(centroid))
    window = slice(fit_start, len(centroid))
    amp = np.where(amplitude[window] <= 0.0, np.nan, amplitude[window])
    good = np.isfinite(amp)
    slope = np.polyfit(turns[window][good], np.log(amp[good]), 1)[0]
    return float(slope)


def _run_closed_loop(delta_omega: float) -> dict:
    """
    Track a kicked matched bunch through the driven feedback and record it.

    Builds a single-station ring (half-drift / RF station with a PI-regulated
    :class:`IQCavityFeedbackTimingClass` / half-drift) at constant ``E0``,
    prepares a matched ``BiGaussian`` bunch, displaces its centroid by
    ``DIPOLE_KICK_FRAC * t_rf`` to excite a coherent dipole, and tracks
    ``N_TURNS`` turns recording the centroid and diagnostics each turn.

    Parameters
    ----------
    delta_omega
        Cavity resonance detuning [rad/s] (the Robinson sign knob).

    Returns
    -------
    dict
        ``centroid`` (per-turn ``<dt>`` [s]), ``in_window`` (captured
        fraction), ``i_max_dev`` (per-turn peak generator-current deviation
        from the bias [A] -- the loop response), ``q_s`` (synchrotron tune),
        ``period`` (synchrotron period in turns) and ``t_rf`` [s].
    """
    cycle = ConstantMagneticCycle(
        reference_particle=mu_plus, value=E0, in_unit="total energy"
    )
    t_rev = cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    t_rf = t_rev / HARMONIC

    profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, N_SLICES, t_rf)
    controller = GeneratorCurrentPIController(
        gain_proportional=GAIN_P,
        gain_integral=GAIN_P / (30.0 * t_rf),
        generator_current_bias=I_GEN_BIAS + 0.0j,
        n_delay=N_DELAY,
    )
    feedback = IQCavityFeedbackTimingClass(
        profile=profile,
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=I_GEN_BIAS + 0.0j,
        n_cavities=1,
        initial_voltage=V_DESIGN,
        n_rf_periods_per_coarse_grid=1,
        delta_omega=delta_omega,
        controller=controller,
        voltage_setpoint=V_DESIGN + 0.0j,
    )
    station = SingleHarmonicRFStation(
        voltage=V_DESIGN,
        phi_rf=0.0,
        harmonic=HARMONIC,
        cavity_feedback=feedback,
        profile=profile,
    )
    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    ring.add_elements(
        [
            DriftSimple(
                orbit_length=CIRCUMFERENCE / 2,
                momentum_compaction_factor=ALPHA_P,
            ),
            station,
            DriftSimple(
                orbit_length=CIRCUMFERENCE / 2,
                momentum_compaction_factor=ALPHA_P,
            ),
        ],
        reorder=False,
    )
    sim = Simulation(ring=ring, magnetic_cycle=cycle)

    beam = Beam(intensity=INTENSITY, particle_type=mu_plus)
    beam.reference.total_energy = E0
    sim.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            n_macroparticles=N_MACROPARTICLES,
            sigma_dt=SIGMA_DT_FRAC * t_rf,
            sigma_dE=None,
            seed=SEED,
            reinsertion=True,
        ),
    )
    q_s = float(station.calc_synchrotron_tune_main_harmonic(beam))
    period = max(int(round(1.0 / q_s)), 4)

    # Shift the whole bunch one RF period into the window centre, then add the
    # coherent-dipole displacement that excites the mode.
    beam._dt.array_local += t_rf + DIPOLE_KICK_FRAC * t_rf

    rec: dict[str, list] = {"centroid": [], "in_window": [], "i_max_dev": []}
    # StaticProfile: transfer the window bounds once, so the per-turn
    # comparison below stays host-side.
    _window = copy_to_cpu(profile.hist_x)
    window_left = float(_window[0])
    window_right = float(_window[-1])

    def callback(_sim, b):
        dt = copy_to_cpu(b.dt.array_local)
        rec["centroid"].append(float(np.mean(dt)))
        rec["in_window"].append(
            float(np.mean((dt > window_left) & (dt < window_right)))
        )
        n_forward = int(feedback._rf_centers_lengths[-1])
        rec["i_max_dev"].append(
            float(
                np.abs(
                    feedback.generator_current_coarse_grid[-n_forward:]
                    - I_GEN_BIAS
                ).max()
            )
        )

    sim.run_simulation(
        (beam,), n_turns=N_TURNS, callbacks=callback, show_progressbar=False
    )
    return {
        "centroid": np.array(rec["centroid"]),
        "in_window": np.array(rec["in_window"]),
        "i_max_dev": np.array(rec["i_max_dev"]),
        "q_s": q_s,
        "period": period,
        "t_rf": t_rf,
    }


class TestClosedLoopRobinsonStability(unittest.TestCase):
    """
    Coherent-dipole stability of the driven closed loop vs the detuning sign.

    Tracks a kicked matched bunch through the PI-regulated cavity feedback plus
    beam loading for many synchrotron periods and checks that the stabilizing
    detuning keeps the coherent dipole bounded while the flipped
    (destabilizing) detuning lets it grow measurably faster.
    """

    _cache: dict = {}

    @classmethod
    def setUpClass(cls):
        """Run the nominal and perturbed closed loops once and cache them."""
        cls.nominal = _run_closed_loop(-DELTA_OMEGA)
        cls.perturbed = _run_closed_loop(+DELTA_OMEGA)
        cls.fit_start = FIT_START_PERIODS * cls.nominal["period"]
        cls.growth_nominal = _growth_rate_per_turn(
            cls.nominal["centroid"], cls.nominal["period"], cls.fit_start
        )
        cls.growth_perturbed = _growth_rate_per_turn(
            cls.perturbed["centroid"], cls.perturbed["period"], cls.fit_start
        )

    # ------------------------------------------------------------------ #
    # guards: the setup is non-trivial and the measurement is well posed #
    # ------------------------------------------------------------------ #
    def test_setup_spans_many_synchrotron_periods(self):
        """The run resolves the dipole over many synchrotron periods."""
        period = self.nominal["period"]
        self.assertGreaterEqual(period, 4)
        self.assertGreater(
            N_TURNS / period,
            10.0,
            f"run spans only {N_TURNS / period:.1f} synchrotron periods",
        )
        # The fit window still leaves many periods after the transient skip.
        self.assertGreater(N_TURNS - self.fit_start, 15 * period)

    def test_bunch_stays_captured_and_loop_is_driven(self):
        """
        Guard against a trivial run: bunch captured, loop strongly driven.

        The centroid measurement is meaningful only if the bunch stays in the
        profile window (no particle loss) and the PI loop actually acts (the
        generator current swings far above its tiny matched bias, as in the PI
        tracking tests).
        """
        for label, run in (
            ("nominal", self.nominal),
            ("perturbed", self.perturbed),
        ):
            with self.subTest(config=label):
                self.assertGreater(
                    float(run["in_window"].min()),
                    0.99,
                    f"{label}: bunch left the profile window "
                    f"(min in-window {run['in_window'].min():.3f})",
                )
                self.assertGreater(
                    float(run["i_max_dev"].max()) / I_GEN_BIAS,
                    1.0,
                    f"{label}: PI loop barely acts on the generator current",
                )

    def test_initial_dipole_is_excited(self):
        """The kick excites a real coherent dipole in both configurations."""
        for label, run in (
            ("nominal", self.nominal),
            ("perturbed", self.perturbed),
        ):
            amp = _sliding_dipole_amplitude(run["centroid"], run["period"])
            with self.subTest(config=label):
                # Initial coherent amplitude is a sizable fraction of t_rf.
                self.assertGreater(
                    float(amp[: run["period"]].max()) / run["t_rf"],
                    0.01,
                    f"{label}: no coherent dipole was excited",
                )

    # ------------------------------------------------------------------ #
    # Assert #1: the nominal (stabilizing) closed loop is bounded        #
    # ------------------------------------------------------------------ #
    def test_nominal_dipole_stays_bounded(self):
        """
        Nominal detuning: the coherent dipole does not grow exponentially.

        The stabilizing-sign closed loop keeps the envelope bounded -- its
        growth rate is below the (small, positive) stability threshold and the
        envelope net-decays from its kicked initial value over the run.
        """
        self.assertLess(
            self.growth_nominal,
            NOMINAL_BOUND,
            f"nominal dipole growth {self.growth_nominal:+.3e}/turn exceeds "
            f"the bounded threshold {NOMINAL_BOUND:+.1e} -- the supposedly "
            "stable closed loop is growing",
        )
        amp = _sliding_dipole_amplitude(
            self.nominal["centroid"], self.nominal["period"]
        )
        period = self.nominal["period"]
        initial = float(amp[:period].max())
        final = float(amp[-period:].mean())
        self.assertLess(
            final,
            initial,
            f"nominal dipole did not net-decay (initial {initial:.2e} -> "
            f"final {final:.2e})",
        )

    # ------------------------------------------------------------------ #
    # Assert #2: the perturbed (destabilizing) loop grows measurably more #
    # ------------------------------------------------------------------ #
    def test_perturbed_dipole_grows(self):
        """Flipped (destabilizing) detuning: the coherent dipole grows."""
        self.assertGreater(
            self.growth_perturbed,
            PERTURBED_GROWS,
            f"perturbed dipole growth {self.growth_perturbed:+.3e}/turn does "
            "not exceed the growth threshold -- the test cannot see growth",
        )

    def test_perturbed_grows_measurably_more_than_nominal(self):
        """
        The differential: perturbed grows faster than nominal, with margin.

        This is the core, seed- and window-robust content: flipping the cavity
        detuning sign turns a bounded coherent dipole into a growing one, so
        the coupled bunch/beam-loading/regulation system's stability genuinely
        depends on the detuning -- and the metric can resolve the difference.
        """
        differential = self.growth_perturbed - self.growth_nominal
        self.assertGreater(
            differential,
            DIFFERENTIAL_MARGIN,
            f"perturbed grows only {differential:+.3e}/turn more than nominal "
            f"(nominal {self.growth_nominal:+.3e}, perturbed "
            f"{self.growth_perturbed:+.3e}); expected a gap > "
            f"{DIFFERENTIAL_MARGIN:.1e}",
        )


if __name__ == "__main__":
    unittest.main()
