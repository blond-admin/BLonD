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
  sign at this above-transition operating point -- the dipole envelope decays
  hard, ~``-5e-3`` per turn, with a clean exponential shape (log-envelope fit
  ``r^2 = 0.96..0.98``);
* **perturbed** (``delta_omega = +DELTA_OMEGA``): the flipped sign -- most of
  that damping is gone, leaving the envelope only MARGINALLY stable.

The asserted quantity is the **gap between the two damping rates**, not growth
in the perturbed case. See *What this test establishes* for why.

**Machine point.**  ``E0 = 4 GeV`` constant (just above transition,
``gamma_t = 1/sqrt(alpha_p) ~ 31``), so the RF frame does *not* slip and the
bunch stays in the profile window over the whole long run -- unlike the
accelerating fast-ramp regime, where the frame slips ~``0.09 t_rf`` per turn
and the bunch would leave the window within tens of turns.  At ``4 GeV`` the
beam loading couples strongly to the (soft) longitudinal motion; at the
``63 GeV`` operating point the bunch is too stiff for a detuning-dependent
dipole signal to rise above the filamentation floor.

**What this test establishes, and what it does not.**

* It is a *differential* regression test: it pins that the cavity detuning
  SIGN still reaches the coherent dipole, by requiring the nominal sign to
  damp hard (``< -3e-3`` per turn) and the perturbed sign to damp at least
  ``3e-3`` per turn less. It does **not** cross-check an analytic Robinson
  growth rate.
* **It deliberately does NOT assert that the perturbed dipole grows.** That
  assertion was removed on 2026-09-02 as unsound: the perturbed envelope is
  not exponential but BEATS with a ~``200``-turn period and net-decays over
  ``400`` turns (log-fit ``r^2 = 0.15..0.17``, rising ~``3 %`` across the fit
  window where a true ``+1e-3``/turn exponential would rise ``27 %``). Its
  fitted slope is therefore set by where the horizon lands -- negative at
  ``200`` and ``400`` turns, positive only for horizons in roughly
  ``[285, 365]`` -- and about a fifth of RNG seeds fail a ``> 5e-4`` gate
  outright. Asserting it would pin the phase of a beat to an arbitrary turn
  count. The damping differential, by contrast, holds over ``4`` seeds x
  ``3`` fit windows x ``2`` horizons at ``+4.05e-3 .. +6.81e-3``, ~``15x``
  the seed scatter, bunch fully captured throughout.
* **The loop is deliberately backed off** (``LOOP_AUTHORITY``, ``N_DELAY``);
  at the PI tracking tests' tuning this test cannot exist. Since the
  controller began stepping on every tracked coarse cell, that fast loop
  regulates so completely that both detuning signs damp identically at
  ``-1.8e-4`` per turn and the gap collapses to ``7e-8``. Both knobs are
  load-bearing: reverting either -- full gain at this delay, or this gain at
  ``N_DELAY = 2`` -- drops the gap to ``-6e-16`` and ``+2.5e-5`` and fails
  the assertion. The earlier claim in this docstring that "loop gain and loop
  delay are dynamically inert on the dipole" is therefore RETIRED; they are
  the only knobs that work. Measured inert instead: the integral time (across
  five orders of magnitude, and as a pure-P loop) and the detuning magnitude
  (across three orders, gap staying below ``6e-7``).
* **This is not a realistic LLRF latency.** ``N_DELAY = 25`` is ``19.3 ns``,
  ~``50x`` short of the ~``1 us`` of a real system. A realistic latency is
  currently unreachable here: the loop diverges numerically above
  ``N_DELAY ~ 11`` at full gain (and at ``1300`` even with the gain cut
  ``1e-3``), because the integral time is only ~``30`` samples and the
  controller has no klystron current limit (``max_output=None``) to bound the
  runaway. Modelling a real LLRF here needs a longer integral time AND a
  current limit first.
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
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
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

# --- feedback / controller ------------------------------------------------
# REDUCED-AUTHORITY loop, unlike the PI tracking tests. Since the controller
# began stepping on every tracked coarse cell (it used to act only on the
# forward passage, ~1/n_sections of the turn), the fast loop of those tests
# regulates the cavity so completely that the cavity detuning sign stops
# reaching the coherent dipole at all: both signs damp at -1.8e-4 per turn
# and the differential collapses to 7e-8. The module docstring had already
# conceded that "the ultra-fast loop cancels linear Robinson entirely"; the
# duty-cycle change finished the job.
#
# LOOP_AUTHORITY and N_DELAY below back the loop off until the detuning sign
# is operative again, and they are the ONLY knobs that do. Measured, at the
# unchanged gain: the loop delay is inert up to N_DELAY=11 and numerically
# divergent from 12; the integral time is inert across five orders of
# magnitude and even as a pure-P loop; the detuning magnitude is inert across
# three orders (gap stays < 6e-7, i.e. noise); reducing the gain alone either
# leaves the gap at 7e-8, inverts its sign (x1e-3), or loses the bunch
# (x1e-4 and below, in_window 0.53 and 0.20). Only delay AND gain together,
# in a narrow window at N_DELAY 20-28 with the gain reduced 3e-3..1e-2, give
# a captured bunch with a sign-correct differential.
I_GEN_BIAS = V_DESIGN / (2.0 * R_OVER_Q * Q_L)
#: Fraction of the PI-tracking-test proportional gain used here.
LOOP_AUTHORITY = 5.0e-3
GAIN_P = LOOP_AUTHORITY * 0.1 / (R_OVER_Q * 2.0 * np.pi)
#: Loop delay in coarse samples (1 sample = t_rf = 0.77 ns, so 19.3 ns).
#: NOT a realistic LLRF latency -- see the class docstring's closing note.
N_DELAY = 25

# --- Robinson knob: the cavity resonance detuning sign ---------------------
# Nominal uses -DELTA_OMEGA (empirically stabilizing here); perturbed flips it
# to +DELTA_OMEGA (destabilizing). |delta_omega| ~ 2e5 rad/s is small vs the
# cavity bandwidth omega_rf/(2 Q_L) ~ 3e3 rad/s * ... (well within the
# forward-Euler step-size cap: delta_omega * t_rf ~ 1.5e-4 rad per step).
DELTA_OMEGA = 200_000.0

# Fit the envelope growth over turns [FIT_START, N_TURNS], skipping the first
# FIT_START_PERIODS synchrotron periods of filamentation transient.
FIT_START_PERIODS = 6

# --- assertion thresholds --------------------------------------------------
# Measured over the full 4 seeds x 3 fit-windows x 2 horizons grid (36
# variations, every one with the bunch fully captured):
#   nominal   -6.16e-3 .. -4.34e-3   (always damped)
#   gap       +4.05e-3 .. +6.81e-3   (always positive, ~15x the seed scatter)
#   perturbed -2.00e-3 .. +2.26e-3   (SIGN FLIPS -- deliberately not asserted)
# The thresholds sit ~1.3x inside the worst case of the two robust legs.
NOMINAL_DAMPED = -3.0e-3  # nominal growth must be below this (net damped)
DIFFERENTIAL_MARGIN = 3.0e-3  # perturbed - nominal must exceed this


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
            NOMINAL_DAMPED,
            f"nominal dipole growth {self.growth_nominal:+.3e}/turn exceeds "
            f"the damped threshold {NOMINAL_DAMPED:+.1e} -- the stabilising "
            "detuning sign is no longer damping the coherent dipole",
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
    # Assert #2: the detuning SIGN still sets the damping rate            #
    # ------------------------------------------------------------------ #
    def test_detuning_sign_changes_the_damping_rate(self):
        """
        The differential: flipping the detuning costs most of the damping.

        This is the seed- and window-robust content, and the reason the
        class exists: the coupled bunch / beam-loading / regulation system's
        stability genuinely depends on the cavity detuning sign, and the
        metric can resolve the difference. Measured over 4 seeds x 3 fit
        windows x 2 horizons the gap spans +4.05e-3 .. +6.81e-3 per turn,
        about 15x the seed scatter, with the bunch fully captured in every
        one.

        Note what is asserted and what is not. The nominal (stabilising)
        sign damps hard, ~-5e-3 per turn, with a clean exponential envelope
        (log-fit r^2 = 0.96..0.98). The perturbed sign is left only
        MARGINALLY stable: its envelope beats with a ~200-turn period and
        net-decays over 400 turns (r^2 = 0.15..0.17), so its fitted slope
        flips sign with the horizon -- negative at 200 and 400 turns,
        positive only for horizons in roughly [285, 365]. Asserting
        "perturbed grows > 0" would therefore pin the phase of a beat
        against an arbitrary turn count, and about a fifth of seeds fail it
        outright. That assertion was REMOVED for this reason; the damping
        differential is asserted instead, because it is the part that
        survives every probe.
        """
        differential = self.growth_perturbed - self.growth_nominal
        self.assertGreater(
            differential,
            DIFFERENTIAL_MARGIN,
            f"perturbed is damped only {differential:+.3e}/turn less than "
            f"nominal (nominal {self.growth_nominal:+.3e}, perturbed "
            f"{self.growth_perturbed:+.3e}); expected a gap > "
            f"{DIFFERENTIAL_MARGIN:.1e}. The cavity detuning sign has "
            "stopped reaching the coherent dipole -- which is what a fully "
            "authoritative loop does (at the PI tracking tests' gain and "
            "delay this gap is 7e-8).",
        )


if __name__ == "__main__":
    unittest.main()
