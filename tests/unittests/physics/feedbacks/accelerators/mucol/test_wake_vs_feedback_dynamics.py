"""
Self-consistent multi-turn *dynamics* twin: multi-pass wake vs cavity feedback.

Fills gap T3.  The neighbouring tests each pin only a slice of the wake /
feedback equivalence:

* ``test_energy_gain_ind_voltage_vs_nondriven_feedback`` -- one turn, applied
  ``dE`` only;
* ``test_feedback_phase_under_acceleration`` -- the induced *voltage* per turn
  against a self-derived analytic reference (no particle back-reaction);
* ``test_mtw_vs_nondriven_feedback`` -- induced voltage on a *frozen* profile
  (``profile.active = False``), a dummy particle-less beam.

None of them lets the induced voltage act back on a *real* bunch over many
turns.  This test does: it tracks the **same** matched ``BiGaussian`` ``mu+``
bunch through two otherwise-identical ``Simulation`` twins on the
transition-adjacent fast ramp and checks that the self-consistent bunch
dynamics agree turn by turn.

**Twin construction.**
Both rings are one ``DriftSimple`` + one ``SingleHarmonicRFStation`` at
``V_DESIGN`` (so the RF focusing, hence the synchrotron motion, is present),
started from the identical matched bunch.  The *only* difference is how the
beam-induced voltage is produced:

* ring **A** (wake): a retuning ``MultiPassResonatorSolver``
  (``retune_to_rf=True``) with
  ``R_s = R_over_Q * Q_L``, ``f_r = 1 / t_rf``, ``Q = Q_L`` -- the retuning
  multi-pass resonator convolution, applied as a separate wake kick on top of
  the clean design voltage;
* ring **B** (feedback): an ``IQCavityFeedbackTimingClass`` at its matched
  operating point (``initial_voltage = V_DESIGN``,
  ``generator_current_bias = V_DESIGN / (2 R_over_Q Q_L)``, ``delta_omega = 0``)
  -- its gap voltage bundles the design reconstruction and the beam loading.

Both rings apply the design kick via ``kick_interpolated`` on the *same* profile
grid, so with the beam removed they coincide to machine precision
(``test_design_kicks_are_identical_without_beam`` pins ~1e-11): the design part
is not a free parameter, so every beam-loaded difference is genuinely the
induced-voltage physics -- the analogue of the energy-gain test's
voltage-suppression trick, here made exact for the *whole* self-consistent
trajectory rather than a single kick.

**Regime.**
Same machine point as ``test_feedback_phase_under_acceleration`` (``E0 =
4 GeV``, just above ``gamma_t ~ 31``): the RF frame slips ~0.09 ``t_rf`` per
turn, so the retuning wake and the always-on-resonance feedback must both track
the slip.  ``Q_L = 1.29e6`` keeps ~88 % of the previous-pass wake, so the beam
loading builds up turn over turn (the induced voltage reaches ~27 % of
``V_DESIGN``).  At ``INTENSITY = 1.5e12`` the loading is strong enough to shift
the coherent centroid by ~0.9 ``sigma_dt`` away from the no-beam trajectory over
13 turns (``test_beam_loading_actually_drives_the_dynamics``) -- yet the two
twins stay within ~1e-3 ``sigma`` of each other.  Thirteen turns span >1
synchrotron period (``Q_s ~ 0.09`` -> ~11 turns), so genuine longitudinal
dynamics (centroid drift + ~4x ``sigma_dt`` breathing) are exercised, not a
static bunch.

Runtime: 4 full simulations (wake / feedback, each with and without beam),
20 000 macroparticles, 13 turns, cached in :attr:`_cache` -- ~6 s total.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Package-relative import: the dirs above ``mucol`` have no __init__.py, so the
# shared machinery is not importable by an absolute path under pytest. The
# module is imported (not the class): pytest collects *any* module-level name
# bound to a ``unittest.TestCase`` subclass, ignoring the ``Test*`` filter, so
# binding the phase test's class here would re-collect its (expensive) cases.
# ``_MatchedBeam`` below is the only handle onto it, and ``__test__ = False``
# keeps that handle out of collection.
from . import test_feedback_phase_under_acceleration as _phase_mod

DEBUG_PLOT = False


class _MatchedBeam(_phase_mod.TestFeedbackPhaseUnderAcceleration):
    """
    Reuse shim for the matched-beam machinery (never collected as a test).

    Subclasses the phase test purely to borrow its constants and its
    ``_t_rf`` / ``_accelerating_cycle`` / ``_matched_template`` classmethods,
    overriding only the ramp length and particle count needed here. It is also
    the sole in-module handle onto the phase test's ``TestCase`` (see the
    import note above); ``__test__ = False`` keeps pytest from collecting it.
    """

    __test__ = False

    # >1 synchrotron period (Q_s ~ 0.09 -> ~11 turns) while staying cheap.
    N_TURNS = 13
    # Modest count: real tracking over 13 turns x 4 sims must stay fast, and
    # 20k particles already resolve the moments compared here to ~1e-3.
    N_MACROPARTICLES = 20000


def _bunch_moments(dt: np.ndarray, dE: np.ndarray) -> dict:
    """
    Longitudinal centroid, spreads and rms emittance of a bunch.

    Parameters
    ----------
    dt
        Macroparticle arrival times [s].
    dE
        Macroparticle energy offsets [eV].

    Returns
    -------
    dict
        ``mean_dt`` [s], ``mean_dE`` [eV], ``sigma_dt`` [s], ``sigma_dE`` [eV]
        and the rms longitudinal ``emittance``
        ``sqrt(<dt^2><dE^2> - <dt dE>^2)`` [eV s].
    """
    mean_dt = float(np.mean(dt))
    mean_dE = float(np.mean(dE))
    d_dt = dt - mean_dt
    d_dE = dE - mean_dE
    var_dt = float(np.mean(d_dt * d_dt))
    var_dE = float(np.mean(d_dE * d_dE))
    cov = float(np.mean(d_dt * d_dE))
    emittance = float(np.sqrt(max(var_dt * var_dE - cov * cov, 0.0)))
    return {
        "mean_dt": mean_dt,
        "mean_dE": mean_dE,
        "sigma_dt": float(np.sqrt(var_dt)),
        "sigma_dE": float(np.sqrt(var_dE)),
        "emittance": emittance,
    }


class TestWakeVsFeedbackDynamics(unittest.TestCase):
    """
    Multi-turn self-consistent bunch dynamics: wake twin vs feedback twin.

    Tracks one matched bunch through two rings that differ only in the
    beam-induced-voltage model (multi-pass resonator wake vs matched-bias
    cavity feedback) and checks the per-turn bunch moments agree, while
    verifying the comparison is non-trivial (strong beam loading, identical
    design kicks, real synchrotron dynamics).
    """

    # Machine point and matched bunch reused from the phase test (reached via
    # the non-collected ``_MatchedBeam`` shim, which inherits them all).
    E0 = _MatchedBeam.E0
    HARMONIC = _MatchedBeam.HARMONIC
    V_DESIGN = _MatchedBeam.V_DESIGN
    R_OVER_Q = _MatchedBeam.R_OVER_Q
    Q_L = _MatchedBeam.Q_L
    CIRCUMFERENCE = _MatchedBeam.CIRCUMFERENCE
    ALPHA_P = _MatchedBeam.ALPHA_P
    N_SLICES = _MatchedBeam.N_SLICES
    N_TURNS = _MatchedBeam.N_TURNS

    # Strong-but-stable beam loading: induced voltage ~27 % of V_DESIGN, bunch
    # stays >97 % in-window (full 2.7e12 pushes ~14 % out of the window).
    INTENSITY = 1.5e12

    # --- twin-agreement gates (observed maxima in parentheses) ---
    CENTROID_GATE = 0.01  # |<dt>_A - <dt>_B| / sigma_dt  (~1.2e-3)
    SHAPE_GATE = 0.01  # rel. sigma_dt / sigma_dE diff     (~1.3e-3)
    EMIT_MAX = 0.01  # rel. rms-emittance diff            (~1.8e-3)
    EMIT_SLOPE_GATE = 1.0e-3  # growth of that diff per turn (~1.2e-4)
    DESIGN_GATE = 1.0e-6  # no-beam wake-vs-fb agreement   (~1e-11)

    # --- guards that the comparison is meaningful ---
    VPEAK_MIN_FRAC = 0.1  # peak induced |V| / V_DESIGN     (~0.27)
    SIGMA_SWING_MIN = 1.5  # max/min sigma_dt over the ramp  (~4.1)
    INWIN_MIN = 0.95  # min fraction of the bunch captured  (~0.972)
    SENS_CENTROID_MIN = 0.3  # |A - noBeam| centroid / sigma  (~0.92)
    SENS_EMIT_MIN = 0.2  # |A - noBeam| rel. emittance        (~0.64)
    SENS_RATIO_MIN = 50.0  # (A vs noBeam) / (A vs B) centroid (~800)

    _cache: dict = {}

    @classmethod
    def _run_dynamics(
        cls,
        mode: str,
        dt_template: np.ndarray,
        dE_template: np.ndarray,
        cut_left_rad: float,
        cut_right_rad: float,
        intensity: float,
    ) -> dict:
        """
        Track the matched bunch through one ring and record per-turn moments.

        Parameters
        ----------
        mode
            ``"wake"`` (design voltage + multi-pass resonator wake) or
            ``"feedback"`` (matched-bias cavity feedback gap voltage).
        dt_template
            Initial macroparticle arrival times [s].
        dE_template
            Initial macroparticle energy offsets [eV].
        cut_left_rad
            Left edge of the profile window [rad of RF phase].
        cut_right_rad
            Right edge of the profile window [rad of RF phase].
        intensity
            Beam intensity; ``0.0`` gives the design-only reference trajectory.

        Returns
        -------
        dict
            Arrays over turns: ``mean_dt``, ``mean_dE``, ``sigma_dt``,
            ``sigma_dE``, ``emittance``, ``in_window`` (captured fraction) and
            ``v_peak`` (peak ``|induced voltage|`` for ``"wake"``; ``nan``
            otherwise -- the feedback gap voltage still contains the design
            part).
        """
        t_rf = _MatchedBeam._t_rf()
        profile = StaticProfile.from_rad(
            cut_left_rad, cut_right_rad, cls.N_SLICES, t_rf
        )
        if mode == "wake":
            wakefield = WakeField(
                sources=(
                    Resonators(cls.R_OVER_Q * cls.Q_L, 1.0 / t_rf, cls.Q_L),
                ),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12, retune_to_rf=True
                ),
                profile=profile,
            )
            rf = SingleHarmonicRFStation(
                voltage=cls.V_DESIGN,
                phi_rf=0.0,
                harmonic=cls.HARMONIC,
                local_wakefield=wakefield,
                profile=profile,
            )
        elif mode == "feedback":
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=cls.R_OVER_Q,
                Q_L=cls.Q_L,
                generator_current_bias=cls.V_DESIGN
                / (2.0 * cls.R_OVER_Q * cls.Q_L),
                n_cavities=1,
                initial_voltage=cls.V_DESIGN,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
            )
            rf = SingleHarmonicRFStation(
                voltage=cls.V_DESIGN,
                phi_rf=0.0,
                harmonic=cls.HARMONIC,
                cavity_feedback=feedback,
                profile=profile,
            )
        else:  # pragma: no cover - guard against typos
            raise ValueError(f"unknown mode {mode!r}")

        ring = Ring(
            circumference=cls.CIRCUMFERENCE, check_section_indices=False
        )
        ring.add_elements(
            [
                DriftSimple(
                    orbit_length=cls.CIRCUMFERENCE,
                    momentum_compaction_factor=cls.ALPHA_P,
                ),
                rf,
            ],
            reorder=False,
        )
        sim = Simulation(
            ring=ring, magnetic_cycle=_MatchedBeam._accelerating_cycle()
        )
        beam = Beam(intensity=intensity, particle_type=mu_plus)
        beam.reference.total_energy = cls.E0
        beam.setup_beam(dt=dt_template, dE=dE_template)

        rec: dict[str, list] = {
            "mean_dt": [],
            "mean_dE": [],
            "sigma_dt": [],
            "sigma_dE": [],
            "emittance": [],
            "in_window": [],
            "v_peak": [],
        }

        # StaticProfile: the grid never moves, so transfer it once.
        hist_x_host = copy_to_cpu(profile.hist_x)

        def callback(_sim, b):
            dt = copy_to_cpu(b.dt.array_local)
            dE = copy_to_cpu(b.dE.array_local)
            moments = _bunch_moments(dt, dE)
            for key, value in moments.items():
                rec[key].append(value)
            rec["in_window"].append(
                float(np.mean((dt > hist_x_host[0]) & (dt < hist_x_host[-1])))
            )
            rec["v_peak"].append(
                float(np.max(np.abs(copy_to_cpu(wakefield.induced_voltage))))
                if mode == "wake"
                else np.nan
            )

        sim.run_simulation(
            (beam,),
            n_turns=cls.N_TURNS,
            callbacks=callback,
            show_progressbar=False,
        )
        return {key: np.asarray(value) for key, value in rec.items()}

    @classmethod
    def _results(cls) -> dict:
        """
        Run the four simulations once and cache them.

        Returns
        -------
        dict
            ``wake`` / ``feedback`` -- beam-loaded twins;
            ``wake_nobeam`` / ``feedback_nobeam`` -- their zero-intensity
            (design-only) references. Each value is the per-turn record from
            :meth:`_run_dynamics`.
        """
        if "data" in cls._cache:
            return cls._cache["data"]

        dt_t, dE_t, cl, cr = _MatchedBeam._matched_template()
        data = {
            "wake": cls._run_dynamics(
                "wake", dt_t, dE_t, cl, cr, cls.INTENSITY
            ),
            "feedback": cls._run_dynamics(
                "feedback", dt_t, dE_t, cl, cr, cls.INTENSITY
            ),
            "wake_nobeam": cls._run_dynamics("wake", dt_t, dE_t, cl, cr, 0.0),
            "feedback_nobeam": cls._run_dynamics(
                "feedback", dt_t, dE_t, cl, cr, 0.0
            ),
        }
        cls._cache["data"] = data
        return data

    @staticmethod
    def _rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Symmetric per-turn relative difference ``|a - b| / mean(|a|, |b|)``.

        Parameters
        ----------
        a, b
            Per-turn scalar series to compare.

        Returns
        -------
        numpy.ndarray
            The relative difference on each turn.
        """
        return np.abs(a - b) / (0.5 * (np.abs(a) + np.abs(b)))

    # ------------------------------------------------------------------ #
    # main twin-agreement assertions                                     #
    # ------------------------------------------------------------------ #
    def test_centroid_tracks_between_wake_and_feedback(self):
        """The coherent centroid stays far inside ``sigma`` of the twin."""
        data = self._results()
        wake, feedback = data["wake"], data["feedback"]
        for turn in range(self.N_TURNS):
            sigma_dt = 0.5 * (
                wake["sigma_dt"][turn] + feedback["sigma_dt"][turn]
            )
            sigma_dE = 0.5 * (
                wake["sigma_dE"][turn] + feedback["sigma_dE"][turn]
            )
            d_dt = abs(wake["mean_dt"][turn] - feedback["mean_dt"][turn])
            d_dE = abs(wake["mean_dE"][turn] - feedback["mean_dE"][turn])
            with self.subTest(turn=turn):
                self.assertLess(
                    d_dt / sigma_dt,
                    self.CENTROID_GATE,
                    f"turn {turn}: <dt> differs by "
                    f"{d_dt / sigma_dt:.3e} sigma_dt",
                )
                self.assertLess(
                    d_dE / sigma_dE,
                    self.CENTROID_GATE,
                    f"turn {turn}: <dE> differs by "
                    f"{d_dE / sigma_dE:.3e} sigma_dE",
                )

    def test_bunch_shape_tracks_between_wake_and_feedback(self):
        """Per-turn ``sigma_dt`` and ``sigma_dE`` agree to well below 1 %."""
        data = self._results()
        wake, feedback = data["wake"], data["feedback"]
        rel_sigma_dt = self._rel_diff(wake["sigma_dt"], feedback["sigma_dt"])
        rel_sigma_dE = self._rel_diff(wake["sigma_dE"], feedback["sigma_dE"])
        for turn in range(self.N_TURNS):
            with self.subTest(turn=turn):
                self.assertLess(
                    rel_sigma_dt[turn],
                    self.SHAPE_GATE,
                    f"turn {turn}: sigma_dt rel diff {rel_sigma_dt[turn]:.3e}",
                )
                self.assertLess(
                    rel_sigma_dE[turn],
                    self.SHAPE_GATE,
                    f"turn {turn}: sigma_dE rel diff {rel_sigma_dE[turn]:.3e}",
                )

    def test_emittance_difference_stays_small_and_barely_grows(self):
        """
        The rms-emittance difference stays tiny and its per-turn growth is slow.

        A small *slope* is the real content: a per-turn induced-voltage
        mismatch would pump the two trajectories apart, so a bounded emittance
        difference with a shallow slope certifies the models stay phase-locked
        over the whole ramp, not just on turn 0.
        """
        data = self._results()
        wake, feedback = data["wake"], data["feedback"]
        rel_emit = self._rel_diff(wake["emittance"], feedback["emittance"])

        self.assertLess(
            float(np.max(rel_emit)),
            self.EMIT_MAX,
            f"max rms-emittance rel diff {np.max(rel_emit):.3e}",
        )
        turns = np.arange(self.N_TURNS)
        slope = float(np.polyfit(turns, rel_emit, 1)[0])
        self.assertLess(
            abs(slope),
            self.EMIT_SLOPE_GATE,
            f"rms-emittance difference grows {slope:.3e} per turn",
        )

    # ------------------------------------------------------------------ #
    # guards: the comparison is exact where it should be, and non-trivial #
    # ------------------------------------------------------------------ #
    def test_design_kicks_are_identical_without_beam(self):
        """
        With the beam removed the two rings coincide to machine precision.

        Both apply the design kick via ``kick_interpolated`` on the same
        profile grid, so the design part is not a free knob: any beam-loaded
        difference the other tests see is the induced-voltage physics, not a
        design-kick convention mismatch. This is the exactness the energy-gain
        test buys with its voltage-suppression trick, here over the whole
        self-consistent trajectory.
        """
        data = self._results()
        wake0, feedback0 = data["wake_nobeam"], data["feedback_nobeam"]
        rel_emit = self._rel_diff(wake0["emittance"], feedback0["emittance"])
        for turn in range(self.N_TURNS):
            sigma_dt = wake0["sigma_dt"][turn]
            d_dt = abs(wake0["mean_dt"][turn] - feedback0["mean_dt"][turn])
            with self.subTest(turn=turn):
                self.assertLess(
                    d_dt / sigma_dt,
                    self.DESIGN_GATE,
                    f"turn {turn}: design centroids differ by "
                    f"{d_dt / sigma_dt:.3e} sigma",
                )
                self.assertLess(
                    rel_emit[turn],
                    self.DESIGN_GATE,
                    f"turn {turn}: design emittance rel diff "
                    f"{rel_emit[turn]:.3e}",
                )

    def test_beam_loading_is_strong_and_bunch_stays_captured(self):
        """
        Guard against a trivial setup: strong loading, captured bunch.

        The induced voltage is a sizable fraction of ``V_DESIGN``, the bunch
        executes real synchrotron dynamics, and it stays in-window.
        """
        data = self._results()
        wake, feedback = data["wake"], data["feedback"]

        v_peak_frac = float(np.max(wake["v_peak"])) / self.V_DESIGN
        self.assertGreater(
            v_peak_frac,
            self.VPEAK_MIN_FRAC,
            f"induced voltage only {v_peak_frac:.3f} of V_DESIGN -- beam "
            "loading too weak to be a meaningful twin test",
        )

        sigma_swing = float(
            np.max(wake["sigma_dt"]) / np.min(wake["sigma_dt"])
        )
        self.assertGreater(
            sigma_swing,
            self.SIGMA_SWING_MIN,
            f"sigma_dt swing {sigma_swing:.2f} -- bunch is nearly static, "
            "no synchrotron dynamics exercised",
        )

        for label, run in (("wake", wake), ("feedback", feedback)):
            min_in_window = float(np.min(run["in_window"]))
            self.assertGreater(
                min_in_window,
                self.INWIN_MIN,
                f"{label}: only {min_in_window:.3f} of the bunch stayed in "
                "the profile window",
            )

    def test_beam_loading_actually_drives_the_dynamics(self):
        """
        Move the bunch by beam loading far more than the two models differ.

        Sensitivity guard: the beam-loaded run diverges from the design-only
        (no-beam) trajectory by ~0.9 ``sigma`` in the centroid, while the wake
        and feedback twins stay within ~1e-3 ``sigma`` of each other -- so the
        tight agreement is a genuine cross-check of the beam loading, not two
        runs that ignore it.
        """
        data = self._results()
        wake, feedback, wake0 = (
            data["wake"],
            data["feedback"],
            data["wake_nobeam"],
        )

        sens_centroid = float(
            np.max(
                np.abs(wake["mean_dt"] - wake0["mean_dt"]) / wake["sigma_dt"]
            )
        )
        twin_centroid = float(
            np.max(
                np.abs(wake["mean_dt"] - feedback["mean_dt"])
                / (0.5 * (wake["sigma_dt"] + feedback["sigma_dt"]))
            )
        )
        sens_emit = float(
            np.max(self._rel_diff(wake["emittance"], wake0["emittance"]))
        )

        self.assertGreater(
            sens_centroid,
            self.SENS_CENTROID_MIN,
            f"beam loading shifts the centroid only {sens_centroid:.3f} sigma "
            "vs no beam -- too weak to certify the twin agreement",
        )
        self.assertGreater(
            sens_emit,
            self.SENS_EMIT_MIN,
            f"beam loading changes the emittance only {sens_emit:.3f} vs no "
            "beam",
        )
        self.assertGreater(
            sens_centroid / twin_centroid,
            self.SENS_RATIO_MIN,
            "beam-loading centroid shift is not much larger than the "
            f"wake-vs-feedback difference (ratio "
            f"{sens_centroid / twin_centroid:.1f})",
        )

    def _plot(self, data: dict) -> None:  # pragma: no cover - debug only
        """
        Overlay the per-turn moments of the two twins (debug only).

        Parameters
        ----------
        data
            Cached records from :meth:`_results`.
        """
        turns = np.arange(self.N_TURNS)
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for ax, key, ylabel in zip(
            axes,
            ("mean_dt", "sigma_dt", "emittance"),
            ("<dt> [s]", "sigma_dt [s]", "emittance [eV s]"),
            strict=True,
        ):
            ax.plot(turns, data["wake"][key], "o-", label="wake")
            ax.plot(turns, data["feedback"][key], "x--", label="feedback")
            ax.plot(
                turns,
                data["wake_nobeam"][key],
                ":",
                color="0.6",
                label="no beam",
            )
            ax.set_ylabel(ylabel)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("turn")
        fig.suptitle("Wake vs feedback self-consistent dynamics")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def test_debug_plot_opt_in(self):
        """Render the diagnostic overlay when ``DEBUG_PLOT`` is enabled."""
        if not DEBUG_PLOT:
            self.skipTest("DEBUG_PLOT disabled")
        self._plot(self._results())


if __name__ == "__main__":
    unittest.main()
