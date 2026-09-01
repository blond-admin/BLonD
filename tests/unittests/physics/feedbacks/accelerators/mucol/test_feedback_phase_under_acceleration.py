"""
Feedback multi-turn wake phase under acceleration vs an analytic reference.

Validate the feedback's multi-turn wake *phase* under acceleration against an
independent, analytic multi-pass resonator reference.

Under acceleration the RF frequency slips turn to turn, and the solver re-derives the phase of its
whole stored history from a single (current or fixed) frequency each pass.  Over
an inter-turn gap of ``omega * t_rev ~ 2 pi * harmonic`` radians even a ~1e-6
frequency/precision error becomes radians of phase error, so the solver is not a
trustworthy reference once the frame slips.  (It is fine at the ~63 GeV operating
point, where the slip is ~1e-4 ``t_rf`` and the static-profile test passes.)

Instead the induced voltage is rebuilt from first principles.  The resonator
contribution between a source bin emitted at ``t_e`` and an observation bin at
``t_o > t_e`` is ::

    exp(-Phi / (2 Q_L)) * cos(Phi),     Phi = theta(t_o) - theta(t_e),

with the *accumulated* phase ``theta(t) = integral of omega(t) dt`` -- the phase
the cavity field winds up as its resonance tracks the accelerating RF (``omega``
piecewise constant per turn).  This shares none of the feedback's machinery
(I/Q demodulation, coarse-grid downsampling, backfill, sparse-matrix
integration), so agreement is a genuine cross-check.

A real, *matched* ``BiGaussian`` beam is tracked through an accelerating cycle so
the profile follows the slipping frame -- a frozen profile cannot represent the
frame slip and is what makes the dummy-beam comparison ill-posed here.  The
energy sits just above transition (``gamma_t = 1/sqrt(alpha_p) ~ 31`` ->
``~3.28 GeV``): below transition the bunch matcher fails, and much lower the
synchrotron tune blows up so no bunch stays in the window.  At 4 GeV the matched
beam is stable (``Q_s ~ 0.09``) yet the frame slips ~0.09 ``t_rf`` per turn --
orders of magnitude more than at the operating point, enough to expose any
phase-handling error.

The feedback reproduces the integrated-phase reference to <~1 % on every turn
(after a single overall scale calibrated on the first, single-pass turn).  A
fixed-frequency phase treatment diverges by tens to hundreds of percent within a
few turns -- ``test_integrated_phase_is_required`` pins that, so the comparison
is demonstrably sensitive to the accumulated-phase handling it certifies.
"""

import unittest
from copy import copy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    mu_plus,
)
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.drifts import DriftSubstepped
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Package-relative import: the dirs above ``mucol`` have no __init__.py, so the
# shared helpers are not importable by an absolute path under pytest.
from .support import rel_err

DEBUG_PLOT = False


def analytic_multipass_induced_voltage(
    hist_y_per_turn: list[np.ndarray],
    hist_x_per_turn: list[np.ndarray],
    turn_start_times: list[float],
    omegas: list[float],
    q_l: float,
    *,
    fixed_freq: bool,
) -> list[np.ndarray]:
    """
    Multi-pass resonator induced voltage via a phase-clock wake sum.

    Independent of the feedback and of :class:`MultiPassResonatorSolver`: a
    direct double sum of the analytic resonator wake over every past bunch
    passage at its true emission time.  The resonator contribution from a source
    bin to a (later) observation bin is ``exp(-Phi / (2 Q_L)) * cos(Phi)`` with
    the accumulated phase ``Phi = theta(t_o) - theta(t_e)``, where the phase
    clock ``theta(t) = integral of omega(t) dt`` is built from the per-turn
    angular frequency (piecewise constant).  The within-turn self/beam-loading
    contribution uses the ``W(0)/2`` half-wake on the exact diagonal.

    Parameters
    ----------
    hist_y_per_turn
        Per-turn profile histograms (per-bin charge, up to a constant).
    hist_x_per_turn
        Per-turn within-turn bin times [s] (the fixed profile window).
    turn_start_times
        Per-turn reference time ``T_j`` [s]; the absolute time of a bin is
        ``T_j + hist_x``.
    omegas
        Per-turn RF design angular frequency [rad/s].
    q_l
        Loaded quality factor, sets the per-radian decay ``exp(-Phi / 2 Q_L)``.
    fixed_freq
        If ``True`` the phase clock uses a single fixed ``omega_0`` (``theta =
        omega_0 * t``) instead of the time integral -- the *wrong* treatment
        under acceleration, used only to show the test is sensitive to it.

    Returns
    -------
    list of numpy.ndarray
        Per-turn induced voltage on the profile grid, up to one overall real
        scale (the resonator amplitude / charge normalisation), which the caller
        calibrates on the first, single-pass turn.
    """
    n_turns = len(hist_y_per_turn)
    omega = np.asarray(omegas, dtype=float)
    times = np.asarray(turn_start_times, dtype=float)
    omega_0 = omega[0]

    if fixed_freq:
        # theta(bin in turn j) = omega_0 * (T_j + hist_x)
        def theta(j: int) -> np.ndarray:
            return omega_0 * (times[j] + hist_x_per_turn[j])
    else:
        # theta accumulates the per-turn omega over the elapsed reference time
        theta_start = np.concatenate(
            [[0.0], np.cumsum(omega[:-1] * np.diff(times))]
        )

        def theta(j: int) -> np.ndarray:
            return theta_start[j] + omega[j] * hist_x_per_turn[j]

    induced = []
    for k in range(n_turns):
        theta_obs = theta(k)
        acc = np.zeros(len(theta_obs))
        for j in range(k + 1):
            phase = theta_obs[:, None] - theta(j)[None, :]
            wake = np.where(
                phase > 0.0, np.exp(-phase / (2.0 * q_l)) * np.cos(phase), 0.0
            )
            if j == k:  # within-turn self term: W(0)/2 on the exact diagonal
                np.fill_diagonal(wake, 0.5)
            acc += wake @ hist_y_per_turn[j]
        induced.append(acc)
    return induced


class TestFeedbackPhaseUnderAcceleration(unittest.TestCase):
    """
    Feedback multi-turn wake phase vs an analytic reference, under acceleration.

    A real matched beam is accelerated just above transition so the RF frame
    slips meaningfully turn to turn; the feedback's beam-induced voltage is
    compared per turn against :func:`analytic_multipass_induced_voltage`.
    """

    # Machine: RCS1-like single cavity, just above transition (gamma_t ~ 31).
    E0 = 4.0e9
    DELTA_E = 20e6  # < charge_state * V_DESIGN, so phi_s is well defined
    HARMONIC = 25900
    V_DESIGN = 30e6
    R_OVER_Q = 518.0
    Q_L = 1.29e6
    CIRCUMFERENCE = 5990.0
    ALPHA_P = 10.395e-4
    INTENSITY = 2.7e12
    N_SLICES = 1024
    N_TURNS = 5
    N_MACROPARTICLES = 30000
    SIGMA_DT_FRAC = 0.05  # bunch length as a fraction of t_rf
    SEED = 7

    _cache: dict = {}

    @classmethod
    def _t_rf(cls) -> float:
        """
        RF period at the initial energy.

        Returns
        -------
        float
            RF period [s] at ``E0``.
        """
        t_rev = ConstantMagneticCycle(
            reference_particle=mu_plus, value=cls.E0, in_unit="total energy"
        ).get_t_rev_init(cls.CIRCUMFERENCE, particle_type=mu_plus)
        return t_rev / cls.HARMONIC

    @classmethod
    def _accelerating_cycle(cls) -> MagneticCyclePerTurnAllRFStations:
        """
        Per-turn cycle ramping the energy by ``DELTA_E`` each turn.

        Returns
        -------
        MagneticCyclePerTurnAllRFStations
            Cycle starting at ``E0`` and gaining ``DELTA_E`` per turn.
        """
        values = (
            cls.E0 + cls.DELTA_E * np.arange(1, cls.N_TURNS + 1)
        ).reshape(1, cls.N_TURNS, order="F")
        return MagneticCyclePerTurnAllRFStations(
            reference_particle=mu_plus,
            value_init=cls.E0,
            values_after_rf_station_per_turn=values,
            in_unit="total energy",
        )

    @classmethod
    def _matched_template(cls) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Match a ``BiGaussian`` bunch to the accelerating bucket.

        The bunch is matched on the *accelerating* cycle (so it sits at the
        correct synchronous phase, ``phi_s = arcsin(DELTA_E / V_DESIGN)``, with
        the matched energy spread of the moving bucket), then shifted by one RF
        period into a profile window centred on it.

        Returns
        -------
        tuple
            ``(dt_template, dE_template, cut_left_rad, cut_right_rad)``.
        """
        t_rf = cls._t_rf()
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, cls.N_SLICES, t_rf
        )
        ring = Ring(
            circumference=cls.CIRCUMFERENCE, check_section_indices=False
        )
        rf = SingleHarmonicRFStation(
            voltage=cls.V_DESIGN,
            phi_rf=0.0,
            harmonic=cls.HARMONIC,
            profile=profile,
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
        sim = Simulation(ring=ring, magnetic_cycle=cls._accelerating_cycle())
        beam = Beam(intensity=cls.INTENSITY, particle_type=mu_plus)
        beam.reference.total_energy = cls.E0
        # sigma_dE=None -> the matcher computes the matched energy spread.
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=cls.N_MACROPARTICLES,
                sigma_dt=cls.SIGMA_DT_FRAC * t_rf,
                sigma_dE=None,
                seed=cls.SEED,
                reinsertion=True,
            ),
        )
        dt = copy_to_cpu(beam.dt.array_local)
        dE = copy_to_cpu(beam.dE.array_local)
        center = float(np.mean(dt)) + t_rf  # shift one period into the window
        cut_left_rad = (center - 0.75 * t_rf) / t_rf * 2 * np.pi
        cut_right_rad = (center + 0.75 * t_rf) / t_rf * 2 * np.pi
        return dt + t_rf, dE, cut_left_rad, cut_right_rad

    @classmethod
    def _run(
        cls,
        with_beam: bool,
        dt_template: np.ndarray,
        dE_template: np.ndarray,
        cut_left_rad: float,
        cut_right_rad: float,
    ) -> dict:
        """
        Track the matched bunch through the accelerating cycle with the feedback.

        Parameters
        ----------
        with_beam
            If ``False`` the intensity is zero -- the feedback then reconstructs
            only the (decaying) design voltage, which is subtracted from the
            beam run to isolate the beam-induced part by linearity.
        dt_template
            Initial particle arrival times [s].
        dE_template
            Initial particle energy offsets [eV].
        cut_left_rad
            Left edge of the profile window [rad of RF phase].
        cut_right_rad
            Right edge of the profile window [rad of RF phase].

        Returns
        -------
        dict
            Per-turn gap voltage ``V``; and for the beam run also the profile
            histograms ``hy``, within-turn times ``hx``, reference times ``T``,
            relativistic ``beta`` and in-window fractions ``inwin``.
        """
        t_rf = cls._t_rf()
        profile = StaticProfile.from_rad(
            cut_left_rad, cut_right_rad, cls.N_SLICES, t_rf
        )
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=cls.R_OVER_Q,
            Q_L=cls.Q_L,
            generator_current_bias=0.0,
            n_cavities=1,
            initial_voltage=cls.V_DESIGN,  # operating point, keeps the kick check happy
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
        sim = Simulation(ring=ring, magnetic_cycle=cls._accelerating_cycle())
        beam = Beam(
            intensity=(cls.INTENSITY if with_beam else 0.0),
            particle_type=mu_plus,
        )
        beam.reference.total_energy = cls.E0
        beam.setup_beam(dt=dt_template, dE=dE_template)

        rec: dict = {
            "V": [],
            "hy": [],
            "hx": [],
            "T": [],
            "beta": [],
            "inwin": [],
        }

        def callback(_sim, b):
            rec["V"].append(copy_to_cpu(rf.calc_gap_voltage_with_feedbacks()))
            if with_beam:
                hist_x = copy_to_cpu(profile.hist_x)
                rec["hy"].append(copy_to_cpu(profile.hist_y))
                rec["hx"].append(hist_x)
                rec["T"].append(float(b.reference.time))
                rec["beta"].append(float(b.reference.beta))
                dt_local = copy_to_cpu(b.dt.array_local)
                rec["inwin"].append(
                    float(
                        np.mean(
                            (dt_local > hist_x[0]) & (dt_local < hist_x[-1])
                        )
                    )
                )

        sim.run_simulation(
            (beam,),
            n_turns=cls.N_TURNS,
            callbacks=callback,
            show_progressbar=False,
        )
        return rec

    @classmethod
    def _results(cls) -> dict:
        """
        Run both passes once and build the references (cached).

        Returns
        -------
        dict
            ``fb_induced`` (per-turn beam-induced feedback voltage),
            ``ref_integrated`` and ``ref_fixed`` (the two analytic references),
            and the ``frame_slip`` and ``min_in_window`` diagnostics.
        """
        if "data" in cls._cache:
            return cls._cache["data"]

        dt_t, dE_t, cl, cr = cls._matched_template()
        beam_run = cls._run(True, dt_t, dE_t, cl, cr)
        ref_run = cls._run(False, dt_t, dE_t, cl, cr)

        fb_induced = [
            beam_run["V"][k] - ref_run["V"][k] for k in range(cls.N_TURNS)
        ]
        omegas = [
            cls.HARMONIC * 2 * np.pi * c * b / cls.CIRCUMFERENCE
            for b in beam_run["beta"]
        ]
        ref_integrated = analytic_multipass_induced_voltage(
            beam_run["hy"],
            beam_run["hx"],
            beam_run["T"],
            omegas,
            cls.Q_L,
            fixed_freq=False,
        )
        ref_fixed = analytic_multipass_induced_voltage(
            beam_run["hy"],
            beam_run["hx"],
            beam_run["T"],
            omegas,
            cls.Q_L,
            fixed_freq=True,
        )

        t_rev = [cls.CIRCUMFERENCE / (b * c) for b in beam_run["beta"]]
        frame_slip = abs(t_rev[-1] - t_rev[0]) / cls._t_rf()

        data = {
            "fb_induced": fb_induced,
            "ref_integrated": ref_integrated,
            "ref_fixed": ref_fixed,
            "frame_slip": frame_slip,
            "min_in_window": min(beam_run["inwin"]),
        }
        cls._cache["data"] = data
        return data

    @staticmethod
    def _calibrate(target: np.ndarray, reference: np.ndarray) -> float:
        """
        Single real scale matching ``reference`` onto ``target`` (LSQ).

        Parameters
        ----------
        target
            Array to be matched (the feedback voltage).
        reference
            Array to be scaled onto ``target`` (an analytic reference).

        Returns
        -------
        float
            Scale ``<target, reference> / <reference, reference>``.
        """
        return float(np.dot(target, reference) / np.dot(reference, reference))

    def _plot(self, data: dict, scale_int: float, scale_fix: float) -> None:
        """
        Overlay feedback vs both references per turn (debug only).

        Parameters
        ----------
        data
            Cached results from :meth:`_results`.
        scale_int
            Calibration scale for the integrated-phase reference.
        scale_fix
            Calibration scale for the fixed-frequency reference.
        """
        fb = data["fb_induced"]
        fig, axes = plt.subplots(
            self.N_TURNS, 1, figsize=(8, 2.0 * self.N_TURNS), sharex=True
        )
        for k, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(fb[k], lw=1.2, label="feedback (beam-induced)")
            ax.plot(
                scale_int * data["ref_integrated"][k],
                "--",
                lw=1.0,
                label="analytic, integrated phase",
            )
            ax.plot(
                scale_fix * data["ref_fixed"][k],
                ":",
                lw=1.0,
                label="analytic, fixed frequency",
            )
            ax.set_ylabel(f"turn {k}\nV [V]")
        np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("profile bin")
        fig.suptitle(
            "Feedback vs analytic multi-pass reference (accelerating)"
        )
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def test_feedback_matches_analytic_multipass_reference(self):
        """Feedback reproduces the integrated-phase wake on every carried turn."""
        data = self._results()
        fb = data["fb_induced"]
        ref = data["ref_integrated"]
        # One overall scale, fixed on the single-pass turn; turns >=1 then test
        # purely the carried-wake phase propagation.
        scale = self._calibrate(fb[0], ref[0])

        if DEBUG_PLOT:
            self._plot(
                data, scale, self._calibrate(fb[0], data["ref_fixed"][0])
            )

        for k in range(1, self.N_TURNS):
            with self.subTest(turn=k):
                err = rel_err(fb[k], scale * ref[k])
                self.assertLess(
                    err,
                    0.05,
                    f"turn {k}: feedback vs integrated-phase reference "
                    f"rel_err={err:.4f}",
                )

    def test_integrated_phase_is_required(self):
        """A fixed-frequency reference cannot explain the feedback (sensitivity)."""
        data = self._results()
        fb = data["fb_induced"]
        ref_fixed = data["ref_fixed"]
        scale = self._calibrate(fb[0], ref_fixed[0])
        err_last = rel_err(fb[-1], scale * ref_fixed[-1])
        # If this ever drops, the frame stopped slipping (setup broke) or the
        # reference lost its sensitivity to the accumulated phase.
        self.assertGreater(
            err_last,
            0.3,
            f"fixed-frequency reference unexpectedly close (rel_err={err_last:.4f}): "
            "the test is no longer sensitive to the accumulated-phase handling",
        )

    def test_setup_is_in_the_frame_slipping_regime(self):
        """Guard: the frame slips meaningfully and the bunch stays in-window."""
        data = self._results()
        self.assertGreater(
            data["frame_slip"],
            0.2,
            f"frame slip {data['frame_slip']:.3f} t_rf is too small to exercise "
            "the phase handling",
        )
        self.assertGreater(
            data["min_in_window"],
            0.99,
            f"bunch left the profile window (min in-window "
            f"{data['min_in_window']:.3f}); the comparison is degenerate",
        )


class TestSolverPhaseUnderAcceleration(unittest.TestCase):
    """
    ``MultiPassResonatorSolver`` multi-turn wake phase under acceleration.

    Reuses the matched-beam, just-above-transition setup of
    :class:`TestFeedbackPhaseUnderAcceleration` and asserts the retuning solver
    (``delta_f=0.0``) reproduces the same integrated-phase reference the
    feedback does -- i.e. that the solver also accumulates the carried-wake
    phase as ``integral of omega dt`` rather than re-deriving it from the
    current frequency each pass.
    """

    _cache: dict = {}

    @classmethod
    def _run_solver(
        cls,
        dt_template: np.ndarray,
        dE_template: np.ndarray,
        cut_left_rad: float,
        cut_right_rad: float,
    ) -> dict:
        """
        Track the matched bunch through the ramp with a retuning solver.

        Parameters
        ----------
        dt_template
            Initial particle arrival times [s].
        dE_template
            Initial particle energy offsets [eV].
        cut_left_rad
            Left edge of the profile window [rad of RF phase].
        cut_right_rad
            Right edge of the profile window [rad of RF phase].

        Returns
        -------
        dict
            Per-turn induced voltage ``V``, profile histograms ``hy``,
            within-turn times ``hx``, reference times ``T`` and ``beta``.
        """
        base = TestFeedbackPhaseUnderAcceleration
        t_rf = base._t_rf()
        profile = StaticProfile.from_rad(
            cut_left_rad, cut_right_rad, base.N_SLICES, t_rf
        )
        wakefield = WakeField(
            sources=(
                Resonators(base.R_OVER_Q * base.Q_L, 1.0 / t_rf, base.Q_L),
            ),
            solver=MultiPassResonatorSolver(
                decay_fraction_threshold=1e-12, delta_f=0.0
            ),
            profile=profile,
        )
        rf = SingleHarmonicRFStation(
            voltage=base.V_DESIGN,
            phi_rf=0.0,
            harmonic=base.HARMONIC,
            local_wakefield=wakefield,
            profile=profile,
        )
        ring = Ring(
            circumference=base.CIRCUMFERENCE, check_section_indices=False
        )
        ring.add_elements(
            [
                DriftSimple(
                    orbit_length=base.CIRCUMFERENCE,
                    momentum_compaction_factor=base.ALPHA_P,
                ),
                rf,
            ],
            reorder=False,
        )
        sim = Simulation(ring=ring, magnetic_cycle=base._accelerating_cycle())
        beam = Beam(intensity=base.INTENSITY, particle_type=mu_plus)
        beam.reference.total_energy = base.E0
        beam.setup_beam(dt=dt_template, dE=dE_template)

        rec: dict = {"V": [], "hy": [], "hx": [], "T": [], "beta": []}

        def callback(_sim, b):
            rec["V"].append(copy_to_cpu(wakefield.induced_voltage))
            rec["hy"].append(copy_to_cpu(profile.hist_y))
            rec["hx"].append(copy_to_cpu(profile.hist_x))
            rec["T"].append(float(b.reference.time))
            rec["beta"].append(float(b.reference.beta))

        sim.run_simulation(
            (beam,),
            n_turns=base.N_TURNS,
            callbacks=callback,
            show_progressbar=False,
        )
        return rec

    @classmethod
    def _results(cls) -> dict:
        """
        Run the solver once and build the integrated-phase reference (cached).

        Returns
        -------
        dict
            ``solver`` (per-turn induced voltage) and ``ref_integrated`` (the
            analytic integrated-phase reference from the same profiles).
        """
        if "data" in cls._cache:
            return cls._cache["data"]

        base = TestFeedbackPhaseUnderAcceleration
        dt_t, dE_t, cl, cr = base._matched_template()
        run = cls._run_solver(dt_t, dE_t, cl, cr)
        omegas = [
            base.HARMONIC * 2 * np.pi * c * b / base.CIRCUMFERENCE
            for b in run["beta"]
        ]
        ref_integrated = analytic_multipass_induced_voltage(
            run["hy"], run["hx"], run["T"], omegas, base.Q_L, fixed_freq=False
        )
        data = {"solver": run["V"], "ref_integrated": ref_integrated}
        cls._cache["data"] = data
        return data

    def test_solver_matches_analytic_multipass_reference(self):
        """Retuning solver reproduces the integrated-phase wake every turn."""
        data = self._results()
        solver = data["solver"]
        ref = data["ref_integrated"]
        # One overall scale on the single-pass turn; turns >=1 test the carry.
        scale = float(np.dot(solver[0], ref[0]) / np.dot(ref[0], ref[0]))
        for k in range(1, TestFeedbackPhaseUnderAcceleration.N_TURNS):
            with self.subTest(turn=k):
                err = rel_err(solver[k], scale * ref[k])
                self.assertLess(
                    err,
                    0.05,
                    f"turn {k}: solver vs integrated-phase reference "
                    f"rel_err={err:.4f}",
                )


class TestFixedFrequencyWakeWithSubsteppedFrame(unittest.TestCase):
    """
    Fixed-frequency (HOM) multi-pass wake: a sub-stepped frame makes it exact.

    A higher-order mode does not retune with the RF, so its carried-wake phase
    is ``omega_0 * (arrival-time gap)`` -- the phase-clock rotation is
    identically zero and the only acceleration error left is the frame
    (arrival) time. The solver is driven on a *fixed* profile while the
    reference frame is advanced by :class:`DriftSubstepped` (no beam tracking):
    a single-beta frame (``n_substeps=1``) diverges from the analytic
    fixed-frequency reference, while a sub-stepped frame reproduces it to
    machine zero -- so the residual is frame-time granularity, fixable in the
    tracking, not the resonator.
    """

    N_SUBSTEPS_FINE = 512
    N_TURNS = 5

    def setUp(self):
        """Derive the machine quantities from the shared acceleration constants."""
        b = TestFeedbackPhaseUnderAcceleration
        self.E0 = b.E0
        self.harmonic = b.HARMONIC
        self.R_over_Q = b.R_OVER_Q
        self.Q_L = b.Q_L
        self.circumference = b.CIRCUMFERENCE
        self.alpha_p = b.ALPHA_P
        self.intensity = b.INTENSITY
        self.n_slices = b.N_SLICES
        t_rev0 = ConstantMagneticCycle(
            reference_particle=mu_plus, value=self.E0, in_unit="total energy"
        ).get_t_rev_init(self.circumference, particle_type=mu_plus)
        self.t_rf = t_rev0 / self.harmonic
        self.f_res = 1.0 / self.t_rf
        self.omega_0 = 2 * np.pi * self.f_res
        self.ramp_rate = b.DELTA_E / t_rev0  # eV/s

    def _fixed_profile(self) -> StaticProfile:
        """
        Build a static Gaussian bunch with zeroed leading/trailing bins.

        Returns
        -------
        StaticProfile
            Fixed profile driving the solver every pass.
        """
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, self.n_slices, self.t_rf
        )
        t = copy_to_cpu(profile.hist_x)
        t0 = 0.5 * (t[0] + t[-1])
        hist_y = np.exp(-0.5 * ((t - t0) / (0.08 * self.t_rf)) ** 2)
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        return profile

    def _cycle_stub(self) -> SimpleNamespace:
        """
        Build a magnetic cycle stub whose energy ramps linearly with time.

        Returns
        -------
        SimpleNamespace
            Object exposing ``get_target_total_energy`` like a MagneticCycleByTime.
        """
        E0, rate = self.E0, self.ramp_rate

        def get_target_total_energy(
            *, turn_i, section_i, reference_time, particle_type
        ):
            return E0 + rate * reference_time

        return SimpleNamespace(get_target_total_energy=get_target_total_energy)

    def _run(self, n_substeps: int) -> tuple[list, list, StaticProfile]:
        """
        Drive the fixed-frequency solver, advancing the frame with DriftSubstepped.

        Parameters
        ----------
        n_substeps
            Energy-adaptation segments per arc (1 = single-beta frame).

        Returns
        -------
        tuple
            Per-turn ``(reference times, induced voltages, profile)``.
        """
        profile = self._fixed_profile()
        solver = MultiPassResonatorSolver(
            decay_fraction_threshold=1e-12
        )  # delta_f=None -> fixed frequency
        wakefield = WakeField(
            sources=(
                Resonators(self.R_over_Q * self.Q_L, self.f_res, self.Q_L),
            ),
            solver=solver,
            profile=profile,
        )
        solver._parent_wakefield = wakefield
        solver._circumference = self.circumference
        solver._maximum_storage_time = 1.0  # >> run time, keeps every pass
        solver._last_reference_time = -np.finfo(float).eps
        drift = DriftSubstepped(
            orbit_length=self.circumference,
            n_substeps=n_substeps,
            momentum_compaction_factor=self.alpha_p,
        )
        drift.configure(
            turn_counter=SimpleNamespace(value=0),
            magnetic_cycle=self._cycle_stub(),
        )
        beam = SimpleNamespace(
            reference=ReferenceCoordinates(0.0, self.E0, mu_plus),
            intensity=self.intensity,
            particle_type=mu_plus,
            is_counter_rotating=False,
            # Direction-signed charge read by the solver's source side;
            # co-rotating, so it is the plain particle charge.
            signed_charge_with_direction=lambda: mu_plus.charge,
        )
        times, voltages = [], []
        for _ in range(self.N_TURNS):
            drift.track_reference(beam.reference)
            voltages.append(copy_to_cpu(solver.calc_induced_voltage(beam)))
            times.append(beam.reference.time)
        return times, voltages, profile

    def test_substepped_frame_makes_fixed_frequency_wake_exact(self):
        """Sub-stepped frame reproduces the analytic wake; the coarse frame does not."""
        t_coarse, v_coarse, profile = self._run(1)
        t_fine, v_fine, _ = self._run(self.N_SUBSTEPS_FINE)

        # the frame slips meaningfully between coarse and fine (non-trivial setup)
        self.assertGreater(abs(t_coarse[-1] - t_fine[-1]) / self.t_rf, 0.1)

        # analytic fixed-frequency reference from the true (fine) arrival times
        reference = analytic_multipass_induced_voltage(
            [copy_to_cpu(profile.hist_y)] * self.N_TURNS,
            [copy_to_cpu(profile.hist_x)] * self.N_TURNS,
            t_fine,
            [self.omega_0] * self.N_TURNS,
            self.Q_L,
            fixed_freq=True,
        )
        scale_fine = float(
            np.dot(v_fine[0], reference[0])
            / np.dot(reference[0], reference[0])
        )
        scale_coarse = float(
            np.dot(v_coarse[0], reference[0])
            / np.dot(reference[0], reference[0])
        )

        # the sub-stepped frame reproduces the analytic wake on every carried turn
        for k in range(1, self.N_TURNS):
            with self.subTest(turn=k):
                self.assertLess(
                    rel_err(v_fine[k], scale_fine * reference[k]), 0.01
                )
        # the single-beta frame is far off -- the frame time is the cause
        self.assertGreater(
            rel_err(v_coarse[-1], scale_coarse * reference[-1]), 0.3
        )


if __name__ == "__main__":
    unittest.main()
