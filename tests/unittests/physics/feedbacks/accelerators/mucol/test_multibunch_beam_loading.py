"""
Multi-bunch / multiple-populated-coarse-cell transient beam loading.

Every other feedback test in this package holds a single bunch in a ~1.5
``t_rf`` window, so only 1-2 of the ~25900 coarse cells are ever populated and
the intra-turn wake propagation *between* well-separated populated coarse cells
is never exercised. That propagation is the mu+/mu- offset-passage code path and
the design multi-bunch case: a trailing bunch sees the decaying wake of the
leading bunch several RF periods earlier.

This module widens the profile window to ~8 ``t_rf`` and places two (single-pass
also three) Gaussian bunches at uneven spacings -- centres at 2 and 6 ``t_rf`` --
so several coarse cells are populated with an empty gap between them. It then
runs the same frozen-profile comparison as
``test_mtw_vs_nondriven_feedback.py`` (``profile.active = False``): the
beam-induced voltage of the non-driven :class:`IQCavityFeedbackTimingClass` is
compared against the :class:`MultiPassResonatorSolver` convolution of the *same*
cavity (``R_shunt = R_over_Q * Q_L``, ``f_res = 1 / t_rf``).

Two classes, mirroring the reference module:

* :class:`TestSinglePassMultiBunch` -- single pass, driven directly on one
  static multi-bunch profile (no ``Beam``/``Simulation``). Solver vs feedback
  induced voltage, two and three bunches.
* :class:`TestMultiBunchMultiTurn` -- full ``Simulation``, dummy particle-less
  beam, single section, static then transition-adjacent fast ramp. The
  feedback's coarse grid propagates turn over turn and its beam-induced gap
  voltage (beam run minus a no-beam reference run, isolating the beam-induced
  part by linearity of the cavity equation) is compared per turn against the
  accumulating multi-pass convolution.

**The load-bearing assertion is local, not global.** The physics of interest is
the trailing-bunch voltage *sag* from the leading bunch's persistent wake
(``f_res = 1 / t_rf`` and ``Q_L`` large, so the wake is essentially undamped over
the window and adds in phase; the trailing-bunch peak is ~2x the leading one on a
single pass). A global L2 match can hide a local trailing-bunch error, so every
test slices the window around the second bunch and asserts the relative error
*there*. The local gates are anchored a few x above the measured discretization
floor (trailing/leading/global feedback-vs-solver ~0.19 %/0.46 %/0.11 %), not a
loose 2 %, so a sub-percent error in the carried inter-cell wake fails.

To keep the first coarse cell charge-free (the
``forbid_charge_in_first_coarse_cell`` guard in
:func:`~blond.physics.feedbacks.beam_current.rf_beam_current`, which the
timing class enforces because it seeds the fine-grid initial antenna voltage from
that cell), the profile is zeroed below :data:`ZERO_BELOW_TRF` ``t_rf`` -- well
below the leading bunch's left tail (>6 sigma), so no physical charge is lost.
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
    backend,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.beam_current import rf_beam_current
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Import the reference *module* (not the class). Binding
# ``TestMultiTurnFeedbackVsConvolution`` -- a unittest.TestCase subclass -- to a
# module-level name here would make pytest re-collect its whole suite under this
# module. Reusing it via attribute access on the module keeps its cycle /
# harmonic / regime helpers and cavity constants available without that
# duplicate collection.
from . import test_mtw_vs_nondriven_feedback as _mtw_reference

# Package-relative imports: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .stubs import StubBeam, StubRFStation
from .support import lab_frame_voltage, rel_err

DEBUG_PLOT = False

# Multi-bunch profile geometry, in units of t_rf.
WINDOW_TRF = (0.5, 8.5)  # profile window [cut_left, cut_right], ~8 t_rf wide
TWO_BUNCH_CENTERS_TRF = (2.0, 6.0)  # uneven spacing: 4 t_rf apart, off-centre
THREE_BUNCH_CENTERS_TRF = (2.0, 4.0, 7.0)  # uneven: 2 then 3 t_rf gaps
SIGMA_TRF = 0.08  # bunch rms length
NOISE_LEVEL = 0.03  # additive profile noise (relative to bunch peak)
# Zero all charge below this (t_rf) so the first coarse cell stays empty. The
# leading bunch centre is >= 2 t_rf, so this cuts >6 sigma of its left tail:
# negligible charge, but it guarantees the forbid_charge_in_first_coarse_cell
# guard of the timing class is satisfied whatever the exact coarse alignment.
ZERO_BELOW_TRF = 1.5

# Fine-grid resolution: ~512 bins per t_rf over the ~8 t_rf window. Enough
# samples per RF period for the forward-Euler cavity response to agree with the
# convolution to < 0.2 % (checked below); the reference module uses a comparable
# ~683 bins/t_rf on its 1.5 t_rf window.
N_SLICES = 4096


def make_multibunch_profile(
    t_rf: float,
    n_slices: int,
    centers_trf: tuple[float, ...] = TWO_BUNCH_CENTERS_TRF,
    section_index: int = 0,
    seed: int = 12345,
) -> StaticProfile:
    """
    Build a multi-bunch noisy static profile spanning ~8 RF periods.

    Gaussian bunches of equal charge at ``centers_trf`` (in units of ``t_rf``)
    on a shared noisy floor, with all charge below :data:`ZERO_BELOW_TRF`
    ``t_rf`` (and the trailing edge bins) zeroed so the first coarse-grid cell
    is charge-free and the resonator solver stays stable at the edges.

    Parameters
    ----------
    t_rf
        RF period; sets the profile window and the bunch positions/widths.
    n_slices
        Number of profile bins.
    centers_trf
        Bunch-centre positions, in units of ``t_rf``.
    section_index
        Section the profile belongs to (also offsets the noise seed).
    seed
        Base RNG seed for the additive noise.

    Returns
    -------
    StaticProfile
        The multi-bunch static profile.
    """
    profile = StaticProfile.from_rad(
        WINDOW_TRF[0] * 2 * np.pi,
        WINDOW_TRF[1] * 2 * np.pi,
        n_slices,
        t_rf,
        section_index=section_index,
    )
    t = copy_to_cpu(profile.hist_x)
    sigma = SIGMA_TRF * t_rf

    rng = np.random.default_rng(seed + section_index)
    hist_y = np.zeros(n_slices)
    for center in centers_trf:
        hist_y = hist_y + np.exp(-0.5 * ((t - center * t_rf) / sigma) ** 2)
    hist_y = hist_y + NOISE_LEVEL * rng.standard_normal(n_slices)
    hist_y = np.clip(hist_y, 0.0, None)
    # Keep the first coarse cell empty and the edges quiet.
    hist_y[t < ZERO_BELOW_TRF * t_rf] = 0.0
    hist_y[-5:] = 0.0

    profile._hist_y = backend.array(hist_y, dtype=backend.float)
    profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
    return profile


def bunch_window_mask(
    hist_x: np.ndarray,
    t_rf: float,
    center_trf: float,
    half_width_trf: float = 0.5,
) -> np.ndarray:
    """
    Boolean mask selecting the fine-grid bins around one bunch.

    Parameters
    ----------
    hist_x
        Fine-grid time coordinates.
    t_rf
        RF period.
    center_trf
        Bunch centre, in units of ``t_rf``.
    half_width_trf
        Half-width of the window, in units of ``t_rf``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` inside ``center +- half_width`` RF periods.
    """
    return (
        np.abs(copy_to_cpu(hist_x) - center_trf * t_rf)
        <= half_width_trf * t_rf
    )


class TestSinglePassMultiBunch(unittest.TestCase):
    """
    Single-pass multi-bunch induced voltage: solver vs non-driven feedback.

    Drives the multi-pass solver and the non-driven feedback directly on one
    static multi-bunch profile (no ``Beam`` tracking, no ``Simulation``) and
    checks the lab-frame induced voltage, both globally and -- the load-bearing
    check -- locally at the trailing bunch, where the leading bunch's persistent
    wake dominates the sag.
    """

    def setUp(self):
        """Shared RCS1-like single-cavity parameters and a two-bunch profile."""
        self.R_over_Q = 518.0
        self.Q_L = 1.29e4
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.f_res = 1.0 / self.t_rf
        self.circumference = 5990.0
        self.intensity = 2.7e12

        self.profile = make_multibunch_profile(self.t_rf, N_SLICES)
        self.stub_beam = StubBeam(self.intensity)

    def _solver_induced_voltage(self, profile) -> np.ndarray:
        """
        Lab-frame induced voltage from a single pass of the multi-turn solver.

        Parameters
        ----------
        profile
            Static profile to drive the solver with.

        Returns
        -------
        numpy.ndarray
            Lab-frame induced voltage on the profile's fine grid.
        """
        resonator = Resonators(
            shunt_impedances=self.R_over_Q * self.Q_L,
            center_frequencies=self.f_res,
            quality_factors=self.Q_L,
        )
        solver = MultiPassResonatorSolver(
            decay_fraction_threshold=1e-12, retune_to_rf=True
        )
        wakefield = WakeField(
            sources=(resonator,),
            solver=solver,
            profile=profile,
            parent_rf_station=StubRFStation(self.omega_rf),
        )
        solver._parent_wakefield = wakefield
        solver._circumference = self.circumference
        solver._maximum_storage_time = 1.0  # >> t_rf, keeps the single pass
        solver._last_reference_time = -np.finfo(float).eps
        return copy_to_cpu(solver.calc_induced_voltage(self.stub_beam))

    def _feedback_induced_voltage(self, profile) -> np.ndarray:
        """
        Lab-frame beam-induced voltage from the non-driven feedback.

        Parameters
        ----------
        profile
            Static profile to drive the feedback with.

        Returns
        -------
        numpy.ndarray
            Lab-frame beam-induced voltage on the profile's fine grid.
        """
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=0.0,  # non-driven: no generator current
            n_cavities=1,
            initial_voltage=0.0,  # antenna voltage starts at zero
            n_rf_periods_per_coarse_grid=1,
            delta_omega=0.0,
        )
        charges_fine = rf_beam_current(
            beam=self.stub_beam,
            profile=profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
        )
        feedback.beam_current_fine_grid = charges_fine / profile.hist_step
        feedback.generator_current_fine_grid = np.zeros_like(
            feedback.beam_current_fine_grid
        )
        feedback.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_generator_current_fine_grid=0.0,
            omega_times_dt_fine_grid=(self.omega_rf * profile.hist_step),
            relative_detuning=0.0,
        )
        return lab_frame_voltage(
            feedback.antenna_voltage_fine_grid, self.omega_rf, profile.hist_x
        )

    def _plot(self, profile, v_solver, v_feedback, title):
        """
        Debug plot of the induced voltage along the bunch train.

        Parameters
        ----------
        profile
            Multi-bunch profile whose fine grid sets the time axis.
        v_solver
            Lab-frame induced voltage of the multi-pass solver.
        v_feedback
            Lab-frame beam-induced voltage of the feedback.
        title
            Figure title.
        """
        t_ns = profile.hist_x * 1e9
        fig, (ax_v, ax_d) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        fig.suptitle(title)
        ax_v.plot(t_ns, v_solver, color="C0", label="solver")
        ax_v.plot(t_ns, v_feedback, color="C1", ls="--", label="feedback")
        ax_v.set_ylabel("induced voltage [V]")
        ax_v.legend(loc="best")
        ax_d.plot(t_ns, v_feedback - v_solver, color="C3")
        ax_d.set_ylabel("feedback - solver [V]")
        ax_d.set_xlabel("time [ns]")
        fig.tight_layout()
        plt.show()

    def test_two_bunch_trailing_matches_solver(self):
        """
        Two bunches: feedback vs solver, globally and at the trailing bunch.

        The trailing bunch (6 t_rf) rides the leading bunch's (2 t_rf)
        essentially undamped, in-phase wake, so its induced-voltage peak is ~2x
        the leading one. The feedback reproduces the whole train -- and, in
        particular, the trailing-bunch sag -- to a few x above the ~0.19 %
        discretization floor (observed trailing 0.19 %, leading 0.46 %, global
        0.11 %); the gates are set at 0.6 %/1.0 %/0.3 % respectively.
        """
        v_solver = self._solver_induced_voltage(self.profile)
        v_feedback = self._feedback_induced_voltage(self.profile)
        if DEBUG_PLOT:
            self._plot(self.profile, v_solver, v_feedback, "two bunches")

        peak = np.max(np.abs(v_solver))
        self.assertGreater(peak, 0.0)

        lead = bunch_window_mask(
            self.profile.hist_x, self.t_rf, TWO_BUNCH_CENTERS_TRF[0]
        )
        trail = bunch_window_mask(
            self.profile.hist_x, self.t_rf, TWO_BUNCH_CENTERS_TRF[1]
        )

        # The physics under test: the trailing bunch is enhanced by the
        # leading bunch's persistent wake. Guards against a degenerate profile
        # where the "local" check would be trivially small.
        lead_peak = np.max(np.abs(v_solver[lead]))
        trail_peak = np.max(np.abs(v_solver[trail]))
        self.assertGreater(trail_peak / lead_peak, 1.5)

        # Load-bearing: the trailing-bunch voltage matches locally. Gates are
        # anchored ~3x above the measured floor so a real inter-cell-wake
        # regression (>~0.6 %) fails here rather than hiding under a loose 2 %.
        self.assertLess(rel_err(v_feedback[trail], v_solver[trail]), 0.006)
        # And the leading bunch and the whole train agree too.
        self.assertLess(rel_err(v_feedback[lead], v_solver[lead]), 0.010)
        self.assertLess(rel_err(v_feedback, v_solver), 0.003)

    def test_three_bunch_trailing_matches_solver(self):
        """
        Three unevenly-spaced bunches: local match at the last bunch.

        With bunches at 2, 4 and 7 t_rf the last bunch integrates the wake of
        two upstream bunches at two different lags; the feedback still tracks
        the solver locally to a few x above the ~0.12 % floor (gate 0.6 %).
        """
        profile = make_multibunch_profile(
            self.t_rf, N_SLICES, centers_trf=THREE_BUNCH_CENTERS_TRF
        )
        v_solver = self._solver_induced_voltage(profile)
        v_feedback = self._feedback_induced_voltage(profile)
        if DEBUG_PLOT:
            self._plot(profile, v_solver, v_feedback, "three bunches")

        first = bunch_window_mask(
            profile.hist_x, self.t_rf, THREE_BUNCH_CENTERS_TRF[0]
        )
        last = bunch_window_mask(
            profile.hist_x, self.t_rf, THREE_BUNCH_CENTERS_TRF[-1]
        )
        # Wake accumulates down the train: the third bunch is enhanced above
        # the first.
        self.assertGreater(
            np.max(np.abs(v_solver[last])) / np.max(np.abs(v_solver[first])),
            1.5,
        )
        self.assertLess(rel_err(v_feedback[last], v_solver[last]), 0.006)
        self.assertLess(rel_err(v_feedback, v_solver), 0.003)

    def test_first_coarse_cell_precondition(self):
        """
        The multi-bunch profile leaves the *first coarse cell* charge-free.

        Exercises the real invariant the timing class relies on by driving the
        *actual* mucol coarse-grid downsampling,
        the coarse path of
        :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` with the
        first-coarse-cell guard on (as the forward pass calls it -- it seeds the
        fine-grid initial antenna voltage from the first coarse cell and
        hard-enforces that it stay
        charge-free), rather than re-reading the profile builder's hard-zeroed
        *fine* region. It returns normally and the first *coarse* cell carries
        negligible charge; a coarse-alignment regression that spilled charge into
        it (the ``ValueError`` path of
        ``test_mtw_vs_nondriven_feedback.test_multiturn_nondivisible_harmonic``)
        would raise here. The guard's teeth themselves are covered by
        ``test_helpers.py::test_error_when_first_coarse_cell_populated``.
        """
        # Drive the real forward-pass coarse path (n_rf_periods_per_coarse_grid
        # = 1 -> sampling_time = t_rf); n_points comfortably exceeds the ~8
        # coarse indices the ~8 t_rf window spans. Returning without raising
        # *is* the load-bearing check.
        charges_fine, charges_coarse = rf_beam_current(
            beam=self.stub_beam,
            profile=self.profile,
            omega_c=self.omega_rf,
            sampling_time=self.t_rf,
            n_points=int(np.ceil(WINDOW_TRF[1])) + 4,
            dT=0.0,
            forbid_charge_in_first_coarse_cell=True,
        )
        # Explicit companion check, normalised exactly as the guard is
        # (against the total *fine* charge): the first coarse cell -- whose beam
        # kick would be double-counted by the fine-grid seed -- is empty.
        total_charge = np.sum(np.abs(charges_fine))
        self.assertGreater(total_charge, 0.0)
        self.assertLess(np.abs(charges_coarse[0]) / total_charge, 1e-9)


class TestMultiBunchMultiTurn(unittest.TestCase):
    """
    Full-Simulation multi-turn multi-bunch comparison.

    A dummy beam without macroparticles holds the static multi-bunch profile
    (``profile.active = False``); the feedback's coarse grid propagates turn over
    turn and its beam-induced gap voltage (beam run minus no-beam reference run)
    is compared per turn against the accumulating multi-pass convolution. Single
    section, static then transition-adjacent fast ramp.

    The machinery (harmonic, ``t_rf``, magnetic cycle, high ``Q_L`` so the
    previous-pass wake survives) is reused from
    :class:`.test_mtw_vs_nondriven_feedback.TestMultiTurnFeedbackVsConvolution`;
    only the profile is multi-bunch. Each config runs three full simulations, so
    a class-level cache keyed on the config avoids re-running shared setups.
    """

    # Own cache (same pattern as the reference class), keyed on
    # (acceleration, fast_ramp): each config is three full simulations.
    _multiturn_cache: dict = {}

    @classmethod
    def _run_case(
        cls, mode: str, acceleration: bool, fast_ramp: bool = False
    ) -> list:
        """
        Run one full single-section multi-turn Simulation, collect per turn.

        Adapted from
        :meth:`.test_mtw_vs_nondriven_feedback.TestMultiTurnFeedbackVsConvolution._run_multiturn_case`
        (single section) with the multi-bunch profile builder. The cavity
        parameters, harmonic, ``t_rf``, magnetic cycle and regime come from the
        reused base class so this stays consistent with the single-bunch tests.

        Parameters
        ----------
        mode
            ``"mtw"`` (convolution wakefield), ``"fb"`` (feedback with beam) or
            ``"fb_reference"`` (feedback with zero intensity, to isolate the
            beam-induced part by linearity).
        acceleration
            If True, run with the accelerating cycle.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime (implies
            acceleration); the convolution reference then retunes
            with the RF (``retune_to_rf=True``, zero offset),
            matching the feedback's on-resonance cavity.

        Returns
        -------
        list
            Per turn, a one-element list (single section) with the wakefield
            induced voltage (``"mtw"``) or the station gap voltage otherwise.
        """
        base = _mtw_reference.TestMultiTurnFeedbackVsConvolution
        n_sections = 1
        harmonic, t_rf = base._calc_multiturn_harmonic_and_t_rf(
            n_sections, fast_ramp=fast_ramp
        )
        energy, n_turns = base._regime(fast_ramp)
        half_drift_length = base.MULTITURN_CIRCUMFERENCE / n_sections / 2

        ring = Ring(
            circumference=base.MULTITURN_CIRCUMFERENCE,
            check_section_indices=False,
        )
        profile = make_multibunch_profile(t_rf, N_SLICES, section_index=0)
        profile.active = False  # keep the histogram static (no particles)

        if mode == "mtw":
            solver_kwargs = {
                "decay_fraction_threshold": 1e-12,
                # Retuning solver: follows the RF frequency turn by turn,
                # matching the feedback's always-on-resonance cavity.
                "retune_to_rf": fast_ramp,
            }
            local_wf = WakeField(
                sources=(
                    Resonators(
                        base.MULTITURN_R_OVER_Q * base.MULTITURN_Q_L,
                        1.0 / t_rf,
                        base.MULTITURN_Q_L,
                    ),
                ),
                solver=MultiPassResonatorSolver(**solver_kwargs),
                profile=profile,
            )
            rf_station = SingleHarmonicRFStation(
                voltage=base.MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=harmonic,
                local_wakefield=local_wf,
                profile=profile,
                section_index=0,
            )
            collected = local_wf
        else:
            # Operating-point cavity (V_init = V_design): a cold start trips the
            # coarse-grid beam-kick magnitude check.
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=base.MULTITURN_R_OVER_Q,
                Q_L=base.MULTITURN_Q_L,
                generator_current_bias=0.0,
                n_cavities=1,
                initial_voltage=base.MULTITURN_V_DESIGN,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
            )
            rf_station = SingleHarmonicRFStation(
                voltage=base.MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=harmonic,
                cavity_feedback=feedback,
                profile=profile,
                section_index=0,
            )
            collected = rf_station

        ring.add_elements(
            [
                DriftSimple(
                    orbit_length=half_drift_length,
                    momentum_compaction_factor=base.MULTITURN_ALPHA_P,
                    section_index=0,
                ),
                rf_station,
                DriftSimple(
                    orbit_length=half_drift_length,
                    momentum_compaction_factor=base.MULTITURN_ALPHA_P,
                    section_index=0,
                ),
            ],
            reorder=False,
        )
        sim = Simulation(
            ring=ring,
            magnetic_cycle=base._multiturn_cycle(
                n_sections, acceleration, fast_ramp=fast_ramp
            ),
        )

        beam = Beam(
            intensity=(
                0.0 if mode == "fb_reference" else base.MULTITURN_INTENSITY
            ),
            particle_type=mu_plus,
        )
        beam.reference.total_energy = energy
        beam.setup_beam(dt=np.array([]), dE=np.array([]))

        per_turn = []

        def collect(simulation, beam_in_callback):
            if mode == "mtw":
                per_turn.append([copy_to_cpu(collected.induced_voltage)])
            else:
                per_turn.append(
                    [copy_to_cpu(collected.calc_gap_voltage_with_feedbacks())]
                )

        sim.run_simulation(
            (beam,),
            n_turns=n_turns,
            callbacks=collect,
            show_progressbar=False,
        )
        return per_turn

    @classmethod
    def _feedback_vs_convolution(cls, acceleration: bool, fast_ramp: bool):
        """
        Run (once per config) and cache the convolution and feedback voltages.

        Parameters
        ----------
        acceleration
            If True, run with the accelerating cycle.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime.

        Returns
        -------
        convolution
            ``[turn][0]`` induced voltage of the multi-pass convolution.
        feedback
            ``[turn][0]`` beam-induced voltage of the feedback (beam run gap
            voltage minus the no-beam reference run).
        """
        key = (acceleration, fast_ramp)
        if key not in cls._multiturn_cache:
            convolution = cls._run_case("mtw", acceleration, fast_ramp)
            gap_beam = cls._run_case("fb", acceleration, fast_ramp)
            gap_reference = cls._run_case(
                "fb_reference", acceleration, fast_ramp
            )
            feedback = [
                [
                    beam_section - reference_section
                    for beam_section, reference_section in zip(
                        gap_beam_turn, gap_reference_turn, strict=True
                    )
                ]
                for gap_beam_turn, gap_reference_turn in zip(
                    gap_beam, gap_reference, strict=True
                )
            ]
            cls._multiturn_cache[key] = (convolution, feedback)
        return cls._multiturn_cache[key]

    def _assert_trailing_bunch_consistency(
        self,
        acceleration: bool,
        fast_ramp: bool,
        *,
        trail_tol: float = 0.006,
        lead_tol: float = 0.010,
        global_tol: float = 0.003,
    ):
        """
        Per-turn local trailing-bunch agreement of feedback vs convolution.

        Gates default to a few x above the measured per-turn floors (trailing
        <=0.19 %, leading <=0.46 %, global <=0.12 %, both configs), tightened
        from a loose 2 % so a real sub-percent carried-wake regression fails.
        They are parametrised so a future regime that legitimately needs a
        looser bar can raise them explicitly rather than blanket-loosening.

        Parameters
        ----------
        acceleration
            If True, run with the accelerating cycle.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime.
        trail_tol, lead_tol, global_tol
            Per-turn relative-error gates for the trailing bunch, leading bunch
            and whole train.
        """
        convolution, feedback = self._feedback_vs_convolution(
            acceleration, fast_ramp
        )
        base = _mtw_reference.TestMultiTurnFeedbackVsConvolution
        _, t_rf = base._calc_multiturn_harmonic_and_t_rf(
            1, fast_ramp=fast_ramp
        )
        hist_x = make_multibunch_profile(t_rf, N_SLICES).hist_x
        lead = bunch_window_mask(hist_x, t_rf, TWO_BUNCH_CENTERS_TRF[0])
        trail = bunch_window_mask(hist_x, t_rf, TWO_BUNCH_CENTERS_TRF[1])

        if DEBUG_PLOT:
            self._plot_multiturn(
                [turn[0] for turn in convolution],
                [turn[0] for turn in feedback],
            )

        for turn_i, (conv_turn, fb_turn) in enumerate(
            zip(convolution, feedback, strict=True)
        ):
            v_conv, v_fb = conv_turn[0], fb_turn[0]
            # Global sanity, then the load-bearing local trailing-bunch check.
            self.assertLess(
                rel_err(v_fb, v_conv), global_tol, f"global turn {turn_i}"
            )
            self.assertLess(
                rel_err(v_fb[trail], v_conv[trail]),
                trail_tol,
                f"trailing bunch turn {turn_i}",
            )
            self.assertLess(
                rel_err(v_fb[lead], v_conv[lead]),
                lead_tol,
                f"leading bunch turn {turn_i}",
            )

        # First turn == single pass: the trailing bunch rides the leading
        # bunch's undamped wake, so its peak is clearly enhanced. Later turns
        # sit on a common accumulated pedestal, which levels the ratio.
        v_conv_turn0 = convolution[0][0]
        self.assertGreater(
            np.max(np.abs(v_conv_turn0[trail]))
            / np.max(np.abs(v_conv_turn0[lead])),
            1.5,
        )

    def test_multibunch_static(self):
        """
        Static single-section: intra-turn wake between two populated cells.

        The leading and trailing bunches occupy well-separated coarse cells; the
        feedback's coarse grid must carry the leading-bunch wake across the empty
        gap to the trailing cell. Matches the convolution locally at the trailing
        bunch to < 2 % on every turn.
        """
        self._assert_trailing_bunch_consistency(
            acceleration=False, fast_ramp=False
        )

    def test_multibunch_fast_ramp(self):
        """
        Transition-adjacent fast ramp: multi-bunch carried wake holds.

        At ~4 GeV (gamma_t ~ 31) the RF frame slips ~0.09 t_rf per turn, so the
        coarse grid is rebuilt each turn and the two populated cells shift
        relative to it. The convolution reference retunes with the RF
        (``retune_to_rf=True``). The trailing-bunch beam loading
        still tracks the convolution locally to < 2 % across all
        turns.
        """
        self._assert_trailing_bunch_consistency(
            acceleration=True, fast_ramp=True
        )

    def _plot_multiturn(self, v_convolution_turns, v_feedback_turns):
        """
        Debug plot: per-turn convolution vs feedback induced voltage.

        Parameters
        ----------
        v_convolution_turns
            Per-turn induced voltage of the multi-pass convolution.
        v_feedback_turns
            Per-turn beam-induced voltage of the feedback.
        """
        fig, axes = plt.subplots(
            len(v_convolution_turns), 1, sharex=True, figsize=(9, 9)
        )
        for turn_i, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(v_convolution_turns[turn_i], color="C0", label="conv")
            ax.plot(
                v_feedback_turns[turn_i], color="C1", ls="--", label="feedback"
            )
            ax.set_ylabel(f"turn {turn_i} [V]")
        np.atleast_1d(axes)[0].legend(loc="best")
        np.atleast_1d(axes)[-1].set_xlabel("profile bin")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
