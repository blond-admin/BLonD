"""
Compare the multi-turn induced voltage with a non-driven cavity feedback.

A single static profile (a noisy Gaussian with zeroed leading/trailing bins)
drives two models of the *same* single cavity
(``R_shunt = R_over_Q * Q_L``, ``f_res = 1 / t_rf``):

* a :class:`MultiPassResonatorSolver` -- the multi-turn resonator convolution,
  evaluated for a single pass, and
* a *non-driven* :class:`IQCavityFeedbackTimingClass` -- generator current
  ``I_g = 0``, initial antenna voltage ``V_init = 0`` and ``n_cavities = 1``.

With the beam as the only excitation, the feedback's antenna voltage is purely
the beam-induced voltage. Both objects are driven directly on the static
profile; no ``Beam`` tracking and no full ``Simulation`` run is required, which
mirrors the mock/patch style of ``test_mucol_cav_fdbk.py``.

The feedback works with the complex (I/Q) envelope of the antenna voltage,
while the solver returns the real, lab-frame induced voltage. Projecting the
envelope back to the lab frame recovers the solver result to < 1 %::

    v_induced_solver  ~=  -Im[ V_ant * exp(i * omega_rf * t) ]

The 90-degree rotation is the ``exp(i * pi / 2)`` demodulation convention
applied inside :func:`rf_beam_current`, which both the feedback and this test
go through. Over the bunch window the cavity decay (``~ exp(-omega t / Q_L)``
with ``Q_L ~ 1e6``) is negligible, so the two formulations agree to the
discretization error of the forward-Euler cavity response.

On top of the single-pass comparison, the genuinely *multi-turn* regime is
covered by a full-``Simulation`` setup (static ``ConstantMagneticCycle``,
drift + RF station, a dummy beam without macroparticles): the same noisy
profile is held static (``profile.active = False``), the feedback propagates
its coarse grid turn over turn, and its gap voltage -- minus a no-beam
reference run, which isolates the beam-induced part by linearity of the
cavity equation -- is compared per turn against the accumulating
multi-pass convolution voltage.
"""

import unittest
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.helpers import rf_beam_current
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Package-relative imports: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .stubs import StubBeam, StubRFStation
from .support import (
    lab_frame_voltage,
    rel_err,
)

DEBUG_PLOT = False


class TestMultiTurnInducedVoltageVsNonDrivenFeedback(unittest.TestCase):
    """Multi-turn induced voltage vs a non-driven IQ cavity feedback."""

    def setUp(self):
        """Build a noisy-Gaussian static profile and shared cavity parameters."""
        # RCS1-like single-cavity parameters.
        self.R_over_Q = 518.0
        self.Q_L = 1.29e4
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.f_res = 1.0 / self.t_rf
        self.circumference = 5990.0
        self.intensity = 2.7e12
        self.n_slices = 1024

        # Static profile spanning 1.5 RF periods, cut_left > 0 as the feedback
        # requires.
        self.noisy_profile = self._make_noisy_profile(self.t_rf, self.n_slices)

        self.stub_beam = StubBeam(self.intensity)

    @staticmethod
    def _make_noisy_profile(t_rf: float, n_slices: int) -> StaticProfile:
        """
        Build the noisy-Gaussian static profile used throughout this module.

        Parameters
        ----------
        t_rf
            RF period defining the profile window (1.5 to 4.5 pi in rad).
        n_slices
            Number of profile bins.

        Returns
        -------
        StaticProfile
            Noisy Gaussian bunch with zeroed leading/trailing bins.
        """
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, n_slices, t_rf
        )
        t = profile.hist_x
        t0 = 0.5 * (t[0] + t[-1])
        sigma = 0.08 * t_rf

        rng = np.random.default_rng(12345)
        hist_y = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        hist_y = hist_y + 0.05 * rng.standard_normal(n_slices)
        hist_y = np.clip(hist_y, 0.0, None)
        # The resonator solver warns / can go unstable with charge in the
        # leading or trailing edge bins, so force them to zero.
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0

        profile._hist_y = hist_y
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        return profile

    def _multi_turn_induced_voltage(self) -> np.ndarray:
        """
        Induced voltage from a single pass of the multi-turn solver.

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
            decay_fraction_threshold=1e-12, delta_f=0.0
        )
        wakefield = WakeField(
            sources=(resonator,),
            solver=solver,
            profile=self.noisy_profile,
            parent_rf_station=StubRFStation(self.omega_rf),
        )
        # Wire up the bits normally set in on_wakefield_init_simulation so the
        # solver can run without a full Simulation.
        solver._parent_wakefield = wakefield
        solver.circumference = self.circumference
        solver._maximum_storage_time = 1.0  # >> t_rf, keeps the single pass
        solver._last_reference_time = -np.finfo(float).eps

        return np.asarray(solver.calc_induced_voltage(self.stub_beam))

    def _non_driven_feedback_induced_voltage(self) -> np.ndarray:
        """
        Lab-frame beam-induced voltage from the non-driven feedback.

        Returns
        -------
        numpy.ndarray
            Lab-frame beam-induced voltage on the profile's fine grid.
        """
        feedback = IQCavityFeedbackTimingClass(
            profile=self.noisy_profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=0.0,  # non-driven: no generator current
            n_cavities=1,
            initial_voltage=0.0,  # antenna voltage starts at zero
            n_rf_periods_per_coarse_grid=1,
            delta_omega=0.0,
        )

        # Beam current on the fine grid, exactly as the feedback computes it
        # internally (rf_beam_current, then divide by the bin width).
        charges_fine = rf_beam_current(
            beam=self.stub_beam,
            profile=self.noisy_profile,
            omega_c=self.omega_rf,
            T_rev=self.t_rf,
            use_lowpass_filter=False,
            external_reference=True,
            dT=0.0,
        )
        feedback.beam_current_fine_grid = (
            charges_fine / self.noisy_profile.hist_step
        )
        feedback.generator_current_fine_grid = np.zeros_like(
            feedback.beam_current_fine_grid
        )

        # Drive the cavity response on the fine grid with zero initial
        # conditions and no generator current.
        feedback.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_voltage_gradient_fine_grid=0.0,
            initial_generator_current_fine_grid=0.0,
            samples_per_rf_fine_grid=(
                self.omega_rf * self.noisy_profile.hist_step
            ),
            relative_detuning=0.0,
        )

        v_ant = feedback.antenna_voltage_fine_grid
        # Project the I/Q envelope back to the lab-frame induced voltage.
        return lab_frame_voltage(
            v_ant, self.omega_rf, self.noisy_profile.hist_x
        )

    def _plot_induced_voltage(self, v_solver, v_feedback):
        """
        Save a debug plot of the induced voltage vs time along the bunch.

        Disabled by default. Enable by setting the module-level
        ``support.DEBUG_PLOTS`` constant to ``"save"`` (write a PNG) or
        ``"show"`` (also open an interactive window).

        Parameters
        ----------
        v_solver
            Induced voltage from the multi-turn resonator solver.
        v_feedback
            Induced voltage from the non-driven cavity feedback.
        """
        t_ns = self.noisy_profile.hist_x * 1e9
        fig, (ax_v, ax_diff) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.suptitle("Induced voltage along the bunch")
        ax_v.plot(t_ns, v_solver, color="C0", label="MultiPassResonatorSolver")
        ax_v.plot(
            t_ns, v_feedback, color="C1", ls="--", label="non-driven feedback"
        )
        ax_v.set_ylabel("induced voltage [V]")
        ax_v.legend(loc="best")
        ax_diff.plot(t_ns, v_feedback - v_solver, color="C3")
        ax_diff.set_ylabel("feedback - solver [V]")
        ax_diff.set_xlabel("time [ns]")
        fig.tight_layout()

        # plt.savefig("induced_voltage_over_time.png")
        plt.show()

    def test_induced_voltage_matches_non_driven_feedback(self):
        """The two models agree on the induced voltage to < 1 %."""
        v_solver = self._multi_turn_induced_voltage()
        v_feedback = self._non_driven_feedback_induced_voltage()

        # Debug plot (opt-in) before the assertions.
        if DEBUG_PLOT:
            self._plot_induced_voltage(v_solver, v_feedback)

        peak = np.max(np.abs(v_solver))
        self.assertGreater(peak, 0.0)

        # Pointwise: every sample within 1 % of the peak induced voltage.
        np.testing.assert_allclose(
            v_feedback, v_solver, atol=0.01 * peak, rtol=0.0
        )

        # Overall shape: relative L2 difference well below 1 %.
        self.assertLess(rel_err(v_feedback, v_solver), 0.01)

        # Peak amplitude agreement within 1 %.
        self.assertAlmostEqual(
            np.max(np.abs(v_feedback)) / peak, 1.0, delta=0.01
        )

    def test_zeroed_profile_edges_remain_zero(self):
        """Guard the precondition that the edge bins carry no charge."""
        self.assertEqual(self.noisy_profile.hist_y[0], 0.0)
        self.assertEqual(self.noisy_profile.hist_y[-1], 0.0)

    def test_feedback_without_beam_or_generator_is_silent(self):
        """A non-driven feedback with zero initial voltage induces nothing."""
        feedback = IQCavityFeedbackTimingClass(
            profile=self.noisy_profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=0.0,
            n_cavities=1,
            initial_voltage=0.0,
            n_rf_periods_per_coarse_grid=1,
            delta_omega=0.0,
        )
        feedback.beam_current_fine_grid = np.zeros(
            self.n_slices, dtype=complex
        )
        feedback.generator_current_fine_grid = np.zeros(
            self.n_slices, dtype=complex
        )
        feedback.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_voltage_gradient_fine_grid=0.0,
            initial_generator_current_fine_grid=0.0,
            samples_per_rf_fine_grid=(
                self.omega_rf * self.noisy_profile.hist_step
            ),
            relative_detuning=0.0,
        )
        np.testing.assert_array_equal(
            feedback.antenna_voltage_fine_grid,
            np.zeros(self.n_slices, dtype=complex),
        )

    # ----- full-simulation multi-turn comparison -------------------------
    # High Q_L so the previous-pass wake survives ~88 % per turn
    # (exp(-omega * t_rev / Q_L)); with the setUp value of 1.29e4 only
    # ~6e-5 would survive and the multi-turn aspect would be invisible.
    MULTITURN_R_OVER_Q = 518.0
    MULTITURN_Q_L = 1.29e6
    MULTITURN_N_TURNS = 3
    MULTITURN_V_DESIGN = 30e6
    MULTITURN_HARMONIC = 25900
    MULTITURN_ENERGY = 63e9
    MULTITURN_ALPHA_P = 10.395e-4
    MULTITURN_CIRCUMFERENCE = 5990.0
    MULTITURN_INTENSITY = 2.7e12
    MULTITURN_N_SLICES = 1024

    _multiturn_cache = None

    @classmethod
    def _run_multiturn_case(cls, mode: str, cycle, t_rf: float) -> list:
        """
        Run a full multi-turn Simulation and collect a voltage per turn.

        A dummy beam without macroparticles drives nothing physically; the
        static noisy profile (``profile.active = False`` so the empty beam
        never overwrites the histogram) is the only excitation. The beam's
        reference still advances by t_rev per turn, which is what propagates
        both the convolution's past-wake times and the feedback's coarse
        grid.

        Parameters
        ----------
        mode
            ``"mtw"`` (convolution wakefield), ``"fb"`` (feedback with beam
            current) or ``"fb_reference"`` (feedback with zero intensity, to
            isolate the beam-induced part by linearity).
        cycle
            Static magnetic cycle shared by all cases.
        t_rf
            RF period consistent with cycle, harmonic and circumference.

        Returns
        -------
        list
            Per-turn voltage arrays: the wakefield induced voltage for
            ``"mtw"``, the station gap voltage otherwise.
        """
        profile = cls._make_noisy_profile(t_rf, cls.MULTITURN_N_SLICES)
        profile.active = False  # keep the histogram static (no particles)

        ring = Ring(
            circumference=cls.MULTITURN_CIRCUMFERENCE,
            check_section_indices=False,
        )
        drift = DriftSimple(
            orbit_length=cls.MULTITURN_CIRCUMFERENCE,
            momentum_compaction_factor=cls.MULTITURN_ALPHA_P,
        )
        if mode == "mtw":
            element = WakeField(
                sources=(
                    Resonators(
                        cls.MULTITURN_R_OVER_Q * cls.MULTITURN_Q_L,
                        1.0 / t_rf,
                        cls.MULTITURN_Q_L,
                    ),
                ),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12
                ),
                profile=profile,
            )
            rf = SingleHarmonicRFStation(
                voltage=cls.MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=cls.MULTITURN_HARMONIC,
                local_wakefield=element,
                profile=profile,
            )
        else:
            # Operating-point cavity (V_init = V_design): a cold start
            # (V_init = 0) trips the coarse-grid beam-kick magnitude check,
            # whose heuristic assumes an established antenna voltage.
            element = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=cls.MULTITURN_R_OVER_Q,
                Q_L=cls.MULTITURN_Q_L,
                generator_current=0.0,
                n_cavities=1,
                initial_voltage=cls.MULTITURN_V_DESIGN,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
            )
            rf = SingleHarmonicRFStation(
                voltage=cls.MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=cls.MULTITURN_HARMONIC,
                cavity_feedback=element,
                profile=profile,
            )
        # Drift first so the feedback's RF station is not the first
        # reference-altering element.
        ring.add_elements([drift, rf], reorder=False)
        sim = Simulation(ring=ring, magnetic_cycle=cycle)

        beam = Beam(
            intensity=(
                0.0 if mode == "fb_reference" else cls.MULTITURN_INTENSITY
            ),
            particle_type=mu_plus,
        )
        beam.reference.total_energy = cls.MULTITURN_ENERGY
        beam.setup_beam(dt=np.array([]), dE=np.array([]))

        per_turn = []

        def collect(simulation, beam_in_callback):
            if mode == "mtw":
                per_turn.append(np.copy(np.asarray(element.induced_voltage)))
            else:
                per_turn.append(
                    np.copy(np.asarray(rf.calc_gap_voltage_with_feedbacks()))
                )

        sim.run_simulation(
            (beam,),
            n_turns=cls.MULTITURN_N_TURNS,
            callbacks=collect,
            show_progressbar=False,
        )
        return per_turn

    @classmethod
    def _multiturn_results(cls):
        """
        Run (once) and cache the three multi-turn cases.

        Returns
        -------
        v_convolution_turns
            Per-turn induced voltage of the multi-pass convolution.
        v_feedback_turns
            Per-turn beam-induced voltage of the feedback (gap voltage of
            the beam-driven run minus the no-beam reference run).
        """
        if cls._multiturn_cache is None:
            cycle = ConstantMagneticCycle(
                reference_particle=mu_plus,
                value=cls.MULTITURN_ENERGY,
                in_unit="total energy",
            )
            t_rev = cycle.get_t_rev_init(
                cls.MULTITURN_CIRCUMFERENCE, particle_type=mu_plus
            )
            t_rf = t_rev / cls.MULTITURN_HARMONIC

            v_convolution_turns = cls._run_multiturn_case("mtw", cycle, t_rf)
            gap_beam_turns = cls._run_multiturn_case("fb", cycle, t_rf)
            gap_reference_turns = cls._run_multiturn_case(
                "fb_reference", cycle, t_rf
            )
            v_feedback_turns = [
                gap_beam - gap_reference
                for gap_beam, gap_reference in zip(
                    gap_beam_turns, gap_reference_turns, strict=True
                )
            ]
            cls._multiturn_cache = (v_convolution_turns, v_feedback_turns)
        return cls._multiturn_cache

    def test_multiturn_wake_accumulates_over_turns(self):
        """
        The multi-pass wake genuinely builds up turn over turn.

        With ``MULTITURN_Q_L = 1.29e6`` the previous-pass wake survives
        ~88 % per turn and the in-phase buildup follows the geometric sum
        (measured peaks ~1.0, 1.9, 2.8). The first turn (no previous pass
        yet) must also agree with the feedback to the single-pass accuracy.
        """
        v_convolution_turns, v_feedback_turns = self._multiturn_results()

        # First turn == single pass: feedback and convolution agree.
        self.assertLess(
            rel_err(v_feedback_turns[0], v_convolution_turns[0]), 0.02
        )

        peaks = [np.max(np.abs(v)) for v in v_convolution_turns]
        self.assertGreater(peaks[1] / peaks[0], 1.5)
        self.assertGreater(peaks[2] / peaks[1], 1.2)
        # Bounded by the geometric series of the per-turn survival.
        self.assertLess(peaks[2] / peaks[0], 3.5)

    def test_multiturn_feedback_propagation_matches_convolution(self):
        """
        Feedback coarse-grid propagation vs convolution on every turn.

        Regression test for the dropped downsample remainder in
        :func:`blond.physics.feedbacks.helpers.rf_beam_current`: all
        demodulated charge after the last coarse-cell boundary used to be
        silently discarded (up to ~half the bunch, with a rotated phase
        centroid), which corrupted the coarse-grid beam loading and every
        carried-over turn by 34-48 % while leaving the fine grid -- and
        therefore all single-turn comparisons -- untouched. With the
        remainder included, all turns agree to < 0.3 %.
        """
        v_convolution_turns, v_feedback_turns = self._multiturn_results()

        if DEBUG_PLOT:
            self._plot_multiturn(v_convolution_turns, v_feedback_turns)

        for turn_i, (v_convolution, v_feedback) in enumerate(
            zip(v_convolution_turns, v_feedback_turns, strict=True)
        ):
            self.assertLess(
                rel_err(v_feedback, v_convolution),
                0.02,
                f"turn {turn_i}",
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
            len(v_convolution_turns), 1, sharex=True, figsize=(8, 9)
        )
        fig.suptitle("Multi-turn induced voltage: convolution vs feedback")
        for turn_i, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(
                v_convolution_turns[turn_i], color="C0", label="convolution"
            )
            ax.plot(
                v_feedback_turns[turn_i], color="C1", ls="--", label="feedback"
            )
            ax.set_ylabel(f"turn {turn_i} [V]")
        np.atleast_1d(axes)[0].legend(loc="best")
        np.atleast_1d(axes)[-1].set_xlabel("profile bin")
        fig.tight_layout()
        plt.show()

    def test_step_size_check_fires_on_run_simulation(self):
        """
        An unphysical detuning aborts the run-start initialisation.

        Companion to the unit-level step-size tests in
        ``test_mucol_cav_fdbk.py`` (which patch the carrier properties): here
        the check runs inside ``on_run_simulation`` with the carrier frequency
        resolved through a real RF station. Only ``delta_omega`` is relevant,
        so the beam and simulation are stubbed -- no beam preparation or
        tracking is needed.
        """
        feedback = IQCavityFeedbackTimingClass(
            profile=self.noisy_profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current=0.0,
            n_cavities=1,
            initial_voltage=0.0,
            n_rf_periods_per_coarse_grid=1,
            # detuning_phase_per_step = delta_omega * sampling_time_coarse
            # ~ 1e12 * 1e-9 ~ 1000, far beyond the hard limit of 2.0.
            delta_omega=1e12,
        )
        # Voltage/harmonic are placeholders; only omega_rf enters the check.
        rf = SingleHarmonicRFStation(
            voltage=30e6,
            phi_rf=0.0,
            harmonic=25900,
            cavity_feedback=feedback,
            profile=self.noisy_profile,
        )
        # Normally set by the station's own initialisation at run start.
        rf.omega_rf_design = self.omega_rf

        # on_run_simulation only needs the ring's reference-altering elements
        # (to locate the parent station) and a deepcopy-able beam reference.
        stub_simulation = Mock()
        stub_simulation.ring.elements.get_elements.return_value = (rf,)

        with self.assertRaises(ValueError) as cm:
            feedback.on_run_simulation(
                simulation=stub_simulation,
                beam=StubBeam(self.intensity),
                n_turns=1,
            )
        self.assertIn("detuning_phase_per_step", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
