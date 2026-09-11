"""Unit tests for the muon collider cavity feedback timing class."""

import unittest
import warnings
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    Resonators,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.beam_current import rf_beam_current
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedbackTimingClass,
)
from blond.physics.feedbacks.cavity_solvers import (
    coarse_step_exponent,
    exponential_drive_weight,
    exponential_voltage_multiplier,
)
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)

# Package-relative imports: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .stubs import StubBeam
from .support import lab_frame_voltage

DEBUG_PLOT = False


class TestCavityFeedback(unittest.TestCase):
    """The IQCavityFeedbackTimingClass coarse step on RCS1 parameters."""

    def setUp(self):
        """Build a cavity feedback instance with RCS1 4-station parameters."""
        # RCS1 4 stations
        self.prof = Mock(StaticProfile)
        self.prof.hist_x = np.linspace(
            5.791514370530446e-10, 1.7351942079901463e-09, num=1024
        )
        self.prof.hist_y = np.zeros(1024)
        self.prof.cut_left = self.prof.hist_x[0]

        self.R_over_Q = 518
        self.Q_L = 1287601.7251526634
        self.n_cavities = 42.217908605563096
        self.generator_current = 0.0233441090290177 + 0.04958176818202371j
        self.initial_voltage = 30e6
        self.n_rf_periods_per_coarse_grid = 1  # TODO: check for 2 and 0.5
        self.delta_omega = -6717.47508329349

        self.cav_fdbk = IQCavityFeedbackTimingClass(
            profile=self.prof,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=self.generator_current,
            n_cavities=self.n_cavities,
            initial_voltage=self.initial_voltage,
            n_rf_periods_per_coarse_grid=self.n_rf_periods_per_coarse_grid,
            delta_omega=self.delta_omega,
        )

    def test_circuit_track_applies_delta_omega_phase_shift(self):
        """
        Check circuit_track() applies the delta_omega phase shift.

        circuit_track() feeds `relative_detuning = delta_omega / omega_input`
        into the coarse step, which advances the antenna voltage each
        coarse-grid step by the exact propagator `exp(L)` with
        `L = -0.5 * omega * dt / Q_L + 1j * delta_omega * dt`.

        This test drives circuit_track() with a hand-built, constant-step
        rf_centers grid and zero generator/beam current (no_beam=True), so
        the antenna voltage on the coarse grid should evolve purely by that
        per-step complex multiplier. We recompute the expected trajectory
        from the multiplier directly and compare it element-wise to what
        circuit_track() actually produced.
        """
        omega_input = 2 * np.pi * 1e9
        n_steps = 50
        dt = 1.2e-9

        # Build a constant-step rf_centers grid covering a single segment.
        self.cav_fdbk._rf_centers = np.arange(1, n_steps + 1) * dt
        self.cav_fdbk._rf_centers_lengths = np.array([n_steps])
        self.cav_fdbk._residual_time_last_rf_centers_calculation = 0.0
        self.cav_fdbk._last_rf_centers_entry = None

        # Zero out generator/beam current contributions so only the
        # `exp(-0.5*omega*dt/Q_L + 1j*relative_detuning*omega*dt)` term
        # governs the antenna voltage evolution.
        self.cav_fdbk._generator_current_bias = 0.0 + 0.0j
        self.cav_fdbk.generator_current_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )
        self.cav_fdbk._last_val_generator_current = 0.0 + 0.0j
        self.cav_fdbk._last_val_beam_current = 0.0 + 0.0j

        v0 = self.initial_voltage + 0.0j
        # The carried field is generator-established (design-anchored),
        # so it seeds the generator-sourced component.
        self.cav_fdbk._last_val_ant_voltage = v0
        self.cav_fdbk._last_val_ant_voltage_gen = v0
        self.cav_fdbk._last_val_ant_voltage_beam = 0.0 + 0.0j
        self.cav_fdbk.antenna_voltage_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )
        self.cav_fdbk.antenna_voltage_gen_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )
        self.cav_fdbk.antenna_voltage_beam_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )

        self.cav_fdbk.circuit_track(
            omega_input=omega_input,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        v = self.cav_fdbk.antenna_voltage_coarse_grid

        omega_times_dt = omega_input * dt
        step_exponent = (
            -0.5 * omega_times_dt / self.Q_L + 1j * self.delta_omega * dt
        )
        expected = v0 * np.exp(step_exponent * np.arange(1, n_steps + 1))

        np.testing.assert_allclose(v, expected, rtol=1e-12)


class TestFineGridResonatorBenchmark(unittest.TestCase):
    """
    Benchmark FB against resonator induced voltage single turn.

    Benchmark the single-turn (fine-grid) cavity beam-loading response of
    IQCavityFeedbackTimingClass against an independent resonator induced
    voltage model, on a real Gaussian-plus-noise beam profile.

    The fine-grid antenna voltage (generator current zeroed) is the purely
    beam-induced voltage. Demodulated at omega_rf, it is remodulated to the
    lab frame and compared to the induced voltage of a matching Resonators
    source (R_s = R_over_Q * Q_L, Q = Q_L, f_r = f_rf + delta_omega/2pi)
    convolved with the same profile.
    """

    R_over_Q = 518.0
    Q_L = 1287601.7251526634
    f_rf = 1.3e9
    intensity = 2.7e12
    n_macroparticles = int(1e6)
    n_bins = 2**12

    def _build_beam_and_profile(self, seed=0):
        rng = np.random.default_rng(seed)
        t_rf = 1.0 / self.f_rf
        profile = StaticProfile.from_rad(
            0.5 * np.pi, 3.5 * np.pi, self.n_bins, t_rf
        )
        t_center = t_rf
        sigma_t = 0.06 * t_rf
        n_noise = self.n_macroparticles // 10
        n_gauss = self.n_macroparticles - n_noise
        dt = np.concatenate(
            [
                rng.normal(t_center, sigma_t, n_gauss),
                rng.uniform(
                    t_center - 4 * sigma_t, t_center + 4 * sigma_t, n_noise
                ),
            ]
        )
        beam = Beam(
            intensity=self.intensity,
            particle_type=mu_plus,
            is_counter_rotating=False,
        )
        beam.setup_beam(
            dt=dt, dE=np.zeros_like(dt), mpi_mode="root-distributes"
        )
        profile.track(beam=beam)
        return beam, profile

    def _cavity_lab_voltage(self, beam, profile, delta_omega):
        omega_rf = 2.0 * np.pi * self.f_rf
        charges_fine = rf_beam_current(
            beam=beam,
            profile=profile,
            omega_c=omega_rf,
            use_lowpass_filter=False,
        )
        cav = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=0.0 + 0.0j,
            n_cavities=1,
            initial_voltage=0.0,
            delta_omega=delta_omega,
        )
        cav.beam_current_fine_grid = charges_fine / profile.hist_step
        cav.generator_current_fine_grid = np.zeros(self.n_bins, dtype=complex)
        cav.cavity_response_fine(
            initial_voltage_fine_grid=0.0,
            initial_generator_current_fine_grid=0.0,
            omega_times_dt_fine_grid=omega_rf * profile.hist_step,
            relative_detuning=delta_omega / omega_rf,
        )
        return lab_frame_voltage(
            cav.antenna_voltage_fine_grid,
            omega_rf,
            profile.hist_x,
        )

    def _resonator_induced_voltage(self, beam, profile, delta_omega):
        res = Resonators(
            shunt_impedances=self.R_over_Q * self.Q_L,
            quality_factors=self.Q_L,
            center_frequencies=self.f_rf + delta_omega / (2.0 * np.pi),
        )
        wf = WakeField(
            sources=(res,),
            solver=SingleTurnResonatorConvolutionSolver(),
            profile=profile,
        )
        wf.solver.on_wakefield_init_simulation(Mock(), wf)
        return copy_to_cpu(wf.solver.calc_induced_voltage(beam=beam))

    def _assert_matches(self, delta_omega):
        beam, profile = self._build_beam_and_profile()
        v_cav = self._cavity_lab_voltage(beam, profile, delta_omega)
        v_res = self._resonator_induced_voltage(beam, profile, delta_omega)

        # Best-fit amplitude scale (should be ~1 with the sign convention
        # already folded into v_cav).
        scale = np.dot(v_res, v_cav) / np.dot(v_cav, v_cav)
        nrmse = (
            np.sqrt(np.mean((v_res - scale * v_cav) ** 2))
            / np.abs(v_res).max()
        )
        corr = np.corrcoef(v_cav, v_res)[0, 1]

        self.assertGreater(
            corr, 0.999, f"shape mismatch (corr={corr}) for {delta_omega=}"
        )
        self.assertAlmostEqual(
            scale,
            1.0,
            delta=0.05,
            msg=f"amplitude scale off ({scale}) for {delta_omega=}",
        )
        self.assertLess(
            nrmse,
            1e-2,
            f"waveform mismatch (nrmse={nrmse}) for {delta_omega=}",
        )

    def test_fine_grid_matches_resonator_on_resonance(self):
        """On resonance, fine-grid response matches the resonator model."""
        self._assert_matches(delta_omega=0.0)

    def test_fine_grid_matches_resonator_positive_detuning(self):
        """With positive detuning, the phase shift matches a detuned resonator."""
        self._assert_matches(delta_omega=5e6)

    def test_fine_grid_matches_resonator_negative_detuning(self):
        """With negative detuning, the phase shift matches a detuned resonator."""
        self._assert_matches(delta_omega=-2e7)


class TestCavityPrefill(unittest.TestCase):
    """
    Feedforward cavity pre-fill / injection matching.

    The no-beam, constant-current cavity fills from cold as
    ``V(t) = V_ss * (1 - exp(lambda t))`` with
    ``lambda = -omega/(2 Q_L) + 1j delta_omega`` and
    ``V_ss = -(R/Q) omega I_gen / lambda``. ``pretrack_fill_voltage`` returns
    the complex seed antenna voltage; with ``injection_voltage`` it returns the
    fill-transient value when ``|V|`` first reaches that target.
    """

    R_over_Q = 518.0
    Q_L = 1287601.7251526634
    f_rf = 1.3e9
    omega_rf = 2.0 * np.pi * f_rf
    V0 = 30.0e6
    # On resonance V_ss = 2 (R/Q) Q_L I_g, so this bias fills the cavity to V0.
    I_g = V0 / (2.0 * R_over_Q * Q_L)
    t_rev = 25900 / f_rf  # harmonic / f_rf

    def _plot_fill_evolution(
        self,
        n_pretrack,
        seed,
        delta_omega=0.0,
        generator_current=None,
        injection_voltage=None,
    ):
        """
        Save a debug plot of the pre-track cavity fill ``V(t)`` vs turns.

        Disabled by default. Enable by setting the module-level ``DEBUG_PLOT``
        flag to ``True`` to open an interactive window.

        Reconstructs the closed-form fill envelope
        ``V(t) = V_ss (1 - exp(lambda t))`` (the same expression
        ``pretrack_fill_voltage`` integrates) over ``[0, n_pretrack T_0]``. The
        antenna voltage is complex, so the top panel shows ``Re V`` and
        ``Im V`` (with the complex seed marked on each) and the bottom panel
        shows ``|V|`` with the steady-state fill ``|V_ss|``; with
        ``injection_voltage`` the injection crossing is marked too. On
        resonance with a real generator current the fill stays on the real
        axis (``Im V ~ 0``) -- the imaginary content only appears once the
        cavity is detuned or the (complex) beam loading enters after injection.

        Parameters
        ----------
        n_pretrack
            Cavity fill budget in turns (the plotted window).
        seed
            Complex seed antenna voltage returned by ``pretrack_fill_voltage``.
        delta_omega
            Cavity resonance detuning [rad/s] used for the fill.
        generator_current
            Constant generator current [A]; defaults to the on-resonance
            ``self.I_g`` that fills to ``V0``.
        injection_voltage
            If given, the injection target magnitude [V] to mark.
        """
        if generator_current is None:
            generator_current = self.I_g + 0.0j
        lam = -self.omega_rf / (2.0 * self.Q_L) + 1j * delta_omega
        v_ss = -(self.R_over_Q * self.omega_rf) * generator_current / lam
        t = np.linspace(0.0, n_pretrack * self.t_rev, 2000)
        turns = t / self.t_rev
        voltage = v_ss * (1.0 - np.exp(lam * t))

        # The seed sits at the end of the window (no injection) or on the rise
        # at the injection crossing; place its marker at the matching |V| turn.
        seed_turn = turns[
            int(np.argmin(np.abs(np.abs(voltage) - np.abs(seed))))
        ]

        fig, (ax_ri, ax_mag) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        fig.suptitle("Cavity pre-track fill")

        ax_ri.plot(
            turns, voltage.real / 1e6, color="C0", label=r"$\mathrm{Re}\,V$"
        )
        ax_ri.plot(
            turns, voltage.imag / 1e6, color="C2", label=r"$\mathrm{Im}\,V$"
        )
        ax_ri.plot(seed_turn, seed.real / 1e6, "o", color="C0")
        ax_ri.plot(seed_turn, seed.imag / 1e6, "o", color="C2")
        ax_ri.set_ylabel("antenna voltage [MV]")
        ax_ri.legend(loc="best")
        ax_ri.grid(True, alpha=0.3)

        ax_mag.plot(
            turns, np.abs(voltage) / 1e6, color="C3", label=r"$|V(t)|$"
        )
        ax_mag.axhline(
            np.abs(v_ss) / 1e6,
            color="k",
            ls=":",
            lw=0.8,
            label=r"$|V_\mathrm{ss}|$",
        )
        ax_mag.plot(
            seed_turn, np.abs(seed) / 1e6, "o", color="C3", label="seed"
        )
        if injection_voltage is not None:
            ax_mag.axhline(
                injection_voltage / 1e6,
                color="C1",
                ls="--",
                lw=0.8,
                label="injection target",
            )
        ax_mag.set_xlabel("pre-track turn")
        ax_mag.set_ylabel("antenna voltage [MV]")
        ax_mag.legend(loc="best")
        ax_mag.grid(True, alpha=0.3)
        fig.tight_layout()

        # Debug-save next to this test file (not the repo root); needs ``import os``:
        # fig.savefig(
        #     os.path.join(os.path.dirname(__file__), "cavity_prefill.png"), dpi=200
        # )
        plt.show()

    def test_steady_state_fill_on_resonance_matches_two_r_q_ql_ig(self):
        """No-injection fill converges to V_ss = 2 (R/Q) Q_L I_g on resonance."""
        from blond.physics.feedbacks.cavity_solvers import (
            pretrack_fill_voltage,
        )

        seed = pretrack_fill_voltage(
            r_over_q=self.R_over_Q,
            q_l=self.Q_L,
            omega=self.omega_rf,
            delta_omega=0.0,
            generator_current=self.I_g + 0.0j,
            n_pretrack=500,  # well past the ~16-turn fill time constant
            t_rev=self.t_rev,
        )
        self.assertAlmostEqual(seed.imag, 0.0, delta=1.0)
        self.assertAlmostEqual(seed.real / self.V0, 1.0, places=6)

    def test_injection_voltage_seeds_at_the_requested_magnitude(self):
        """With injection_voltage, the seed magnitude is that target."""
        from blond.physics.feedbacks.cavity_solvers import (
            pretrack_fill_voltage,
        )

        injection_voltage = 20.0e6  # below the V0 = 30 MV steady-state fill
        seed = pretrack_fill_voltage(
            r_over_q=self.R_over_Q,
            q_l=self.Q_L,
            omega=self.omega_rf,
            delta_omega=0.0,
            generator_current=self.I_g + 0.0j,
            n_pretrack=500,
            t_rev=self.t_rev,
            injection_voltage=injection_voltage,
        )
        if DEBUG_PLOT:
            self._plot_fill_evolution(
                n_pretrack=500,
                seed=seed,
                injection_voltage=injection_voltage,
            )
        self.assertAlmostEqual(np.abs(seed) / injection_voltage, 1.0, places=4)

    def test_unreachable_injection_voltage_raises(self):
        """A target above the steady-state fill cannot be reached, so it raises."""
        from blond.physics.feedbacks.cavity_solvers import (
            pretrack_fill_voltage,
        )

        with self.assertRaises(ValueError) as cm:
            pretrack_fill_voltage(
                r_over_q=self.R_over_Q,
                q_l=self.Q_L,
                omega=self.omega_rf,
                delta_omega=0.0,
                generator_current=self.I_g + 0.0j,
                n_pretrack=500,
                t_rev=self.t_rev,
                injection_voltage=40.0e6,  # above the 30 MV steady-state fill
            )
        self.assertIn("injection_voltage", str(cm.exception))

    @staticmethod
    def _run_on_run_simulation(feedback, rf, t_rf, omega_rf):
        """
        Drive ``on_run_simulation`` far enough to apply the pre-fill.

        Only ``omega_rf`` and the parent station's reference bookkeeping are
        needed, so the beam and simulation are stubbed.

        Parameters
        ----------
        feedback
            Feedback under test.
        rf
            Parent RF station.
        t_rf
            RF period.
        omega_rf
            Design RF angular frequency to install on the station.
        """
        rf.omega_rf_design = omega_rf
        stub_simulation = Mock()
        stub_simulation.ring.elements.get_elements.return_value = (rf,)
        feedback.on_run_simulation(
            simulation=stub_simulation, beam=StubBeam(2.7e12), n_turns=1
        )

    def _build_feedback(self, t_rf, **kwargs):
        """
        Build a feedback + parent station on a real profile for run-start.

        Parameters
        ----------
        t_rf
            RF period used to size the profile.
        **kwargs
            Overrides forwarded to ``IQCavityFeedbackTimingClass``.

        Returns
        -------
        feedback, rf
            The feedback instance and its parent RF station.
        """
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 1024, t_rf)
        params = {
            "profile": profile,
            "R_over_Q": self.R_over_Q,
            "Q_L": 1.29e4,  # short fill so a few pre-fill turns suffice
            "generator_current_bias": 0.02 + 0.0j,
            "n_cavities": 1,
            "initial_voltage": 0.0,
            "n_rf_periods_per_coarse_grid": 1,
            "delta_omega": 0.0,
        }
        params.update(kwargs)
        feedback = IQCavityFeedbackTimingClass(**params)
        rf = SingleHarmonicRFStation(
            voltage=30e6,
            phi_rf=0.0,
            harmonic=25900,
            cavity_feedback=feedback,
            profile=profile,
        )
        return feedback, rf

    def test_n_pretrack_seeds_init_voltage_with_the_fill(self):
        """On_run_simulation replaces init_voltage with the pre-fill seed."""
        from blond.physics.feedbacks.cavity_solvers import (
            pretrack_fill_voltage,
        )

        t_rf = 1.0e-9
        omega_rf = 2.0 * np.pi / t_rf
        feedback, rf = self._build_feedback(t_rf, n_pretrack=50)
        self._run_on_run_simulation(feedback, rf, t_rf, omega_rf)

        expected = pretrack_fill_voltage(
            r_over_q=feedback.R_over_Q,
            q_l=feedback.Q_L,
            omega=feedback.omega_rf,
            delta_omega=feedback.delta_omega,
            generator_current=feedback._generator_current_bias,
            n_pretrack=50,
            t_rev=feedback.t_rev,
        )
        self.assertAlmostEqual(feedback._init_voltage, expected, places=6)
        self.assertGreater(abs(feedback._init_voltage), 0.0)

    def test_fill_seed_is_an_equilibrium_of_the_coarse_step(self):
        """A no-beam cavity started at the fill seed does not drift."""
        n_steps = 30
        dt = 1.0 / self.f_rf
        cav = IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=self.I_g + 0.0j,
            n_cavities=1,
            delta_omega=0.0,
        )
        # On resonance the fill seed is V_ss = 2 (R/Q) Q_L I_g = V0.
        v_ss = self.V0 + 0.0j
        cav._rf_centers = np.arange(1, n_steps + 1) * dt
        cav._rf_centers_lengths = np.array([n_steps])
        cav._residual_time_last_rf_centers_calculation = 0.0
        cav._last_rf_centers_entry = None
        cav.generator_current_coarse_grid = np.full(
            n_steps, self.I_g, dtype=complex
        )
        cav._last_val_generator_current = self.I_g + 0.0j
        cav._last_val_beam_current = 0.0 + 0.0j
        # The fill is generator-established: seed the gen component.
        cav._last_val_ant_voltage = v_ss
        cav._last_val_ant_voltage_gen = v_ss
        cav._last_val_ant_voltage_beam = 0.0 + 0.0j
        cav.antenna_voltage_coarse_grid = np.zeros(n_steps, dtype=complex)
        cav.antenna_voltage_gen_coarse_grid = np.zeros(n_steps, dtype=complex)
        cav.antenna_voltage_beam_coarse_grid = np.zeros(n_steps, dtype=complex)

        cav.circuit_track(
            omega_input=self.omega_rf,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        np.testing.assert_allclose(
            cav.antenna_voltage_coarse_grid,
            v_ss * np.ones(n_steps),
            rtol=1e-9,
        )

    def test_detuned_fill_seed_is_an_equilibrium_of_the_coarse_step(self):
        """The seed carries the detuning phase, so a detuned fill is flat.

        At ``delta_omega = 0`` the no-beam fixed point
        ``V* = -(R/Q) omega I_gen / lambda`` is real and positive, so a
        seed in the wrong frame is indistinguishable from one in the right
        frame -- the sibling test above cannot see a phase convention at
        all.  With a detuning the fixed point acquires the phase
        ``arg(-1 / lambda)``, and the seed has to carry it: if the fill and
        the coarse recursion disagreed about which frame the seed is in,
        turn 0 would open with an injection transient that relaxes over
        ``2 Q_L / omega``.

        The cavity is started exactly at the analytic fixed point, so every
        cell must reproduce it -- magnitude AND phase.
        """
        n_steps = 30
        dt = 1.0 / self.f_rf
        # Half a bandwidth of detuning: enough to rotate the fixed point
        # well away from the real axis (arg = -26.6 deg here).
        delta_omega = 0.5 * self.omega_rf / (2.0 * self.Q_L)
        cav = IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=self.I_g + 0.0j,
            n_cavities=1,
            delta_omega=delta_omega,
        )
        lam = -self.omega_rf / (2.0 * self.Q_L) + 1j * delta_omega
        v_ss = -(self.R_over_Q * self.omega_rf) * (self.I_g + 0.0j) / lam
        # Guard the premise: the fixed point really is off the real axis.
        self.assertGreater(abs(np.angle(v_ss)), 0.4)

        cav._rf_centers = np.arange(1, n_steps + 1) * dt
        cav._rf_centers_lengths = np.array([n_steps])
        cav._residual_time_last_rf_centers_calculation = 0.0
        cav._last_rf_centers_entry = None
        cav.generator_current_coarse_grid = np.full(
            n_steps, self.I_g, dtype=complex
        )
        cav._last_val_generator_current = self.I_g + 0.0j
        cav._last_val_beam_current = 0.0 + 0.0j
        cav._last_val_ant_voltage = v_ss
        cav._last_val_ant_voltage_gen = v_ss
        cav._last_val_ant_voltage_beam = 0.0 + 0.0j
        cav.antenna_voltage_coarse_grid = np.zeros(n_steps, dtype=complex)
        cav.antenna_voltage_gen_coarse_grid = np.zeros(n_steps, dtype=complex)
        cav.antenna_voltage_beam_coarse_grid = np.zeros(n_steps, dtype=complex)

        cav.circuit_track(
            omega_input=self.omega_rf,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        np.testing.assert_allclose(
            cav.antenna_voltage_coarse_grid,
            v_ss * np.ones(n_steps),
            rtol=1e-9,
        )

    def test_injection_voltage_without_n_pretrack_raises(self):
        """Injection_voltage is meaningless without a pre-fill budget."""
        profile = StaticProfile.from_rad(np.pi * 1.5, np.pi * 4.5, 1024, 1e-9)
        with self.assertRaises(ValueError) as cm:
            IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=self.R_over_Q,
                Q_L=1.29e4,
                generator_current_bias=0.02 + 0.0j,
                n_cavities=1,
                injection_voltage=20.0e6,  # no n_pretrack
            )
        self.assertIn("n_pretrack", str(cm.exception))

    def test_fill_seed_uses_the_design_clock_under_an_rf_offset(self):
        """The pre-fill seed is the DESIGN-clock coarse fixed point.

        The coarse recursion drives every step at the design RF
        frequency (``calc_omega_rf_design``), so its no-beam fixed
        point is ``V* = -(R/Q) omega_design I_gen / lambda(omega_design)``.
        Evaluating the fill at the *actual* RF frequency (design plus
        ``delta_omega_rf``) misses that fixed point by
        ``O(delta_omega_rf / omega)``, so a cavity started at the seed
        drifts instead of sitting still -- exactly the spurious
        injection transient the pre-fill exists to avoid.

        A cavity detuning (``delta_omega != 0``) is what makes the seed
        frequency-dependent at all: on resonance
        ``V_ss = 2 (R/Q) Q_L I_gen`` carries no ``omega``.
        """
        t_rf = 1.0e-9
        omega_design = 2.0 * np.pi / t_rf
        # One permille RF-frequency offset, and a cavity detuning of the
        # order of the cavity half-bandwidth (~50 deg loading angle).
        delta_omega_rf = 1.0e-3 * omega_design
        feedback, rf = self._build_feedback(
            t_rf, n_pretrack=5, delta_omega=3.0e5
        )
        with warnings.catch_warnings():
            # Programming the offset after construction warns by design
            # (single station); here it is just run-start configuration.
            warnings.simplefilter("ignore", UserWarning)
            rf.delta_omega_rf = delta_omega_rf
        self._run_on_run_simulation(feedback, rf, t_rf, omega_design)
        # Guard the premise: the two clocks really do disagree here.
        self.assertNotAlmostEqual(
            feedback.omega_rf / feedback.omega_rf_design, 1.0, places=6
        )

        n_steps = 300
        feedback._rf_centers = np.arange(1, n_steps + 1) * t_rf
        feedback._rf_centers_lengths = np.array([n_steps])
        feedback._residual_time_last_rf_centers_calculation = 0.0
        feedback._last_rf_centers_entry = None
        feedback.generator_current_coarse_grid = np.full(
            n_steps, feedback._generator_current_bias, dtype=complex
        )
        feedback._last_val_generator_current = feedback._generator_current_bias
        feedback._last_val_beam_current = 0.0 + 0.0j
        # The pre-fill seed is generator-established: gen component.
        feedback._last_val_ant_voltage = feedback._init_voltage
        feedback._last_val_ant_voltage_gen = feedback._init_voltage
        feedback._last_val_ant_voltage_beam = 0.0 + 0.0j
        feedback.antenna_voltage_coarse_grid = np.zeros(n_steps, dtype=complex)
        feedback.antenna_voltage_gen_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )
        feedback.antenna_voltage_beam_coarse_grid = np.zeros(
            n_steps, dtype=complex
        )

        # The forward span is always tracked at the design frequency.
        feedback.circuit_track(
            omega_input=feedback.omega_rf_design,
            no_beam=True,
            start_index=0,
            end_index=n_steps,
        )

        np.testing.assert_allclose(
            feedback.antenna_voltage_coarse_grid,
            feedback._init_voltage * np.ones(n_steps),
            rtol=1e-9,
        )


class TestExactCoarsePropagator(unittest.TestCase):
    """
    The coarse-grid step is the exact exponential propagator.

    Over one coarse step of length ``dt`` the per-cavity envelope obeys
    ``dV/dt = lambda V + s`` with ``lambda = -omega / (2 Q_L) + 1j
    delta_omega`` and the source ``s = (R/Q) omega (I_gen - I_beam / 2)``
    held constant, so ``V_next = e^L V + s dt (e^L - 1) / L`` with
    ``L = lambda dt``. A default-constructed feedback must advance the
    coarse grid by exactly that, on the per-cell reference path and on the
    compiled kernel alike. The expected values are spelled out by hand here
    (``np.exp``, no shared helper), and the steps are made large enough that
    any first-order truncation of ``e^L`` is off by percent, not by ULPs.
    """

    R_over_Q = 518.0

    def test_default_reference_step_is_the_closed_form(self):
        """One ``_advance_coarse_voltage`` step equals the exact solution."""
        Q_L = 20.0
        t_rf = 1.0 / 1.3e9
        omega = 2.0 * np.pi / t_rf
        delta_omega = 0.3 / t_rf
        feedback = IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=self.R_over_Q,
            Q_L=Q_L,
            generator_current_bias=0.0,
            n_cavities=1,
        )
        v_prev, i_gen, i_beam = 30e6 + 1e6j, 0.02 + 0.01j, 0.005j

        got = feedback._advance_coarse_voltage(
            v_prev, i_gen, i_beam, omega * t_rf, delta_omega / omega
        )

        step_exponent = (-omega / (2.0 * Q_L) + 1j * delta_omega) * t_rf
        source = self.R_over_Q * omega * (i_gen - 0.5 * i_beam)
        expected = (
            np.exp(step_exponent) * v_prev
            + source * t_rf * (np.exp(step_exponent) - 1.0) / step_exponent
        )
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)

    def test_default_kernel_and_reference_paths_advance_the_closed_form(
        self,
    ):
        """Several cells on both coarse paths follow the exact recursion.

        The generator component is driven by the carried current on cell 0
        and by the pre-filled bias grid afterwards; the beam component by
        the carried beam current on cell 0 and by the forward beam grid
        afterwards -- the zero-order hold the coarse recursion documents.
        """
        Q_L = 50.0
        t_rf = 1.0e-9
        omega = 2.0 * np.pi / t_rf
        delta_omega = 0.2 / t_rf
        n_cells = 5
        bias = 0.02 + 0.01j
        carried_generator_current = 0.03 - 0.005j
        carried_beam_current = 0.0 + 0.0j
        beam_current = np.array(
            [0.0, 1e-3 + 2e-4j, -5e-4j, 2e-3 + 0.0j, 1e-3j], dtype=complex
        )
        voltage_gen_init = 1.0e6 + 2.0e5j
        voltage_beam_init = -3.0e3 + 1.0e3j

        step_exponent = (-omega / (2.0 * Q_L) + 1j * delta_omega) * t_rf
        propagator = np.exp(step_exponent)
        drive_weight = (np.exp(step_exponent) - 1.0) / step_exponent
        expected_gen = np.empty(n_cells, dtype=complex)
        expected_beam = np.empty(n_cells, dtype=complex)
        voltage_gen, voltage_beam = voltage_gen_init, voltage_beam_init
        for cell in range(n_cells):
            generator_current = (
                carried_generator_current if cell == 0 else bias
            )
            cell_beam_current = (
                carried_beam_current if cell == 0 else beam_current[cell]
            )
            voltage_gen = propagator * voltage_gen + (
                self.R_over_Q * omega * t_rf * generator_current * drive_weight
            )
            voltage_beam = propagator * voltage_beam + (
                self.R_over_Q
                * omega
                * t_rf
                * (-0.5 * cell_beam_current)
                * drive_weight
            )
            expected_gen[cell] = voltage_gen
            expected_beam[cell] = voltage_beam

        for use_kernel in (True, False):
            with self.subTest(use_numba_envelope_kernel=use_kernel):
                feedback = IQCavityFeedbackTimingClass(
                    profile=Mock(StaticProfile),
                    R_over_Q=self.R_over_Q,
                    Q_L=Q_L,
                    generator_current_bias=bias,
                    n_cavities=1,
                    delta_omega=delta_omega,
                )
                feedback.use_numba_envelope_kernel = use_kernel
                feedback._rf_centers = np.arange(1, n_cells + 1) * t_rf
                feedback._rf_centers_lengths = np.array([n_cells])
                feedback._residual_time_last_rf_centers_calculation = 0.0
                feedback._last_rf_centers_entry = None
                feedback.antenna_voltage_coarse_grid = np.zeros(
                    n_cells, dtype=complex
                )
                feedback.antenna_voltage_gen_coarse_grid = np.zeros(
                    n_cells, dtype=complex
                )
                feedback.antenna_voltage_beam_coarse_grid = np.zeros(
                    n_cells, dtype=complex
                )
                feedback.generator_current_coarse_grid = np.full(
                    n_cells, bias, dtype=complex
                )
                feedback._last_val_ant_voltage_gen = voltage_gen_init
                feedback._last_val_ant_voltage_beam = voltage_beam_init
                feedback._last_val_ant_voltage = (
                    voltage_gen_init + voltage_beam_init
                )
                feedback._last_val_generator_current = (
                    carried_generator_current
                )
                feedback._last_val_beam_current = carried_beam_current
                feedback.beam_current_forward_coarse_grid = beam_current

                feedback._circuit_track_cells(
                    omega_input=omega,
                    no_beam=False,
                    start_index=0,
                    end_index=n_cells,
                )

                np.testing.assert_allclose(
                    feedback.antenna_voltage_gen_coarse_grid,
                    expected_gen,
                    rtol=1e-12,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    feedback.antenna_voltage_beam_coarse_grid,
                    expected_beam,
                    rtol=1e-12,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    feedback.antenna_voltage_coarse_grid,
                    expected_gen + expected_beam,
                    rtol=1e-12,
                    atol=0.0,
                )

    def test_pure_detuning_preserves_magnitude(self):
        """
        Preserve ``|V|`` under pure detuning (an exact rotation).

        The retired forward-Euler step grew the magnitude by
        ``sqrt(1 + (delta_omega dt)^2)`` per step here (``~1.118`` at
        ``delta_omega dt = 0.5``); the exact step only rotates.
        """
        # No decay (Q_L huge), no drive, a large per-step detuning rotation.
        feedback = IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=self.R_over_Q,
            Q_L=1e18,
            generator_current_bias=0.0,
            n_cavities=1,
        )
        v_prev = 30e6 + 0.0j
        # delta_omega * dt = 0.5
        omega_times_dt, rel_det = 2.0 * np.pi, 0.5 / (2.0 * np.pi)
        v_next = feedback._advance_coarse_voltage(
            v_prev, 0.0, 0.0, omega_times_dt, rel_det
        )
        self.assertAlmostEqual(np.abs(v_next) / np.abs(v_prev), 1.0, places=12)
        self.assertAlmostEqual(np.angle(v_next / v_prev), 0.5, places=12)

    def test_solver_switch_is_rejected(self):
        """The retired coarse-solver switch is no longer accepted."""
        for value in (True, False):
            with self.subTest(value=value), self.assertRaises(TypeError):
                IQCavityFeedbackTimingClass(
                    profile=Mock(StaticProfile),
                    R_over_Q=self.R_over_Q,
                    Q_L=1.29e6,
                    generator_current_bias=0.0,
                    n_cavities=1,
                    exponential_coarse_solver_enable=value,
                )


class TestSharedCoarseStepArithmetic(unittest.TestCase):
    """
    The per-cell and vectorised coarse steps share one spelling.

    The coarse recursion exists twice -- ``_advance_coarse_voltage`` (per cell,
    the reference) and ``_kernel_step_multipliers`` (vectorised, feeding the
    numba kernel). Both must be built from the same module-level arithmetic in
    ``blond.physics.feedbacks.cavity_solvers``; two independent spellings are
    exactly how the two paths drifted apart before (the vectorised one lacked
    the scalar zero-step guard). These tests pin the two paths to the shared
    functions *bit-for-bit*, not merely to within a tolerance.
    """

    R_over_Q = 518.0
    Q_L = 1287601.7251526634

    def _feedback(self) -> IQCavityFeedbackTimingClass:
        """
        Build a minimal feedback exposing both coarse-step paths.

        Returns
        -------
        IQCavityFeedbackTimingClass
            Default-constructed feedback instance.
        """
        return IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=0.0,
            n_cavities=1,
        )

    def test_step_exponent_is_shape_agnostic(self):
        """A scalar step and a one-cell array give the identical exponent."""
        omega_times_dt, rel_det = 2.0 * np.pi, -1e-4
        scalar = coarse_step_exponent(omega_times_dt, self.Q_L, rel_det)
        vector = coarse_step_exponent(
            np.array([omega_times_dt]), self.Q_L, rel_det
        )
        self.assertTrue(np.array_equal(np.array([scalar]), vector))

    def test_exponential_update_is_the_shared_propagator(self):
        """The per-cell exact update is ``e^L v + drive W``, bit-exact."""
        cav = self._feedback()
        v_prev, i_gen, i_beam = 30e6 + 1e6j, 0.02 + 0.01j, 0.005j
        omega_times_dt, rel_det = 2.0 * np.pi, -1e-4
        got = cav._advance_coarse_voltage(
            v_prev, i_gen, i_beam, omega_times_dt, rel_det
        )
        step = coarse_step_exponent(omega_times_dt, self.Q_L, rel_det)
        drive = self.R_over_Q * omega_times_dt * (i_gen - 0.5 * i_beam)
        expected = v_prev * exponential_voltage_multiplier(
            step
        ) + drive * exponential_drive_weight(step)
        self.assertEqual(got, expected)

    def test_kernel_multipliers_match_the_per_cell_step(self):
        """
        Per-cell and vectorised multipliers agree bit-for-bit.

        This is the invariant the numba-vs-python bit-identity pin rests on:
        for the same step, whichever path computes the propagator, the same
        bits come out.
        """
        omega_times_dt = np.array([2.0 * np.pi, 0.5 * np.pi, 1e-6])
        rel_det = -1e-4
        cav = self._feedback()
        multiplier, weight = cav._kernel_step_multipliers(
            omega_times_dt, rel_det
        )
        for cell, step_size in enumerate(omega_times_dt):
            with self.subTest(cell=cell):
                step = coarse_step_exponent(step_size, self.Q_L, rel_det)
                self.assertEqual(
                    multiplier[cell], exponential_voltage_multiplier(step)
                )
                self.assertEqual(weight[cell], exponential_drive_weight(step))

    def test_drive_weight_guards_the_scalar_zero_step_only(self):
        """
        The removable singularity is guarded where it is reachable.

        A scalar zero step reaches the per-cell recursion, so ``W`` must take
        its limit ``1``. The vectorised path never sees one: the step-size
        helper defers a segment containing a coincident coarse point to the
        per-cell loop, which warns and skips it. The guard is therefore
        deliberately not elementwise -- an array zero still yields ``nan``
        rather than costing the hot recursion an extra pass.
        """
        self.assertEqual(exponential_drive_weight(0.0 + 0.0j), 1.0)
        self.assertEqual(exponential_drive_weight(0), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vector = exponential_drive_weight(np.array([0.0 + 0.0j]))
        self.assertTrue(np.isnan(vector[0]))

    def test_zero_step_leaves_the_voltage_untouched(self):
        """A zero-length step returns the input voltage exactly."""
        cav = self._feedback()
        got = cav._advance_coarse_voltage(
            30e6 + 1e6j, 0.02 + 0.01j, 0.005j, 0.0, -1e-4
        )
        self.assertEqual(got, 30e6 + 1e6j)


class TestVoltageSetpointValidation(unittest.TestCase):
    """
    The explicit ``voltage_setpoint`` must be real and positive (phase 0).

    The station's phase correction is referenced to the parent-derived
    setpoint at phase 0, so an explicit setpoint with a non-zero phase would
    be regulated by the PI controller but never reflected in the applied
    kick. The constructor therefore rejects non-real (or non-positive)
    setpoints instead of silently splitting the two frames.
    """

    def _build(self, voltage_setpoint):
        """
        Construct a timing-class feedback with the given setpoint.

        Parameters
        ----------
        voltage_setpoint
            Value passed through to the constructor.

        Returns
        -------
        IQCavityFeedbackTimingClass
            The constructed feedback.
        """
        return IQCavityFeedbackTimingClass(
            profile=Mock(StaticProfile),
            R_over_Q=518.0,
            Q_L=1.29e6,
            generator_current_bias=0.0,
            n_cavities=1,
            voltage_setpoint=voltage_setpoint,
        )

    def test_real_positive_setpoint_accepted(self):
        """A real, positive setpoint (float or 0-phase complex) is accepted."""
        self.assertEqual(self._build(30e6)._voltage_setpoint, 30e6)
        self.assertEqual(
            self._build(30e6 + 0.0j)._voltage_setpoint, 30e6 + 0.0j
        )

    def test_none_setpoint_accepted(self):
        """``None`` (parent-derived setpoint) stays supported."""
        self.assertIsNone(self._build(None)._voltage_setpoint)

    def test_complex_setpoint_raises(self):
        """A setpoint with a non-zero imaginary part raises ``ValueError``."""
        with self.assertRaises(ValueError) as ctx:
            self._build(30e6 + 1e6j)
        self.assertIn("phase 0", str(ctx.exception))

    def test_negative_setpoint_raises(self):
        """A negative (phase pi) setpoint raises ``ValueError``."""
        with self.assertRaises(ValueError):
            self._build(-30e6)


class TestFineGridInitialConditionCausality(unittest.TestCase):
    """
    Causality of the fine-grid initial condition in ``circuit_track``.

    The fine solve is seeded with the coarse envelope at the FIRST
    forward coarse centre ``c0``, and then integrates the beam current
    over ``[profile.cut_left, profile.cut_right]``. Both times live in
    the same segment-local frame, so the seed is only causal when
    ``c0 <= cut_left``: otherwise the coarse cell that produced the
    seed already sits *after* the start of the fine window, and any
    charge in that window would be integrated twice.

    A charge-free window has nothing to be causal about, so the guard is
    gated on the beam current the fine solve actually consumes.
    """

    R_over_Q = 518.0
    Q_L = 1287601.7251526634
    f_rf = 1.3e9
    omega_rf = 2.0 * np.pi * f_rf
    t_rf = 1.0 / f_rf
    n_bins = 128
    n_steps = 8

    def _drive(self, cut_left_rad, with_charge):
        """
        Drive ``circuit_track`` over a hand-built constant-step grid.

        The coarse centres are ``(k + 0.5) * t_rf``, so the first
        forward centre sits at ``0.5 * t_rf`` (i.e. ``pi`` in the
        radian units of the profile window).

        Parameters
        ----------
        cut_left_rad
            Left edge of the profile window [rad]; the window is three
            RF periods wide.
        with_charge
            Whether the fine grid carries a non-zero beam current.

        Returns
        -------
        cav
            The feedback instance that was driven.
        """
        profile = StaticProfile.from_rad(
            cut_left_rad, cut_left_rad + 3.0 * np.pi, self.n_bins, self.t_rf
        )
        cav = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            generator_current_bias=0.0 + 0.0j,
            n_cavities=1,
            delta_omega=0.0,
            initial_voltage=0.0,
        )
        cav._rf_centers = (np.arange(self.n_steps) + 0.5) * self.t_rf
        cav._rf_centers_lengths = np.array([self.n_steps])
        cav._residual_time_last_rf_centers_calculation = 0.0
        cav._last_rf_centers_entry = None
        cav.reset_arrays()
        cav.beam_current_forward_coarse_grid = np.zeros(
            self.n_steps, dtype=complex
        )
        cav.beam_current_fine_grid = np.full(
            self.n_bins, 0.02 + 0.0j if with_charge else 0.0 + 0.0j
        )
        cav.circuit_track(
            omega_input=self.omega_rf,
            no_beam=False,
            start_index=0,
            end_index=self.n_steps,
        )
        return cav

    def test_charge_before_first_coarse_centre_raises(self):
        """Charge left of the first forward coarse centre is acausal."""
        with self.assertRaises(ValueError) as ctx:
            self._drive(cut_left_rad=0.5 * np.pi, with_charge=True)
        message = str(ctx.exception)
        self.assertIn("cut_left", message)
        self.assertIn("first forward coarse centre", message)
        self.assertIn("sampling_time_coarse", message)

    def test_charge_free_window_before_first_centre_is_allowed(self):
        """A charge-free window has nothing to be causal about."""
        cav = self._drive(cut_left_rad=0.5 * np.pi, with_charge=False)
        self.assertIsNotNone(cav.antenna_voltage_fine_grid)

    def test_charge_right_of_first_coarse_centre_is_allowed(self):
        """The physical geometry (cut_left >= c0) stays accepted."""
        cav = self._drive(cut_left_rad=1.5 * np.pi, with_charge=True)
        self.assertIsNotNone(cav.antenna_voltage_fine_grid)


if __name__ == "__main__":
    unittest.main()
