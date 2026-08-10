"""
Tests for the cavity-response solvers.

Both ``cavity_response_sparse_matrix`` and
``cavity_response_sparse_matrix_second_order`` live in
``blond.physics.feedbacks.cavity_solvers``.

Both :func:`cavity_response_sparse_matrix` (forward Euler, left-endpoint drive)
and :func:`cavity_response_sparse_matrix_second_order` (trapezoidal /
Crank-Nicolson) integrate the same cavity-envelope ODE and converge to the same
induced voltage. The first is ``O(dt)`` accurate, the second ``O(dt^2)``, so at
coarse profile binning the second-order solver is far closer to the
(essentially exact) multi-turn resonator convolution.

The solvers are driven directly on a static profile -- no ``Beam`` tracking and
no full ``Simulation`` -- with small mock objects.
"""

import os
import unittest
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from blond import Resonators, StaticProfile, WakeField, backend, mu_minus
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.beam_current import rf_beam_current
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.cavity_solvers import (
    cavity_response_sparse_matrix,
    cavity_response_sparse_matrix_second_order,
)
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Package-relative imports: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .stubs import StubBeam, StubRFStation
from .support import (
    lab_frame_voltage,
    rel_err,
)

DEBUG_PLOT = False


class TestCavityResponseSolverConvergence(unittest.TestCase):
    """First-order (Euler) vs second-order (Crank-Nicolson) cavity response."""

    def setUp(self):
        """Set up mucol RCS1-like single-cavity parameters."""
        self.R_over_Q = 518.0
        self.Q_L = 1.29e6
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.f_res = 1.0 / self.t_rf
        self.circumference = 5990.0
        self.intensity = 2.7e12

    def _smooth_profile(self, n_slices: int) -> StaticProfile:
        """
        A smooth Gaussian bunch on ``n_slices`` bins, with zeroed edges.

        Parameters
        ----------
        n_slices
            Number of profile bins.

        Returns
        -------
        StaticProfile
            Gaussian bunch profile with zeroed leading/trailing bins.
        """
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, n_slices, self.t_rf
        )
        t = copy_to_cpu(profile.hist_x)
        t0 = 0.5 * (t[0] + t[-1])
        hist_y = np.exp(-0.5 * ((t - t0) / (0.08 * self.t_rf)) ** 2)

        # zero edges
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        return profile

    def _calculate_induced_voltage(
        self,
        n_slices: int,
        method: Literal["convolution", "first_order", "second_order"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Induced voltage on ``n_slices`` bins; ``method`` selects the solver.

        ``method`` selects the model: ``"convolution"`` (multi-turn resonator
        convolution), ``"first_order"`` (forward-Euler cavity response) or
        ``"second_order"`` (Crank-Nicolson cavity response).

        Parameters
        ----------
        n_slices
            Number of profile bins.
        method
            Solver to use: ``"convolution"``, ``"first_order"`` or
            ``"second_order"``.

        Returns
        -------
        voltage
            Lab-frame induced voltage on the profile's fine grid.
        hist_x
            Bin-center time base of the profile.
        """
        profile = self._smooth_profile(n_slices)
        t = copy_to_cpu(profile.hist_x)
        stub_beam = StubBeam(self.intensity)

        if method == "convolution":
            solver = MultiPassResonatorSolver(decay_fraction_threshold=1e-12)
            wakefield = WakeField(
                sources=(
                    Resonators(self.R_over_Q * self.Q_L, self.f_res, self.Q_L),
                ),
                solver=solver,
                profile=profile,
                parent_rf_station=StubRFStation(self.omega_rf),
            )
            solver._parent_wakefield = wakefield
            solver.circumference = self.circumference
            solver._maximum_storage_time = 1.0
            solver._last_reference_time = -np.finfo(float).eps
            return copy_to_cpu(solver.calc_induced_voltage(stub_beam)), t

        charges = rf_beam_current(
            beam=stub_beam,
            profile=profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
        )
        solver_fn = {
            "first_order": cavity_response_sparse_matrix,
            "second_order": cavity_response_sparse_matrix_second_order,
        }[method]
        v_ant = solver_fn(
            I_beam=charges
            / profile.hist_step,  # TODO: refactor to include in beam_current calculation
            I_gen=np.zeros(n_slices, dtype=complex),
            V_ant_init=0.0,
            I_gen_init=0.0,
            omega_times_dt=self.omega_rf * profile.hist_step,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            relative_detuning=0.0,
        )
        return lab_frame_voltage(v_ant, self.omega_rf, t), t

    def _plot_convergence(self):
        """
        Save a debug plot of solver convergence and low-binning residuals.

        Disabled by default. Enable by setting the module-level
        ``support.DEBUG_PLOTS`` constant to ``"save"`` (write a PNG) or
        ``"show"`` (also open an interactive window).

        Left panel: relative error vs a high-resolution truth as a function of
        the bin count, for forward Euler and Crank-Nicolson, with reference
        ``1/n`` and ``1/n^2`` slopes. Right panel: the residual against the
        convolution solver at coarse binning, showing Euler's systematic bump
        and the near-zero Crank-Nicolson residual.
        """
        v_truth, t_truth = self._calculate_induced_voltage(
            8192, "second_order"
        )
        n_bins = np.array([128, 256, 512, 1024, 2048, 4096])
        err_first_order, err_second_order = [], []
        for n in n_bins:
            v_first_order, t = self._calculate_induced_voltage(
                n, "first_order"
            )
            v_second_order, _ = self._calculate_induced_voltage(
                n, "second_order"
            )
            truth = np.interp(t, t_truth, v_truth)
            err_first_order.append(rel_err(v_first_order, truth))
            err_second_order.append(rel_err(v_second_order, truth))
        err_first_order = np.array(err_first_order)
        err_second_order = np.array(err_second_order)

        n_lo = 256
        v_convolution, t_lo = self._calculate_induced_voltage(
            n_lo, "convolution"
        )
        v_first_order_lo, _ = self._calculate_induced_voltage(
            n_lo, "first_order"
        )
        v_second_order_lo, _ = self._calculate_induced_voltage(
            n_lo, "second_order"
        )
        peak = np.max(np.abs(v_convolution))

        fig, (ax_conv, ax_res) = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle("Cavity-response solver: Euler vs Crank-Nicolson")
        ax_conv.loglog(
            n_bins,
            err_first_order,
            "o-",
            color="C0",
            label="forward Euler (1st order)",
        )
        ax_conv.loglog(
            n_bins,
            err_second_order,
            "s-",
            color="C1",
            label="Crank-Nicolson (2nd order)",
        )
        ref_first = err_first_order[0] * n_bins[0] / n_bins
        ref_second = err_second_order[0] * (n_bins[0] / n_bins) ** 2
        ax_conv.loglog(
            n_bins, ref_first, "k--", lw=0.8, label=r"$\propto 1/n$"
        )
        ax_conv.loglog(
            n_bins, ref_second, "k:", lw=0.8, label=r"$\propto 1/n^2$"
        )
        ax_conv.set_xlabel("profile bins")
        ax_conv.set_ylabel("rel. error vs truth")
        ax_conv.legend(loc="best")
        ax_conv.grid(True, which="both", alpha=0.3)

        residual_first = (v_first_order_lo - v_convolution) / peak * 100
        residual_second = (v_second_order_lo - v_convolution) / peak * 100
        ax_res.plot(
            t_lo * 1e9, residual_first, color="C0", label="Euler - convolution"
        )
        ax_res.plot(
            t_lo * 1e9,
            residual_second,
            color="C1",
            label="Crank-Nicolson - convolution",
        )
        ax_res.set_xlabel("time [ns]")
        ax_res.set_ylabel("residual [% of peak]")
        ax_res.set_title(f"residual vs convolution at {n_lo} bins")
        ax_res.legend(loc="best")
        ax_res.grid(True, alpha=0.3)

        fig.tight_layout()

        # Debug-save next to this test file (not the repo root); needs ``import os``:
        # plt.savefig(
        #     os.path.join(os.path.dirname(__file__), "residual_vs_convolution.png"),
        #     dpi=200, bbox_inches="tight",
        # )
        plt.show()

    def test_second_order_more_accurate_at_low_binning(self):
        """At coarse binning the CN solver beats Euler by orders of magnitude."""
        n = 256
        v_convolution, _ = self._calculate_induced_voltage(n, "convolution")
        v_first_order, _ = self._calculate_induced_voltage(n, "first_order")
        v_second_order, _ = self._calculate_induced_voltage(n, "second_order")

        err_first_order = rel_err(v_first_order, v_convolution)
        err_second_order = rel_err(v_second_order, v_convolution)

        # Euler is ~1 % here; CN agrees with the convolution far better.
        self.assertGreater(err_first_order, 1e-3)
        self.assertLess(err_second_order, err_first_order / 50.0)

    def test_solver_convergence_orders(self):
        """Euler converges at first order, Crank-Nicolson at second order."""
        # Debug plot (opt-in) before the assertions.
        if DEBUG_PLOT:
            self._plot_convergence()

        # High-resolution truth (the second-order solver on a fine grid).
        v_truth, t_truth = self._calculate_induced_voltage(
            8192, "second_order"
        )

        def err(method, n):
            v, t = self._calculate_induced_voltage(n, method)
            return rel_err(v, np.interp(t, t_truth, v_truth))

        # Error ratio when halving the bin size: ~2 for 1st order, ~4 for 2nd.
        first_order_ratio = err("first_order", 256) / err("first_order", 512)
        second_order_ratio = err("second_order", 256) / err(
            "second_order", 512
        )

        # 1.7 < error_ratio_first_order < 2.5 --> should be 2 (linear)
        self.assertGreater(first_order_ratio, 1.7)
        self.assertLess(first_order_ratio, 2.5)

        # 3.0 < error_ratio_first_order < 4.5 --> should be 4 (quadratic)
        self.assertGreater(second_order_ratio, 3.0)
        self.assertLess(second_order_ratio, 4.5)

    def test_solvers_agree_as_predicted_by_convergence_rate(self):
        """
        The two helpers agree to within their (first-order) convergence gap.

        Both functions integrate the *same* cavity ODE, so the difference
        between their outputs is the difference of their truncation errors.
        The second-order solver is far more accurate, so that difference is
        dominated by the Euler error and is therefore ``O(dt)``: small at fine
        binning, and halving each time the bin count doubles. We check both the
        magnitude and that first-order scaling -- i.e. the two solvers are
        exactly as similar as the convergence rate predicts.
        """

        def rel_diff(n):
            v_first_order, _ = self._calculate_induced_voltage(
                n, "first_order"
            )
            v_second_order, _ = self._calculate_induced_voltage(
                n, "second_order"
            )
            return rel_err(v_first_order, v_second_order)

        diff_256 = rel_diff(256)
        diff_512 = rel_diff(512)

        # Small, but not vanishing, at coarse binning (this is the Euler error).
        self.assertGreater(diff_256, 1e-3)
        self.assertLess(diff_256, 5e-2)
        # Halving the bin size halves the disagreement (first-order scaling).
        self.assertAlmostEqual(diff_256 / diff_512, 2.0, delta=0.4)

    def test_second_order_flag_routes_through_the_class(self):
        """
        The ``second_order_fine_grid_solver_enable`` flag selects the solver.

        With the flag off the class reproduces the first-order solver, with it
        on it reproduces the second-order one (bit-for-bit, ``n_cavities=1``),
        and the second-order result is far closer to the convolution at coarse
        binning.
        """
        n = 256
        smooth_profile = self._smooth_profile(n)
        stub_beam = StubBeam(self.intensity)
        charges = rf_beam_current(
            beam=stub_beam,
            profile=smooth_profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
        )
        i_beam = charges / smooth_profile.hist_step
        omega_times_dt = self.omega_rf * smooth_profile.hist_step

        def create_fdbk_class_and_calc_voltage(second_order):
            feedback = IQCavityFeedbackTimingClass(
                profile=smooth_profile,
                R_over_Q=self.R_over_Q,
                Q_L=self.Q_L,
                generator_current_bias=0.0,
                n_cavities=1,
                initial_voltage=0.0,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
                second_order_fine_grid_solver_enable=second_order,
            )
            feedback.beam_current_fine_grid = i_beam
            feedback.generator_current_fine_grid = np.zeros(n, dtype=complex)
            feedback.cavity_response_fine(
                initial_voltage_fine_grid=0.0,
                initial_generator_current_fine_grid=0.0,
                omega_times_dt_fine_grid=omega_times_dt,
                relative_detuning=0.0,
            )
            return feedback.antenna_voltage_fine_grid

        kwargs = {
            "I_beam": i_beam,
            "I_gen": np.zeros(n, dtype=complex),
            "V_ant_init": 0.0,
            "I_gen_init": 0.0,
            "omega_times_dt": omega_times_dt,
            "R_over_Q": self.R_over_Q,
            "Q_L": self.Q_L,
            "relative_detuning": 0.0,
        }
        np.testing.assert_allclose(
            create_fdbk_class_and_calc_voltage(False),
            cavity_response_sparse_matrix(**kwargs),
        )
        np.testing.assert_allclose(
            create_fdbk_class_and_calc_voltage(True),
            cavity_response_sparse_matrix_second_order(**kwargs),
        )

        v_convolution, t = self._calculate_induced_voltage(n, "convolution")

        def project_voltage_into_lab_frame(v_ant):
            return lab_frame_voltage(v_ant, self.omega_rf, t)

        err_first = rel_err(
            project_voltage_into_lab_frame(
                create_fdbk_class_and_calc_voltage(False)
            ),
            v_convolution,
        )
        err_second = rel_err(
            project_voltage_into_lab_frame(
                create_fdbk_class_and_calc_voltage(True)
            ),
            v_convolution,
        )
        self.assertLess(err_second, err_first / 50.0)


class TestRfBeamCurrentDownsampling(unittest.TestCase):
    """
    Charge conservation of the coarse-grid downsampling in rf_beam_current.

    Regression test for a dropped remainder in the coarse-grid branch of
    :func:`blond.physics.feedbacks.beam_current.rf_beam_current`: all demodulated
    charge after the last coarse-cell boundary used to be silently discarded
    -- up to the *whole* bunch, depending on how the bunch sits relative to
    the cell boundaries (in the multi-turn comparison it was ~45 % with a
    ~25 deg rotated phase centroid). That corrupted the coarse-grid beam
    loading and every quantity propagated from it across turns, while the
    fine grid (and with it every single-turn comparison) stayed correct.

    The invariant under test: re-binning the fine-grid demodulated charge
    onto the coarse grid must count every fine bin exactly once, so the
    complex sums of both arrays must be identical -- wherever the cell
    boundaries fall relative to the bunch.
    """

    def setUp(self):
        """Set up RF and downsampling parameters."""
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.intensity = 2.7e12
        self.n_slices = 1024
        # Coarse cells of one RF period each, RCS1-like cell count.
        self.n_points_coarse = 25900

    def _profile_with_bunch_at(self, window_fraction: float) -> StaticProfile:
        """
        A narrow Gaussian bunch centered at a given fraction of the window.

        Parameters
        ----------
        window_fraction
            Bunch-center position as a fraction of the profile window
            (0 = left edge, 1 = right edge).

        Returns
        -------
        StaticProfile
            Profile with the bunch at the requested position.
        """
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, self.n_slices, self.t_rf
        )
        t = copy_to_cpu(profile.hist_x)
        t0 = t[0] + window_fraction * (t[-1] - t[0])
        hist_y = np.exp(-0.5 * ((t - t0) / (0.02 * self.t_rf)) ** 2)
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        return profile

    def test_downsampling_conserves_demodulated_charge(self):
        """
        The coarse-grid re-binning conserves the total demodulated charge.

        The window spans 1.5 RF periods, i.e. it crosses coarse-cell
        boundaries (here at 1.0 and 2.0 t_rf). The swept bunch positions
        cover every downsampling segment: fully in the *leading* segment
        before the first boundary (0.08), straddling the first boundary
        (0.2), mid-window inside a cell (0.5) -- all conserved even pre-fix,
        since only charge past the *last* boundary was dropped -- and fully
        in the *trailing* segment after the last boundary (0.9), where
        pre-fix all of the bunch charge was silently discarded.
        """
        for window_fraction in (0.08, 0.2, 0.5, 0.9):
            with self.subTest(window_fraction=window_fraction):
                profile = self._profile_with_bunch_at(window_fraction)
                charges_fine, charges_coarse = rf_beam_current(
                    beam=StubBeam(self.intensity),
                    profile=profile,
                    omega_c=self.omega_rf,
                    use_lowpass_filter=False,
                    dT=0.0,
                    sampling_time=self.t_rf,
                    n_points=self.n_points_coarse,
                )
                sum_fine = np.sum(charges_fine)
                sum_coarse = np.sum(charges_coarse)
                # Guard against a vacuous comparison.
                self.assertGreater(np.abs(sum_fine), 0.0)
                np.testing.assert_allclose(sum_coarse, sum_fine, rtol=1e-9)

    def test_warns_when_beam_maps_before_turn_zero(self):
        """A coarse index below zero warns that the beam is before turn 0."""
        # A large negative time shift maps the leading bins to coarse indices
        # below zero (``ind_fine < 0``), which the downsampling cannot place
        # correctly.
        profile = self._profile_with_bunch_at(0.5)
        with self.assertWarnsRegex(UserWarning, "before turn time 0"):
            rf_beam_current(
                beam=StubBeam(self.intensity),
                profile=profile,
                omega_c=self.omega_rf,
                use_lowpass_filter=False,
                dT=-2.0 * self.t_rf,
                sampling_time=self.t_rf,
                n_points=self.n_points_coarse,
            )

    def _downsampled(self, profile, **kwargs):
        """
        Call rf_beam_current with the standard downsampling arguments.

        Parameters
        ----------
        profile
            Static profile to demodulate.
        **kwargs
            Extra keyword arguments forwarded to ``rf_beam_current``.

        Returns
        -------
        charges_fine
            Demodulated charge on the fine grid.
        charges_coarse
            Demodulated charge on the coarse grid.
        """
        return rf_beam_current(
            beam=StubBeam(self.intensity),
            profile=profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
            sampling_time=self.t_rf,
            n_points=self.n_points_coarse,
            **kwargs,
        )

    def test_error_when_first_coarse_cell_populated(self):
        """
        Charge in the first coarse cell raises when forbidden.

        The fine-grid initial antenna voltage of
        ``IQCavityFeedbackTimingClass`` is taken from the first coarse cell
        (see ``circuit_track``), so beam charge there would be
        double-counted; the class therefore calls ``rf_beam_current`` with
        ``forbid_charge_in_first_coarse_cell=True``. A bunch early in the
        window (before the first cell boundary) populates exactly that cell.
        """
        profile = self._profile_with_bunch_at(0.08)
        with self.assertRaises(ValueError) as cm:
            self._downsampled(profile, forbid_charge_in_first_coarse_cell=True)
        self.assertIn("first coarse-grid cell", str(cm.exception))

    def test_no_error_when_first_coarse_cell_empty(self):
        """
        A mid-window bunch leaves the first cell numerically empty.

        The far Gaussian tail is non-zero in float arithmetic (~1e-100), so
        the guard must use a relative threshold rather than ``!= 0``.
        """
        profile = self._profile_with_bunch_at(0.5)
        charges_fine, charges_coarse = self._downsampled(
            profile, forbid_charge_in_first_coarse_cell=True
        )
        self.assertLess(
            np.abs(charges_coarse[0]),
            1e-9 * np.sum(np.abs(charges_fine)),
        )
        self.assertGreater(np.abs(np.sum(charges_fine)), 0.0)

    def test_warns_on_particle_loss(self):
        """
        Warn when the profile does not capture the whole beam.

        Particles lost or outside the profile window are invisible to the
        feedback's beam current; ``rf_beam_current`` must warn about them
        before anything else. Modeled by a density factor corresponding to
        only half of the macroparticles being inside the window.
        """
        profile = self._profile_with_bunch_at(0.5)
        profile.hist_y_to_density_factor = 0.5 / float(
            np.sum(copy_to_cpu(profile.hist_y))
        )
        with self.assertWarns(UserWarning) as cm:
            self._downsampled(profile)
        self.assertIn("not be treated correctly", str(cm.warning))

    def test_no_warning_when_profile_captures_full_beam(self):
        """No particle-loss warning when the window captures everything."""
        profile = self._profile_with_bunch_at(0.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._downsampled(profile)
        self.assertEqual(caught, [])

    def _profile_over(self, cut_left_trf, cut_right_trf, centers_trf):
        """
        Static profile with narrow Gaussian bunches at given t_rf offsets.

        Parameters
        ----------
        cut_left_trf
            Left window edge, in units of ``t_rf``.
        cut_right_trf
            Right window edge, in units of ``t_rf``.
        centers_trf
            Bunch centres, in units of ``t_rf``.

        Returns
        -------
        StaticProfile
            The populated profile.
        """
        profile = StaticProfile(
            cut_left=cut_left_trf * self.t_rf,
            cut_right=cut_right_trf * self.t_rf,
            n_bins=self.n_slices,
        )
        t = copy_to_cpu(profile.hist_x)
        hist_y = np.zeros_like(t)
        for center in centers_trf:
            hist_y += np.exp(
                -0.5 * ((t - center * self.t_rf) / (0.02 * self.t_rf)) ** 2
            )
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        return profile

    def test_raises_when_profile_longer_than_coarse_grid(self):
        """
        A window longer than the coarse grid raises instead of wrapping.

        Two bunches 3 t_rf apart in a 5 t_rf window onto a 3-cell grid: the
        trailing bunch's coarse index wrapped onto the leading bunch's cell
        and *overwrote* it, silently losing 50 % of the demodulated charge
        (measured sum_coarse/sum_fine relative error 0.5).
        """
        profile = self._profile_over(0.0, 5.0, (1.5, 4.5))
        with self.assertRaises(ValueError) as cm:
            rf_beam_current(
                beam=StubBeam(self.intensity),
                profile=profile,
                omega_c=self.omega_rf,
                use_lowpass_filter=False,
                dT=0.0,
                sampling_time=self.t_rf,
                n_points=3,
            )
        self.assertIn("longer than", str(cm.exception))

    def test_particle_loss_warning_is_not_shadowed_by_the_span_guard(self):
        """
        The span guard must not pre-empt the particle-loss warning.

        The two answer different questions -- "is the beam inside the
        profile?" versus "does the profile fit the span it is re-binned
        onto?" -- and the fold destroys charge even when the whole beam
        is captured. So the loss warning has to stay reachable: it is
        emitted before the span check, and raising there must not stop it
        being seen.
        """
        profile = self._profile_over(0.0, 5.0, (1.5, 4.5))
        profile.hist_y_to_density_factor = 0.5 / float(
            np.sum(copy_to_cpu(profile.hist_y))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                rf_beam_current(
                    beam=StubBeam(self.intensity),
                    profile=profile,
                    omega_c=self.omega_rf,
                    use_lowpass_filter=False,
                    dT=0.0,
                    sampling_time=self.t_rf,
                    n_points=3,
                )
        loss_warnings = [
            str(entry.message)
            for entry in caught
            if "inside the profile window" in str(entry.message)
        ]
        self.assertEqual(len(loss_warnings), 1)

    def test_raises_when_profile_starts_after_coarse_grid(self):
        """A window past the grid raises ValueError, not a bare IndexError."""
        profile = self._profile_over(10.0, 12.0, (11.0,))
        with self.assertRaises(ValueError) as cm:
            rf_beam_current(
                beam=StubBeam(self.intensity),
                profile=profile,
                omega_c=self.omega_rf,
                use_lowpass_filter=False,
                dT=0.0,
                sampling_time=self.t_rf,
                n_points=3,
            )
        self.assertIn("coarse-grid index", str(cm.exception))

    def test_long_window_that_fits_still_conserves_charge(self):
        """
        The guard must not fire on a long-but-fitting window.

        Anti-false-positive pin: the same 5 t_rf two-bunch window that the
        3-cell grid rejects is perfectly valid on an 8-cell grid, and must
        keep conserving charge. This is the geometry
        ``test_multibunch_beam_loading`` relies on -- an ~8 t_rf window on a
        13-cell grid, the widest legitimate window/span ratio in the suite
        (0.62) -- so the threshold may not creep below it.
        """
        profile = self._profile_over(0.0, 5.0, (1.5, 4.5))
        charges_fine, charges_coarse = rf_beam_current(
            beam=StubBeam(self.intensity),
            profile=profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
            sampling_time=self.t_rf,
            n_points=8,
        )
        self.assertGreater(np.abs(np.sum(charges_fine)), 0.0)
        np.testing.assert_allclose(
            np.sum(charges_coarse), np.sum(charges_fine), rtol=1e-9
        )


class TestRfBeamCurrentCounterRotating(unittest.TestCase):
    """
    Direction-signed charge in the RF beam current.

    In the symmetric muon-collider ring the counter-rotating mu-minus beam has
    *opposite charge and opposite direction*, so its gap (RF beam) current has
    the **same sign** as the co-rotating mu-plus beam and both beams see the
    same beam loading. ``rf_beam_current``
    therefore uses ``beam.signed_charge_with_direction()`` (charge negated for a
    counter-rotating beam) on the source side, matching the RF-kick and
    wake-kick conventions. For co-rotating beams the signed charge equals the
    plain particle charge, so co-rotating behaviour is bit-unchanged.

    The full sign matrix is pinned: flipping *either* charge *or* direction
    alone flips the current; flipping both restores it.
    """

    def setUp(self):
        """Set up RF parameters and a shared mid-window bunch profile."""
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.intensity = 2.7e12
        # Mid-window Gaussian: keeps the first coarse cell charge-free for
        # the partial (mucol) downsampling path.
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, 1024, self.t_rf
        )
        t = copy_to_cpu(profile.hist_x)
        t0 = t[0] + 0.5 * (t[-1] - t[0])
        hist_y = np.exp(-0.5 * ((t - t0) / (0.02 * self.t_rf)) ** 2)
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        self.profile = profile

    def _fine_current(self, beam) -> np.ndarray:
        """
        Fine-grid RF beam charge from the base-class ``rf_beam_current``.

        Parameters
        ----------
        beam
            Beam (stub) to compute the current for.

        Returns
        -------
        numpy.ndarray
            Complex fine-grid RF beam charge.
        """
        return rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.omega_rf,
            use_lowpass_filter=False,
            dT=0.0,
        )

    def _partial_currents(self, beam) -> tuple[np.ndarray, np.ndarray]:
        """
        Fine and coarse RF beam charge from the mucol forward-pass path.

        Parameters
        ----------
        beam
            Beam (stub) to compute the current for.

        Returns
        -------
        charges_fine
            Complex fine-grid RF beam charge.
        charges_coarse
            Complex coarse-grid RF beam charge.
        """
        return rf_beam_current(
            beam=beam,
            profile=self.profile,
            omega_c=self.omega_rf,
            sampling_time=self.t_rf,
            n_points=8,
            dT=0.0,
            forbid_charge_in_first_coarse_cell=True,
        )

    def test_counter_rotating_mu_minus_matches_co_rotating_mu_plus(self):
        """
        The CR mu-minus current is bit-identical to the mu-plus current.

        Opposite charge x opposite direction = same gap current: the
        symmetric-ring requirement. Before the direction-signed charge this
        was exactly sign-flipped, so this test pins the fix on both the
        fine-only and the coarse (forward-pass) ``rf_beam_current``
        paths.
        """
        beam_plus = StubBeam(self.intensity)
        beam_minus_cr = StubBeam(
            self.intensity, particle_type=mu_minus, is_counter_rotating=True
        )

        np.testing.assert_array_equal(
            self._fine_current(beam_minus_cr), self._fine_current(beam_plus)
        )

        fine_plus, coarse_plus = self._partial_currents(beam_plus)
        fine_minus, coarse_minus = self._partial_currents(beam_minus_cr)
        np.testing.assert_array_equal(fine_minus, fine_plus)
        np.testing.assert_array_equal(coarse_minus, coarse_plus)
        # Non-degenerate: there is real current to compare.
        self.assertGreater(np.max(np.abs(fine_plus)), 0.0)

    def test_co_rotating_mu_minus_flips_the_sign(self):
        """
        A co-rotating mu-minus beam still flips the current sign.

        Ordinary opposite-charge physics (same direction) must be untouched
        by the direction handling: charge alone flips the current.
        """
        beam_plus = StubBeam(self.intensity)
        beam_minus = StubBeam(self.intensity, particle_type=mu_minus)
        np.testing.assert_array_equal(
            self._fine_current(beam_minus), -self._fine_current(beam_plus)
        )
        fine_plus, coarse_plus = self._partial_currents(beam_plus)
        fine_minus, coarse_minus = self._partial_currents(beam_minus)
        np.testing.assert_array_equal(fine_minus, -fine_plus)
        np.testing.assert_array_equal(coarse_minus, -coarse_plus)

    def test_counter_rotating_mu_plus_flips_the_sign(self):
        """
        A counter-rotating mu-plus beam flips the current sign.

        Direction alone flips the gap current (same charge, opposite
        traversal) -- the complementary corner of the sign matrix.
        """
        beam_plus = StubBeam(self.intensity)
        beam_plus_cr = StubBeam(self.intensity, is_counter_rotating=True)
        np.testing.assert_array_equal(
            self._fine_current(beam_plus_cr), -self._fine_current(beam_plus)
        )

    def test_co_rotating_signed_charge_is_the_plain_charge(self):
        """
        For a co-rotating beam the signed charge is the plain charge.

        This is the bit-identity guarantee for co-rotating beams: the
        source-side switch to ``signed_charge_with_direction()`` changes
        nothing for any co-rotating beam.
        """
        beam_plus = StubBeam(self.intensity)
        self.assertEqual(
            beam_plus.signed_charge_with_direction(),
            beam_plus.particle_type.charge,
        )
        beam_minus = StubBeam(self.intensity, particle_type=mu_minus)
        self.assertEqual(
            beam_minus.signed_charge_with_direction(),
            beam_minus.particle_type.charge,
        )


@unittest.skipIf(
    os.environ.get("BLOND_BACKEND_MODE", "").lower() == "cuda",
    "byte-identity pin: the reference values were recorded on the host "
    "backend, and CuPy's linspace/reductions differ from NumPy's by 1-2 ULP",
)
class TestUnifiedRfBeamCurrentMigrationPin(unittest.TestCase):
    """
    Migration pin: the unified ``rf_beam_current`` coarse path.

    The values below were recorded by driving the pre-merge
    ``rf_beam_current_partial`` (the timing-class forward-pass variant that
    was folded into ``rf_beam_current``) on this exact fixture, with ``dT``
    and ``carrier_phase_offset`` both nonzero and the first-coarse-cell
    guard active. The unified function must reproduce them byte-exactly:
    this pins the merged coarse path (bin-centre offset
    ``sampling_time / 2``, carrier-phase rotation, downsampling loop,
    remainder handling) to the pre-merge behaviour.

    The pin is host-only by construction: the recorded values came from the
    NumPy backend, and the profile's bin centres are built with
    ``backend.linspace``, whose CuPy result differs from NumPy's in ~16 % of
    the bins by up to 2.2e-16 relative. That propagates through
    ``cos``/``sin(omega_c t)`` into 1-2 ULP differences (measured max 6.3e-15
    relative over the whole fine grid) -- benign rounding, but fatal to an
    exact-equality pin, so the GPU validation run skips it rather than
    reporting a failure that is not a defect.
    """

    def setUp(self):
        """Set up the recorded fixture: mid-window Gaussian, 1024 bins."""
        self.t_rf = 1.0e-9
        self.omega_rf = 2.0 * np.pi / self.t_rf
        self.intensity = 2.7e12
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, 1024, self.t_rf
        )
        t = copy_to_cpu(profile.hist_x)
        t0 = t[0] + 0.5 * (t[-1] - t[0])
        hist_y = np.exp(-0.5 * ((t - t0) / (0.02 * self.t_rf)) ** 2)
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0
        profile._hist_y = backend.array(hist_y, dtype=backend.float)
        profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)
        self.profile = profile

    def test_unified_coarse_path_matches_recorded_partial_output(self):
        """The unified function reproduces the recorded outputs exactly."""
        charges_fine, charges_coarse = rf_beam_current(
            beam=StubBeam(self.intensity),
            profile=self.profile,
            omega_c=self.omega_rf,
            sampling_time=self.t_rf,
            n_points=8,
            dT=0.1 * self.t_rf,
            carrier_phase_offset=0.123,
            forbid_charge_in_first_coarse_cell=True,
        )

        # Fine-grid checksums, recorded from the pre-merge partial path.
        self.assertEqual(
            complex(np.sum(charges_fine)),
            complex(-5.859266402363611e-07, 6.272885835455287e-07),
        )
        self.assertEqual(
            float(np.sum(np.abs(charges_fine))),
            8.651753823599999e-07,
        )
        self.assertEqual(
            complex(charges_fine[512]),
            complex(-1.71594303215507e-08, 1.8541071115626816e-08),
        )

        # Full recorded coarse grid (8 cells).
        recorded_coarse = np.array(
            [
                complex(1.5735508303327867e-205, -2.7422660451546783e-206),
                complex(-5.859266402363609e-07, 6.272885835455287e-07),
                complex(1.0237497371457826e-94, -1.9550762315465037e-95),
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )
        np.testing.assert_array_equal(charges_coarse, recorded_coarse)


if __name__ == "__main__":
    unittest.main()
