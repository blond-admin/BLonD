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
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import Resonators, StaticProfile, WakeField
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
        self.noisy_profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, self.n_slices, self.t_rf
        )
        t = self.noisy_profile.hist_x
        t0 = 0.5 * (t[0] + t[-1])
        sigma = 0.08 * self.t_rf

        rng = np.random.default_rng(12345)
        hist_y = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        hist_y = hist_y + 0.05 * rng.standard_normal(self.n_slices)
        hist_y = np.clip(hist_y, 0.0, None)
        # The resonator solver warns / can go unstable with charge in the
        # leading or trailing edge bins, so force them to zero.
        hist_y[:5] = 0.0
        hist_y[-5:] = 0.0

        self.noisy_profile._hist_y = hist_y
        self.noisy_profile.hist_y_to_density_factor = 1.0 / np.sum(hist_y)

        self.stub_beam = StubBeam(self.intensity)

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


if __name__ == "__main__":
    unittest.main()
