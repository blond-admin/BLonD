"""
Energy-gain consistency of the multi-turn wake and a non-driven feedback.

This test tracks an *actual* ``BiGaussian`` ``mu_plus`` bunch through a real
``Simulation`` and checks that the induced-voltage energy gain applied to the
particles is the same whether it comes from

* the multi-turn resonator solver (:class:`MultiPassResonatorSolver`,
  applied as a separate wake kick), or
* an :class:`IQCavityFeedbackTimingClass` at its operating point
  (``V_init = V_design`` held by the matched generator current
  ``I_g = V / (2 (R/Q) Q_L)``, ``n_cavities = 1``), whose gap voltage is the
  lab-frame antenna voltage; the beam-induced part is isolated by subtracting
  a zero-intensity reference run (exact, by linearity of the cavity
  equation). A cold or undriven cavity is not usable here: it trips the
  coarse-grid beam-kick magnitude check, whose heuristic assumes an
  established antenna voltage.

Both runs share the same minimal ring (one ``DriftSimple`` + one
``SingleHarmonicRFStation``) and the same initial bunch.  The design RF kick is
suppressed (``voltage = 0`` in the MTW run; the feedback replaces the design
voltage with the antenna voltage anyway), so the only energy change per
particle is the induced voltage.  The two paths agree on the applied ``dE`` to
well below 1 %.

The comparison runs for a stationary ``ConstantMagneticCycle`` and for an
accelerating ``MagneticCyclePerTurn``.  With acceleration, the reference energy
climbs by the programmed ``delta_E_turn`` while (the design kick being
suppressed) the particles are not accelerated, so every particle additionally
receives the common acceleration kick ``-delta_E_turn``; the induced parts are
compared after removing that common offset.  ``delta_E_turn`` must stay below
the design voltage, because the feedback evaluates
``phi_s = arcsin(delta_E / (q V_design))`` during tracking.

The bunch is shifted by one RF period after preparation so it sits inside the
profile window ``[1.5 pi, 4.5 pi] == [0.75, 2.25] t_rf`` (``BiGaussian`` places
it near rf phase ``pi``, one period earlier).  Each test guards that the
profile is actually populated and the bunch is in-window, so an empty profile
fails loudly instead of comparing two ~zero arrays.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Package-relative import: the dirs above ``mucol`` have no __init__.py, so
# these test helpers are not importable by an absolute path under pytest.
from .support import rel_err

DEBUG_PLOT = False


class TestEnergyGainMTWvsNonDrivenFeedback(unittest.TestCase):
    """Applied induced-voltage energy gain: MTW vs non-driven feedback."""

    def setUp(self):
        """Set up RCS1-like parameters and a reproducible template bunch."""
        self.R_over_Q = 518.0
        self.Q_L = 1.29e4
        self.alpha_p = 10.395e-4
        self.circumference = 5990.0
        self.energy = 63e9
        self.harmonic = 25900
        self.intensity = 2.7e12
        self.V_design = 30e6
        self.n_slices = 1024
        self.n_macroparticles = int(5e4)
        # Per-turn reference energy gain for the acceleration case. Must be
        # below q * V_design: the feedback evaluates
        # phi_s = arcsin(delta_E / (q V_design)) during tracking.
        self.delta_E_turn = 6e6

        self.cycle = ConstantMagneticCycle(
            reference_particle=mu_plus,
            value=self.energy,
            in_unit="total energy",
        )
        t_rev = self.cycle.get_t_rev_init(
            self.circumference, particle_type=mu_plus
        )
        self.t_rf = t_rev / self.harmonic
        self.omega_rf = 2 * np.pi / self.t_rf
        self.f_res = 1.0 / self.t_rf

        # Prepare the template bunch once on a nonzero-voltage station, then
        # shift it by one RF period into the profile window.
        sim, _ = self._build(mtw=False, profile=self._make_profile())
        beam = self._prepare(sim)
        self.dt_template = copy_to_cpu(beam.dt.array_local) + self.t_rf
        self.dE_template = copy_to_cpu(beam.dE.array_local)

    def _make_profile(self) -> StaticProfile:
        return StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, self.n_slices, self.t_rf
        )

    def _build(
        self, mtw: bool, profile: StaticProfile, acceleration: bool = False
    ):
        """
        Build a one-turn ring (drift + RF station) and its Simulation.

        Parameters
        ----------
        mtw
            If True, use the multi-turn wake model; otherwise the feedback.
        profile
            Static profile the wake/feedback acts on.
        acceleration
            If True, use a ``MagneticCyclePerTurn`` that raises the reference
            energy by ``delta_E_turn`` over the single turn; otherwise the
            stationary ``ConstantMagneticCycle``.

        Returns
        -------
        simulation
            The constructed one-turn Simulation.
        rf
            The RF station element added to the ring.
        """
        ring = Ring(
            circumference=self.circumference, check_section_indices=False
        )
        drift = DriftSimple(
            orbit_length=self.circumference,
            momentum_compaction_factor=self.alpha_p,
        )
        if mtw:
            wakefield = WakeField(
                sources=(
                    Resonators(
                        shunt_impedances=self.R_over_Q * self.Q_L,
                        center_frequencies=self.f_res,
                        quality_factors=self.Q_L,
                    ),
                ),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12
                ),
                profile=profile,
            )
            rf = SingleHarmonicRFStation(
                voltage=self.V_design,  # nonzero for prepare; zeroed before run
                phi_rf=0.0,
                harmonic=self.harmonic,
                local_wakefield=wakefield,
                profile=profile,
            )
        else:
            # Operating-point cavity: V_init = V_design held steady by the
            # matched generator current I_g = V / (2 (R/Q) Q_L) (steady
            # state of the cavity equation at zero detuning). A cold or
            # undriven cavity trips the coarse-grid beam-kick magnitude
            # check, whose heuristic assumes an established antenna voltage.
            # The beam-induced part is isolated by subtracting a
            # zero-intensity reference run (exact, by linearity).
            matched_generator_current = self.V_design / (
                2.0 * self.R_over_Q * self.Q_L
            )
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=self.R_over_Q,
                Q_L=self.Q_L,
                generator_current_bias=matched_generator_current,
                n_cavities=1,
                initial_voltage=self.V_design,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
            )
            rf = SingleHarmonicRFStation(
                voltage=self.V_design,
                phi_rf=0.0,
                harmonic=self.harmonic,
                cavity_feedback=feedback,
                profile=profile,
            )
        # Drift first so the feedback's RF station is not the first
        # reference-altering element (otherwise the first-turn reverse
        # tracking has nothing to track and returns early).
        ring.add_elements([drift, rf], reorder=False)
        if acceleration:
            magnetic_cycle = MagneticCyclePerTurn(
                reference_particle=mu_plus,
                value_init=self.energy,
                values_after_turn=np.array([self.energy + self.delta_E_turn]),
                in_unit="total energy",
            )
        else:
            magnetic_cycle = self.cycle
        return Simulation(ring=ring, magnetic_cycle=magnetic_cycle), rf

    def _prepare(
        self, sim: Simulation, intensity: float | None = None
    ) -> Beam:
        beam = Beam(
            intensity=self.intensity if intensity is None else intensity,
            particle_type=mu_plus,
        )
        beam.reference.total_energy = self.energy
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macroparticles,
                sigma_dt=0.06 * self.t_rf,
                sigma_dE=1.5e7,
                seed=7,
                reinsertion=True,
            ),
        )
        return beam

    def _run_case(
        self,
        mtw: bool,
        acceleration: bool = False,
        intensity: float | None = None,
    ):
        """
        Run one turn and return the applied kick and post-turn state.

        Parameters
        ----------
        mtw
            If True, use the multi-turn wake model; otherwise the feedback.
        acceleration
            If True, run with the accelerating magnetic cycle.
        intensity
            Beam intensity; defaults to the setUp value. Zero gives the
            no-beam reference run for the feedback path.

        Returns
        -------
        applied
            Energy kick applied to each macroparticle over the turn [eV].
        dt_after
            Arrival times of the macroparticles after the turn.
        profile
            Static profile used for the case.
        rf
            The RF station element used for the case.
        delta_E_reference
            Reference total-energy change over the turn [eV].
        """
        profile = self._make_profile()
        sim, rf = self._build(
            mtw=mtw, profile=profile, acceleration=acceleration
        )
        beam = self._prepare(sim, intensity=intensity)
        beam.setup_beam(dt=self.dt_template, dE=self.dE_template)
        energy_reference_before = float(beam.reference.total_energy)
        dE_before = copy_to_cpu(beam.dE.array_local)
        if mtw:
            rf.voltage = 0.0  # suppress the design RF kick; keep only the wake
        sim.run_simulation((beam,), n_turns=1, show_progressbar=False)
        applied = copy_to_cpu(beam.dE.array_local) - dE_before
        dt_after = copy_to_cpu(beam.dt.array_local)
        delta_E_reference = (
            float(beam.reference.total_energy) - energy_reference_before
        )
        return applied, dt_after, profile, rf, delta_E_reference

    def _run_feedback_induced(self, acceleration: bool = False):
        """
        Beam-induced part of the feedback kick, by reference subtraction.

        Runs the feedback case twice -- with the nominal intensity and with
        zero intensity -- and subtracts the applied kicks per particle. The
        design-voltage reconstruction and (with acceleration) the common
        ``-delta_E_reference`` kick cancel exactly, leaving
        ``q * V_induced(dt)`` by linearity of the cavity equation.

        Parameters
        ----------
        acceleration
            If True, run with the accelerating magnetic cycle.

        Returns
        -------
        applied_induced
            Beam-induced energy kick per macroparticle [eV].
        dt_after
            Arrival times of the macroparticles after the turn.
        profile
            Static profile used for the beam-driven case.
        delta_E_reference
            Reference total-energy change over the turn [eV].
        """
        applied_beam, dt_beam, profile, _, delta_E_reference = self._run_case(
            mtw=False, acceleration=acceleration
        )
        applied_reference, dt_reference, _, _, _ = self._run_case(
            mtw=False, acceleration=acceleration, intensity=0.0
        )
        # Identical initial coordinates and drift -> identical arrival times.
        np.testing.assert_allclose(dt_beam, dt_reference, rtol=0, atol=0)
        return (
            applied_beam - applied_reference,
            dt_beam,
            profile,
            delta_E_reference,
        )

    def _plot_energy_kick(
        self, dt, applied_mtw, applied_fb, profile, induced_mtw
    ):
        """
        Save a debug plot of the applied energy kick vs arrival time.

        Disabled by default so normal/CI runs stay headless. Enable by setting
        the module-level ``support.DEBUG_PLOTS`` constant to ``"save"`` (write a
        PNG) or ``"show"`` (also open an interactive window).

        The plot is generated before the assertions, so it is produced even
        when the comparison fails -- exactly when it is most useful.

        Parameters
        ----------
        dt
            Arrival times of the macroparticles.
        applied_mtw
            Energy kick applied by the multi-turn wake model [eV].
        applied_fb
            Energy kick applied by the non-driven feedback [eV].
        profile
            Static profile the models act on.
        induced_mtw
            Induced voltage of the multi-turn wake model.
        """
        order = np.argsort(dt)
        t_ns = dt[order] * 1e9
        fig, (ax_kick, ax_diff) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6)
        )
        fig.suptitle("Applied energy kick over arrival time")
        ax_kick.plot(
            profile.hist_x * 1e9,
            mu_plus.charge * induced_mtw,
            color="0.6",
            lw=1.0,
            label="charge * V_induced (MTW)",
        )
        ax_kick.scatter(
            t_ns, applied_mtw[order], s=3, color="C0", label="MTW kick"
        )
        ax_kick.scatter(
            t_ns,
            applied_fb[order],
            s=3,
            color="C1",
            marker="x",
            label="non-driven feedback kick",
        )
        ax_kick.set_ylabel("applied dE [eV]")
        ax_kick.legend(loc="best")
        ax_diff.scatter(
            t_ns, (applied_fb - applied_mtw)[order], s=3, color="C3"
        )
        ax_diff.set_ylabel("feedback - MTW [eV]")
        ax_diff.set_xlabel("arrival time dt [ns]")
        fig.tight_layout()

        # Debug-save next to this test file (not the repo root); uncomment and
        # ``import os`` to use:
        # plt.savefig(
        #     os.path.join(os.path.dirname(__file__), "energy_kick_over_time.png"),
        #     dpi=200,
        #     bbox_inches="tight",
        # )

        plt.show()

    def _assert_profile_populated(self, profile, dt_after):
        """
        Guard: the profile is non-empty and the bunch is in-window.

        Parameters
        ----------
        profile
            Static profile whose histogram should be populated.
        dt_after
            Arrival times of the macroparticles after the turn.
        """
        self.assertGreater(
            float(np.sum(profile.hist_y)),
            0.0,
            "profile histogram is empty -- bunch outside the window?",
        )
        hist_x = copy_to_cpu(profile.hist_x)
        in_window = np.mean((dt_after > hist_x[0]) & (dt_after < hist_x[-1]))
        self.assertGreater(in_window, 0.99, "bunch is not inside the window")

    def test_feedback_runs_in_full_simulation(self):
        """
        The feedback tracks through a full Simulation.

        Regression for the stale ``_parent_rf_station._turn_i`` attribute
        (renamed to ``_turn_counter``). Also checks the beam-induced part of
        the kick is small compared to the design voltage.
        """
        applied, dt_after, profile, _ = self._run_feedback_induced()
        self._assert_profile_populated(profile, dt_after)
        peak = np.max(np.abs(applied))
        # Beam-induced only (a few % of the 30 MV design voltage), i.e. the
        # feedback applies the induced voltage and not the full design kick.
        self.assertGreater(peak, 1e4)
        self.assertLess(peak, 0.2 * self.V_design)

    def test_mtw_applies_charge_times_induced_voltage(self):
        """The MTW wake applies exactly ``charge * V_induced(dt)`` per particle."""
        applied, dt_after, profile, rf, _ = self._run_case(mtw=True)
        self._assert_profile_populated(profile, dt_after)

        induced = copy_to_cpu(rf._local_wakefield.induced_voltage)
        expected = mu_plus.charge * np.interp(
            dt_after, copy_to_cpu(profile.hist_x), induced
        )
        peak = np.max(np.abs(applied))
        self.assertGreater(peak, 0.0)
        np.testing.assert_allclose(
            applied, expected, atol=1e-6 * peak, rtol=0.0
        )

    def test_applied_energy_gain_consistent(self):
        """MTW and the feedback apply the same beam-induced energy gain."""
        applied_mtw, dt_mtw, profile_mtw, rf_mtw, _ = self._run_case(mtw=True)
        applied_fb, dt_fb, profile_fb, _ = self._run_feedback_induced()

        # Debug plot (opt-in) before any assertion, so a failure still
        # produces the diagnostic.
        if DEBUG_PLOT:
            self._plot_energy_kick(
                dt_mtw,
                applied_mtw,
                applied_fb,
                profile_mtw,
                copy_to_cpu(rf_mtw._local_wakefield.induced_voltage),
            )

        self._assert_profile_populated(profile_mtw, dt_mtw)
        self._assert_profile_populated(profile_fb, dt_fb)
        # Same initial bunch and identical drift -> identical arrival times.
        np.testing.assert_allclose(dt_mtw, dt_fb, rtol=0, atol=0)

        peak = np.max(np.abs(applied_mtw))
        self.assertGreater(peak, 0.0)

        # Pointwise within ~3 % of the peak induced energy gain.
        np.testing.assert_allclose(
            applied_fb, applied_mtw, atol=0.03 * peak, rtol=0.0
        )
        # Overall agreement well below 2 %.
        self.assertLess(rel_err(applied_fb, applied_mtw), 0.02)

    def test_applied_energy_gain_consistent_with_acceleration(self):
        """
        MTW and the non-driven feedback agree under acceleration too.

        With the accelerating cycle the reference gains ``delta_E_turn`` over
        the turn while (the design kick being suppressed) the particles are
        not accelerated, so each particle receives
        ``applied = q * V_induced(dt) - delta_E_turn``.  The test checks that

        * the reference energy gain matches the programmed ``delta_E_turn``
          in both paths,
        * the MTW path applies exactly
          ``q * V_induced(dt) - delta_E_turn`` per particle, and
        * after removing the common acceleration offset, the induced parts of
          both paths agree to the same tolerances as the stationary case.
        """
        applied_mtw, dt_mtw, profile_mtw, rf_mtw, dE_ref_mtw = self._run_case(
            mtw=True, acceleration=True
        )
        induced_part_fb, dt_fb, profile_fb, dE_ref_fb = (
            self._run_feedback_induced(acceleration=True)
        )

        # The programmed acceleration is actually applied to the reference.
        self.assertAlmostEqual(dE_ref_mtw / self.delta_E_turn, 1.0, delta=1e-9)
        self.assertAlmostEqual(dE_ref_fb / self.delta_E_turn, 1.0, delta=1e-9)

        # Remove the common acceleration offset to recover the MTW induced
        # part (the feedback's reference subtraction already cancelled it).
        induced_part_mtw = applied_mtw + dE_ref_mtw

        # Debug plot (opt-in) before any assertion, so a failure still
        # produces the diagnostic.
        if DEBUG_PLOT:
            self._plot_energy_kick(
                dt_mtw,
                induced_part_mtw,
                induced_part_fb,
                profile_mtw,
                copy_to_cpu(rf_mtw._local_wakefield.induced_voltage),
            )

        self._assert_profile_populated(profile_mtw, dt_mtw)
        self._assert_profile_populated(profile_fb, dt_fb)
        # Same initial bunch and identical drift -> identical arrival times.
        np.testing.assert_allclose(dt_mtw, dt_fb, rtol=0, atol=0)

        # MTW exactness under acceleration:
        # applied = q * V_induced(dt) - delta_E_reference.
        induced = copy_to_cpu(rf_mtw._local_wakefield.induced_voltage)
        expected_mtw = (
            mu_plus.charge
            * np.interp(dt_mtw, copy_to_cpu(profile_mtw.hist_x), induced)
            - dE_ref_mtw
        )
        peak = np.max(np.abs(induced_part_mtw))
        self.assertGreater(peak, 0.0)
        np.testing.assert_allclose(
            applied_mtw, expected_mtw, atol=1e-6 * peak, rtol=0.0
        )

        # Consistency of the induced parts, tolerances as in the stationary
        # case (scaled to the induced peak, not the acceleration kick).
        np.testing.assert_allclose(
            induced_part_fb, induced_part_mtw, atol=0.03 * peak, rtol=0.0
        )
        self.assertLess(rel_err(induced_part_fb, induced_part_mtw), 0.02)


if __name__ == "__main__":
    unittest.main()
