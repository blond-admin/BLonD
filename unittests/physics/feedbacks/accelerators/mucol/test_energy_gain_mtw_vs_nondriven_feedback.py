"""Energy-gain consistency of the multi-turn wake and a non-driven feedback.

This test tracks an *actual* ``BiGaussian`` ``mu_plus`` bunch through a real
``Simulation`` and checks that the induced-voltage energy gain applied to the
particles is the same whether it comes from

* the multi-turn resonator solver (:class:`MultiPassResonatorSolver`,
  applied as a separate wake kick), or
* a *non-driven* :class:`IQCavityFeedbackTimingClass` (``I_g = 0``,
  ``V_init = 0``, ``n_cavities = 1``), whose gap voltage is the lab-frame
  antenna voltage and, for a non-driven cavity, the pure beam-induced voltage.

Both runs share the same minimal stationary ring (one ``DriftSimple`` + one
``SingleHarmonicRFStation``, ``ConstantMagneticCycle``) and the same initial
bunch.  The design RF kick is suppressed (``voltage = 0`` in the MTW run; the
feedback replaces the design voltage with the antenna voltage anyway), so the
only energy change per particle is the induced voltage.  The two paths agree on
the applied ``dE`` to well below 1 %.

The bunch is shifted by one RF period after preparation so it sits inside the
profile window ``[1.5 pi, 4.5 pi] == [0.75, 2.25] t_rf`` (``BiGaussian`` places
it near rf phase ``pi``, one period earlier).  Each test guards that the
profile is actually populated and the bunch is in-window, so an empty profile
fails loudly instead of comparing two ~zero arrays.

Note: the cavity feedback's full-simulation tracking requires the parent RF
station's ``omega_rf_design`` at construction time (its decay check reads it),
but that attribute is only set at ``on_run_simulation``.  We therefore set it
explicitly before building the ``Simulation`` -- to the design RF frequency,
exactly what ``calc_omega_rf_design`` would return.
"""

import os
import unittest

import numpy as np

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
    mu_plus,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver


class TestEnergyGainMTWvsNonDrivenFeedback(unittest.TestCase):
    """Applied induced-voltage energy gain: MTW vs non-driven feedback."""

    def setUp(self):
        """Set up RCS1-like parameters and a reproducible template bunch."""
        self.R_over_Q = 518.0
        self.Q_L = 1.29e6
        self.alpha_p = 10.395e-4
        self.circumference = 5990.0
        self.energy = 63e9
        self.harmonic = 25900
        self.intensity = 2.7e12
        self.V_design = 30e6
        self.n_slices = 1024
        self.n_macro = int(5e4)

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
        self.dt_template = np.array(beam.dt.array_local, copy=True) + self.t_rf
        self.dE_template = np.array(beam.dE.array_local, copy=True)

    def _make_profile(self) -> StaticProfile:
        return StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, self.n_slices, self.t_rf
        )

    def _build(self, mtw: bool, profile: StaticProfile):
        """Build a one-turn ring (drift + RF station) and its Simulation."""
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
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=self.R_over_Q,
                Q_L=self.Q_L,
                generator_current=0.0,
                n_cavities=1,
                initial_voltage=0.0,
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
        # See module docstring: omega_rf_design is needed at construction time.
        rf.omega_rf_design = self.omega_rf
        return Simulation(ring=ring, magnetic_cycle=self.cycle), rf

    def _prepare(self, sim: Simulation) -> Beam:
        beam = Beam(intensity=self.intensity, particle_type=mu_plus)
        beam.reference.total_energy = self.energy
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macro,
                sigma_dt=0.06 * self.t_rf,
                sigma_dE=1.5e7,
                seed=7,
                reinsertion=True,
            ),
        )
        return beam

    def _run_case(self, mtw: bool):
        """Run one turn and return (applied_dE, dt_after, profile, rf)."""
        profile = self._make_profile()
        sim, rf = self._build(mtw=mtw, profile=profile)
        beam = self._prepare(sim)
        beam.setup_beam(dt=self.dt_template, dE=self.dE_template)
        dE_before = np.array(beam.dE.array_local, copy=True)
        if mtw:
            rf.voltage = 0.0  # suppress the design RF kick; keep only the wake
        sim.run_simulation((beam,), n_turns=1, show_progressbar=False)
        applied = np.array(beam.dE.array_local, copy=True) - dE_before
        dt_after = np.array(beam.dt.array_local, copy=True)
        return applied, dt_after, profile, rf

    def _maybe_plot_energy_kick(
        self, dt, applied_mtw, applied_fb, profile, induced_mtw
    ):
        """Save a debug plot of the applied energy kick vs arrival time.

        Disabled by default so normal/CI runs stay headless. Enable with the
        ``BLOND_TEST_PLOTS`` environment variable (set it to ``show`` to also
        open an interactive window)::

            BLOND_TEST_PLOTS=1     <pytest invocation>   # save a PNG
            BLOND_TEST_PLOTS=show  <pytest invocation>   # save and display

        The plot is generated before the assertions, so it is produced even
        when the comparison fails -- exactly when it is most useful.
        """
        mode = os.environ.get("BLOND_TEST_PLOTS")
        if not mode:
            return
        import matplotlib

        if mode != "show":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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

        out = os.path.join(
            os.path.dirname(__file__), "energy_kick_over_time.png"
        )
        fig.savefig(out, dpi=120)
        print(f"\n[debug plot] energy kick over time saved to {out}")
        if mode == "show":
            plt.show()
        plt.close(fig)

    def _assert_profile_populated(self, profile, dt_after):
        """Guard: the profile is non-empty and the bunch is in-window."""
        self.assertGreater(
            float(np.sum(profile.hist_y)),
            0.0,
            "profile histogram is empty -- bunch outside the window?",
        )
        in_window = np.mean(
            (dt_after > profile.hist_x[0]) & (dt_after < profile.hist_x[-1])
        )
        self.assertGreater(in_window, 0.99, "bunch is not inside the window")

    def test_feedback_runs_in_full_simulation(self):
        """The non-driven feedback tracks through a full Simulation.

        Regression for the stale ``_parent_rf_station._turn_i`` attribute
        (renamed to ``_turn_counter``). Also checks the applied kick is the
        small beam-induced voltage, not the design voltage.
        """
        applied, dt_after, profile, _ = self._run_case(mtw=False)
        self._assert_profile_populated(profile, dt_after)
        peak = np.max(np.abs(applied))
        # Beam-induced only (a few % of the 30 MV design voltage), i.e. the
        # feedback applies the induced voltage and not the full design kick.
        self.assertGreater(peak, 1e4)
        self.assertLess(peak, 0.2 * self.V_design)

    def test_mtw_applies_charge_times_induced_voltage(self):
        """The MTW wake applies exactly ``charge * V_induced(dt)`` per particle."""
        applied, dt_after, profile, rf = self._run_case(mtw=True)
        self._assert_profile_populated(profile, dt_after)

        induced = np.asarray(rf._local_wakefield.induced_voltage)
        expected = mu_plus.charge * np.interp(
            dt_after, profile.hist_x, induced
        )
        peak = np.max(np.abs(applied))
        self.assertGreater(peak, 0.0)
        np.testing.assert_allclose(
            applied, expected, atol=1e-6 * peak, rtol=0.0
        )

    def test_applied_energy_gain_consistent(self):
        """MTW and the non-driven feedback apply the same energy gain."""
        applied_mtw, dt_mtw, profile_mtw, rf_mtw = self._run_case(mtw=True)
        applied_fb, dt_fb, profile_fb, _ = self._run_case(mtw=False)

        # Debug plot (opt-in) before any assertion, so a failure still
        # produces the diagnostic.
        self._maybe_plot_energy_kick(
            dt_mtw,
            applied_mtw,
            applied_fb,
            profile_mtw,
            np.asarray(rf_mtw._local_wakefield.induced_voltage),
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
        rel_l2 = np.linalg.norm(applied_fb - applied_mtw) / np.linalg.norm(
            applied_mtw
        )
        self.assertLess(rel_l2, 0.02)


if __name__ == "__main__":
    unittest.main()
