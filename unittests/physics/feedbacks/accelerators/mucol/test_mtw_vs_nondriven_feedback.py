"""
Compare the multi-pass resonator solver with non-driven cavity feedback.

Two test classes, by driver and integration depth (the applied-*energy*
comparison lives in
``test_energy_gain_ind_voltage_vs_nondriven_feedback.py``):

* :class:`TestSinglePassInducedVoltage` -- single pass, mock-driven, no
  ``Simulation``. Solver vs feedback induced voltage on one static profile.
* :class:`TestMultiTurnFeedbackVsConvolution` -- full ``Simulation``, a dummy
  particle-less beam, turn-over-turn coarse-grid propagation, multiple
  sections and acceleration.

Both compare the *same* single cavity
(``R_shunt = R_over_Q * Q_L``, ``f_res = 1 / t_rf``):

* a :class:`MultiPassResonatorSolver` -- the multi-turn resonator convolution,
  and
* an :class:`IQCavityFeedbackTimingClass` whose antenna voltage, with the beam
  as the only excitation, is the beam-induced voltage.

In the single-pass class both objects are driven directly on the static
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
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
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


def make_noisy_profile(
    t_rf: float,
    n_slices: int,
    section_index: int = 0,
    seed: int = 12345,
) -> StaticProfile:
    """
    Build the noisy-Gaussian static profile used by both test classes.

    Parameters
    ----------
    t_rf
        RF period defining the profile window (1.5 to 4.5 pi in rad).
    n_slices
        Number of profile bins.
    section_index
        Section the profile belongs to (for multi-section rings). Also
        offsets the noise seed so each section gets a distinct profile.
    seed
        Base RNG seed for the additive noise.

    Returns
    -------
    StaticProfile
        Noisy Gaussian bunch with zeroed leading/trailing bins.
    """
    profile = StaticProfile.from_rad(
        np.pi * 1.5,
        np.pi * 4.5,
        n_slices,
        t_rf,
        section_index=section_index,
    )
    t = profile.hist_x
    t0 = 0.5 * (t[0] + t[-1])
    sigma = 0.08 * t_rf

    rng = np.random.default_rng(seed + section_index)
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


class TestSinglePassInducedVoltage(unittest.TestCase):
    """
    Single-pass induced voltage: multi-pass solver vs non-driven feedback.

    Drives the solver and the feedback directly on one static profile -- no
    Beam tracking and no Simulation -- and checks the lab-frame induced
    voltage they produce agrees to < 1 %, in the mock/patch style of
    ``test_mucol_cav_fdbk.py``.
    """

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
        self.noisy_profile = make_noisy_profile(self.t_rf, self.n_slices)

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
            generator_current_bias=0.0,  # non-driven: no generator current
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

        # Debug-save next to this test file (not the repo root); needs ``import os``:
        # plt.savefig(
        #     os.path.join(os.path.dirname(__file__), "induced_voltage_over_time.png"), dpi=400
        # )
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
            generator_current_bias=0.0,
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


class TestMultiTurnFeedbackVsConvolution(unittest.TestCase):
    """
    Full-Simulation multi-turn comparison: feedback vs multi-pass convolution.

    A dummy beam without macroparticles drives the static noisy profiles over
    several turns (``profile.active = False`` so the empty beam never
    overwrites the histogram). The feedback's coarse grid is propagated turn
    over turn through the reverse/forward reference tracking, and its
    beam-induced gap voltage is compared per turn (and per section) against
    the accumulating multi-pass convolution voltage. Covers single/multiple
    sections and a static or accelerating cycle.

    This is its own class (not the single-pass one above): it ignores that
    fixture entirely, needs a high Q_L so the previous-pass wake survives
    (~88 % per turn via exp(-omega * t_rev / Q_L); the single-pass Q_L of
    1.29e4 would leave only ~6e-5), and drives a real Simulation rather than
    direct method calls.
    """

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
    # Per-RF-station reference energy gain for the acceleration cases. Must
    # stay below the design voltage (the feedback evaluates phi_s).
    MULTITURN_DELTA_E_SECTION = 2e6

    # Fast frame-slip regime (same machine point as
    # test_feedback_phase_under_acceleration): just above transition
    # (gamma_t ~ 31 at ~4 GeV) the RF frame slips ~0.09 t_rf per turn --
    # orders of magnitude more than at the 63 GeV operating point -- so
    # per-segment phase errors that hide at the slow ramp become visible.
    FAST_ENERGY = 4.0e9
    FAST_DELTA_E_TURN = 20e6  # per turn, split evenly across the stations
    FAST_N_TURNS = 5

    # Cache keyed on (n_sections, acceleration, n_rf_periods, fast_ramp):
    # each config runs three full simulations (convolution, beam feedback,
    # no-beam reference), so this avoids re-running shared configurations
    # across tests.
    _multiturn_cache: dict = {}

    @classmethod
    def _regime(cls, fast_ramp: bool):
        """
        Energy, per-station energy gain and turn count of a ramp regime.

        Parameters
        ----------
        fast_ramp
            If True, the transition-adjacent fast frame-slip regime;
            otherwise the 63 GeV operating point.

        Returns
        -------
        energy
            Initial reference total energy [eV].
        n_turns
            Number of turns to simulate.
        """
        if fast_ramp:
            return cls.FAST_ENERGY, cls.FAST_N_TURNS
        return cls.MULTITURN_ENERGY, cls.MULTITURN_N_TURNS

    @classmethod
    def _calc_multiturn_harmonic_and_t_rf(
        cls, n_sections: int, fast_ramp: bool = False
    ):
        """
        Harmonic (divisible by ``2 * n_sections``) and the matching t_rf.

        Parameters
        ----------
        n_sections
            Number of RF stations per turn.
        fast_ramp
            If True, evaluate at the fast-regime injection energy.

        Returns
        -------
        harmonic
            Harmonic reduced to an integer multiple of ``2 * n_sections``
            (so each half-drift spans a whole number of RF periods).
        t_rf
            RF period for that harmonic at the cycle's initial energy.
        """
        energy, _ = cls._regime(fast_ramp)
        harmonic = int(
            cls.MULTITURN_HARMONIC - cls.MULTITURN_HARMONIC % (2 * n_sections)
        )
        cycle = ConstantMagneticCycle(
            reference_particle=mu_plus,
            value=energy,
            in_unit="total energy",
        )
        t_rev = cycle.get_t_rev_init(
            cls.MULTITURN_CIRCUMFERENCE, particle_type=mu_plus
        )
        return harmonic, t_rev / harmonic

    @classmethod
    def _multiturn_cycle(
        cls, n_sections: int, acceleration: bool, fast_ramp: bool = False
    ):
        """
        Magnetic cycle for the multi-turn run: static or accelerating.

        Parameters
        ----------
        n_sections
            Number of RF stations per turn.
        acceleration
            If True, a ``MagneticCyclePerTurnAllRFStations`` that raises the
            reference energy at every RF station; otherwise a stationary
            ``ConstantMagneticCycle``.
        fast_ramp
            If True, use the transition-adjacent fast regime (implies
            ``acceleration``): the per-turn gain ``FAST_DELTA_E_TURN`` is
            split evenly across the stations.

        Returns
        -------
        MagneticCycleBase
            The magnetic cycle.
        """
        energy, n_turns = cls._regime(fast_ramp)
        if fast_ramp:
            acceleration = True
        if not acceleration:
            return ConstantMagneticCycle(
                reference_particle=mu_plus,
                value=energy,
                in_unit="total energy",
            )
        delta_e_section = (
            cls.FAST_DELTA_E_TURN / n_sections
            if fast_ramp
            else cls.MULTITURN_DELTA_E_SECTION
        )
        n_kicks = n_sections * n_turns
        values = (
            energy + delta_e_section * np.arange(1, n_kicks + 1)
        ).reshape(n_sections, n_turns, order="F")
        return MagneticCyclePerTurnAllRFStations(
            reference_particle=mu_plus,
            value_init=energy,
            values_after_rf_station_per_turn=values,
            in_unit="total energy",
        )

    @classmethod
    def _run_multiturn_case(
        cls,
        mode: str,
        n_sections: int,
        acceleration: bool,
        n_rf_periods: float = 1,
        fast_ramp: bool = False,
    ) -> list:
        """
        Run a full multi-turn Simulation and collect a voltage per turn.

        A dummy beam without macroparticles drives nothing physically; the
        static noisy profiles (``profile.active = False`` so the empty beam
        never overwrites the histogram) are the only excitation. The beam's
        reference still advances each turn, which is what propagates both the
        convolution's past-wake times and the feedback's coarse grid -- the
        latter through the reverse/forward tracking across all sections.

        The ring follows the production layout: per section a half-drift, the
        RF station (with its own profile and wake/feedback), and another
        half-drift.

        Parameters
        ----------
        mode
            ``"mtw"`` (convolution wakefield), ``"fb"`` (feedback with beam
            current) or ``"fb_reference"`` (feedback with zero intensity, to
            isolate the beam-induced part by linearity).
        n_sections
            Number of RF stations per turn.
        acceleration
            If True, run with the accelerating cycle.
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback; values below 1
            are the sub-stepping mode.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime
            (implies acceleration, ``FAST_N_TURNS`` turns). The convolution
            reference then uses the retuning solver (``delta_f = 0.0``), the
            counterpart of the feedback's always-on-resonance cavity.

        Returns
        -------
        list
            Per turn, a list (one entry per section) of voltage arrays: the
            wakefield induced voltage for ``"mtw"``, the station gap voltage
            otherwise.
        """
        harmonic, t_rf = cls._calc_multiturn_harmonic_and_t_rf(
            n_sections, fast_ramp=fast_ramp
        )
        energy, n_turns = cls._regime(fast_ramp)
        half_drift_length = cls.MULTITURN_CIRCUMFERENCE / n_sections / 2

        ring = Ring(
            circumference=cls.MULTITURN_CIRCUMFERENCE,
            check_section_indices=False,
        )
        simulation_elements = []
        ind_volt_elements = []  # wakefield (mtw) or RF station (feedback)
        for section_index in range(n_sections):
            profile = make_noisy_profile(
                t_rf, cls.MULTITURN_N_SLICES, section_index=section_index
            )
            profile.active = False  # keep the histogram static (no particles)

            if mode == "mtw":
                solver_kwargs = {"decay_fraction_threshold": 1e-12}
                if fast_ramp:
                    # Retuning solver: the resonator follows the RF frequency
                    # turn by turn (delta_f = 0), matching the feedback's
                    # cavity, which is always on resonance with the current
                    # RF. At the slow ramp the distinction is negligible.
                    solver_kwargs["delta_f"] = 0.0
                local_wf = WakeField(
                    sources=(
                        Resonators(
                            cls.MULTITURN_R_OVER_Q * cls.MULTITURN_Q_L,
                            1.0 / t_rf,
                            cls.MULTITURN_Q_L,
                        ),
                    ),
                    solver=MultiPassResonatorSolver(**solver_kwargs),
                    profile=profile,
                )
                rf_station = SingleHarmonicRFStation(
                    voltage=cls.MULTITURN_V_DESIGN,
                    phi_rf=0.0,
                    harmonic=harmonic,
                    local_wakefield=local_wf,
                    profile=profile,
                    section_index=section_index,
                )
                ind_volt_elements.append(local_wf)
            else:
                # Operating-point cavity (V_init = V_design): a cold start
                # (V_init = 0) trips the coarse-grid beam-kick magnitude
                # check, whose heuristic assumes an established voltage.
                feedback = IQCavityFeedbackTimingClass(
                    profile=profile,
                    R_over_Q=cls.MULTITURN_R_OVER_Q,
                    Q_L=cls.MULTITURN_Q_L,
                    generator_current_bias=0.0,
                    n_cavities=1,
                    initial_voltage=cls.MULTITURN_V_DESIGN,
                    n_rf_periods_per_coarse_grid=n_rf_periods,
                    delta_omega=0.0,
                )
                rf_station = SingleHarmonicRFStation(
                    voltage=cls.MULTITURN_V_DESIGN,
                    phi_rf=0.0,
                    harmonic=harmonic,
                    cavity_feedback=feedback,
                    profile=profile,
                    section_index=section_index,
                )
                ind_volt_elements.append(rf_station)
            simulation_elements += [
                DriftSimple(
                    orbit_length=half_drift_length,
                    momentum_compaction_factor=cls.MULTITURN_ALPHA_P,
                    section_index=section_index,
                ),
                rf_station,
                DriftSimple(
                    orbit_length=half_drift_length,
                    momentum_compaction_factor=cls.MULTITURN_ALPHA_P,
                    section_index=section_index,
                ),
            ]
        ring.add_elements(simulation_elements, reorder=False)
        sim = Simulation(
            ring=ring,
            magnetic_cycle=cls._multiturn_cycle(
                n_sections, acceleration, fast_ramp=fast_ramp
            ),
        )

        beam = Beam(
            intensity=(
                0.0 if mode == "fb_reference" else cls.MULTITURN_INTENSITY
            ),
            particle_type=mu_plus,
        )
        beam.reference.total_energy = energy
        beam.setup_beam(dt=np.array([]), dE=np.array([]))

        per_turn = []

        def collect(simulation, beam_in_callback):
            if mode == "mtw":
                per_turn.append(
                    [
                        np.copy(np.asarray(element.induced_voltage))
                        for element in ind_volt_elements
                    ]
                )
            else:
                per_turn.append(
                    [
                        np.copy(
                            np.asarray(
                                station.calc_gap_voltage_with_feedbacks()
                            )
                        )
                        for station in ind_volt_elements
                    ]
                )

        sim.run_simulation(
            (beam,),
            n_turns=n_turns,
            callbacks=collect,
            show_progressbar=False,
        )
        return per_turn

    @classmethod
    def _feedback_vs_convolution(
        cls,
        n_sections: int,
        acceleration: bool,
        n_rf_periods: float = 1,
        fast_ramp: bool = False,
    ):
        """
        Run (once per config) and cache the convolution and feedback voltages.

        Parameters
        ----------
        n_sections
            Number of RF stations per turn.
        acceleration
            If True, run with the accelerating cycle.
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime.

        Returns
        -------
        v_convolution_turns
            ``[turn][section]`` induced voltage of the multi-pass convolution.
        v_feedback_turns
            ``[turn][section]`` beam-induced voltage of the feedback (the
            beam run's gap voltage minus the no-beam reference run, which
            isolates the beam-induced part by linearity of the cavity
            equation).
        """
        key = (n_sections, acceleration, n_rf_periods, fast_ramp)
        if key not in cls._multiturn_cache:
            convolution = cls._run_multiturn_case(
                "mtw", n_sections, acceleration, n_rf_periods, fast_ramp
            )
            gap_beam = cls._run_multiturn_case(
                "fb", n_sections, acceleration, n_rf_periods, fast_ramp
            )
            gap_reference = cls._run_multiturn_case(
                "fb_reference",
                n_sections,
                acceleration,
                n_rf_periods,
                fast_ramp,
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

    def _assert_multiturn_consistency(
        self,
        n_sections: int,
        acceleration: bool,
        n_rf_periods: float = 1,
        fast_ramp: bool = False,
    ):
        """
        Assert per-section, per-turn feedback/convolution agreement.

        Parameters
        ----------
        n_sections
            Number of RF stations per turn.
        acceleration
            If True, run with the accelerating cycle.
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.
        fast_ramp
            If True, run the transition-adjacent fast frame-slip regime.
        """
        convolution, feedback = self._feedback_vs_convolution(
            n_sections, acceleration, n_rf_periods, fast_ramp
        )

        if DEBUG_PLOT:
            # Plot the first section only.
            self._plot_multiturn(
                [turn[0] for turn in convolution],
                [turn[0] for turn in feedback],
            )

        for turn_i, (convolution_turn, feedback_turn) in enumerate(
            zip(convolution, feedback, strict=True)
        ):
            for section_i, (v_convolution, v_feedback) in enumerate(
                zip(convolution_turn, feedback_turn, strict=True)
            ):
                self.assertLess(
                    rel_err(v_feedback, v_convolution),
                    0.02,
                    f"turn {turn_i} section {section_i}",
                )

    def test_multiturn_wake_accumulates_over_turns(self):
        """
        The multi-pass wake genuinely builds up turn over turn.

        With ``MULTITURN_Q_L = 1.29e6`` the previous-pass wake survives
        ~88 % per turn and the in-phase buildup follows the geometric sum
        (measured peaks ~1.0, 1.9, 2.8). The first turn (no previous pass
        yet) must also agree with the feedback to the single-pass accuracy.
        """
        # Single section, static cycle.
        convolution, feedback = self._feedback_vs_convolution(
            n_sections=1, acceleration=False
        )
        v_convolution_turns = [turn[0] for turn in convolution]
        v_feedback_turns = [turn[0] for turn in feedback]

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
        # Single section, static cycle.
        self._assert_multiturn_consistency(n_sections=1, acceleration=False)

    def test_multiturn_multiple_sections(self):
        """
        Feedback vs convolution holds for multi-section rings.

        Exercises the feedback's reverse/forward reference tracking across
        several RF stations per turn (where the parent station is no longer
        the only reference-altering element) -- the code path that production
        runs use and that was broken until the ``_turn_counter`` fix.
        """
        for n_sections in (2, 3, 10):
            with self.subTest(n_sections=n_sections):
                self._assert_multiturn_consistency(
                    n_sections=n_sections, acceleration=False
                )

    def test_multiturn_with_acceleration(self):
        """
        Feedback vs convolution holds under acceleration.

        The reference energy is raised at every RF station each turn
        (``MagneticCyclePerTurnAllRFStations``), so t_rev, the carrier
        frequency and the reverse-tracking frame slip all vary turn over
        turn. The beam-induced parts (isolated by the no-beam reference
        subtraction, which cancels the common acceleration kick) must still
        track the multi-pass convolution, with one and with several sections.
        """
        for n_sections in (1, 2, 10):
            with self.subTest(n_sections=n_sections):
                self._assert_multiturn_consistency(
                    n_sections=n_sections, acceleration=True
                )

    def test_multiturn_substepped_matches_convolution(self):
        """
        Beam loading computed on a sub-stepped coarse grid stays correct.

        With ``n_rf_periods_per_coarse_grid = 0.5`` the coarse grid halves
        the RF period, so the bunch spans several coarse cells and the
        beam-current downsampling distributes charge across cell edges --
        a regime no other physics test exercises. The carried wake must
        still match the multi-pass convolution at the same tolerance as
        the standard grid.
        """
        self._assert_multiturn_consistency(
            n_sections=1, acceleration=False, n_rf_periods=0.5
        )

    def test_multiturn_fast_ramp(self):
        """
        Feedback vs retuning convolution in the fast frame-slip regime.

        Transition-adjacent energy (~4 GeV, gamma_t ~ 31): the RF frame
        slips ~0.09 t_rf per turn -- orders of magnitude more than at the
        63 GeV operating point of ``test_multiturn_with_acceleration`` --
        over 5 turns, single section. The convolution reference retunes
        with the RF (``delta_f = 0``), matching the feedback's
        always-on-resonance cavity. The carried wake agrees to ~0.1 %.
        """
        self._assert_multiturn_consistency(
            n_sections=1,
            acceleration=True,
            fast_ramp=True,
        )

    def test_multiturn_fast_ramp_multisection(self):
        """
        Multi-section carried wake holds under the fast ramp.

        On the transition-adjacent fast ramp the coarse grid is rebuilt each
        turn from reverse segments across the *other* stations, each re-seeded
        at its own past-station frequency. The multi-section frame correction
        (``_track``) removes the resulting carried-envelope phase error
        ``sum_k (omega_k - omega_0) T_seg,k`` before it seeds the forward
        segment, so the carried wake matches the retuning convolution on every
        turn for 2 and 4 sections. Without the correction the arrival time
        drifted ~0.023 t_rf per turn (turn 4 reached ~29 % / ~57 % rel. error
        for 2 / 4 sections); with it the error stays ~0.2 %.
        """
        for n_sections in (2, 4):
            with self.subTest(n_sections=n_sections):
                self._assert_multiturn_consistency(
                    n_sections=n_sections,
                    acceleration=True,
                    fast_ramp=True,
                )

    def test_multiturn_fast_ramp_substepped(self):
        """
        Sub-stepped carried wake holds under the fast ramp.

        A sub-stepped grid (n = 0.5) on the fast (transition-adjacent) ramp,
        single section. Two former defects are fixed: (1) the stale
        reverse-segment re-pass (the turn-0 reverse omega list was never
        refreshed for a single section, so every turn re-ran the forward
        grid at the frozen injection frequency, corrupting the demodulation
        frame by -(turn+1) * 2 pi S per turn and stepping any attached
        controller on garbage); (2) the sub-stepped demodulation frame is
        now the tiling boundary gap (first-centre offset + carried
        residual, a pure time immune to the float-bistable residual landing
        flip) instead of the mod-2*pi grid-phase reconstruction. Carried
        turns agree with the retuning convolution to ~0.1 % (previously
        ~40 % with a per-kick-turn constant phase error).
        """
        self._assert_multiturn_consistency(
            n_sections=1,
            acceleration=True,
            n_rf_periods=0.5,
            fast_ramp=True,
        )

    def test_multiturn_fast_ramp_multisection_substepped(self):
        """
        The full combination holds: multi-section, fast ramp, sub-stepping.

        Two RF stations, the transition-adjacent fast ramp and a sub-stepped
        grid (n = 0.5). Combines the multi-section frame correction
        (``test_multiturn_fast_ramp_multisection``) with the sub-stepped
        demodulation-frame fix (``test_multiturn_fast_ramp_substepped``,
        whose tiling-gap formula also covers the multi-section
        reverse-to-forward handover -- the former turn-0 section-1 sign
        flip was the same demodulation-frame defect). Exercises the
        reverse-tracking residual carry-over across segments of different
        frequency with actual beam-loading physics against the retuning
        convolution.
        """
        self._assert_multiturn_consistency(
            n_sections=2,
            acceleration=True,
            n_rf_periods=0.5,
            fast_ramp=True,
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
        # plt.savefig(os.path.join(os.path.dirname(__file__), "multiturn_induced_voltage.png"), dpi=400)
        plt.show()


if __name__ == "__main__":
    unittest.main()
