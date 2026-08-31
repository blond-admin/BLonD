"""
PI-controlled cavity feedback inside a real tracked ``Simulation``.

Every other PI test drives the controller on hand-built constant-step
grids (see ``test_generator_current_pi_feedback.py``); here the full chain
runs in anger: a matched ``BiGaussian`` ``mu_plus`` bunch with strong beam
loading is tracked through a real ring with the backfill/forward reference
tracking, under strong acceleration, with the
:class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController`
regulating the generator current -- single- and multi-section, on both the
operating-point (slow) ramp and the transition-adjacent fast ramp.

Each configuration asserts physical behaviour (the loop acts, the voltage
is held near the setpoint, the reference follows the energy program) and
then pins the end-of-turn antenna voltage and generator current
trajectories against hardcoded reference values (characterization test:
any change of the tracked feedback numerics shows up here first).
"""

import os
import unittest
import warnings

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_plus,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)

# Print the recorded trajectories instead of asserting the pins (used once
# to generate / regenerate the hardcoded reference values below).
PRINT_PINS = os.environ.get("PI_TRACKING_PRINT_PINS", "") != ""

R_OVER_Q = 518.0
Q_L = 1.29e6
V_DESIGN = 30.0e6
HARMONIC = 25900
CIRCUMFERENCE = 5990.0
ALPHA_P = 10.395e-4
INTENSITY = 2.7e12
N_SLICES = 1024
N_MACROPARTICLES = 50_000
SEED = 7

# Matched-generator bias: the no-beam steady state of the cavity.
I_GEN_BIAS = V_DESIGN / (2.0 * R_OVER_Q * Q_L)
# Loop tuning as in the coarse-transient unit tests: per-step proportional
# loop gain ~0.1, integral loop ~30 RF periods slower, 2 samples delay.
GAIN_P = 0.1 / (R_OVER_Q * 2.0 * np.pi)
N_DELAY = 2


def _run_config(
    n_sections: int,
    energy: float,
    delta_e_turn: float,
    n_turns: int,
    intensity: float = INTENSITY,
    controller_call_counter: dict | None = None,
    use_controller: bool = True,
    detuning_half_bandwidths: float = 0.0,
    delta_omega_rf: float = 0.0,
) -> dict:
    """
    Track a matched bunch with PI-regulated feedbacks on every station.

    Parameters
    ----------
    n_sections
        Number of RF stations (half-drift / station / half-drift each).
    energy
        Initial reference total energy [eV].
    delta_e_turn
        Reference energy gain per turn [eV], split across the stations.
    n_turns
        Number of turns to track.
    intensity
        Beam intensity; ``0`` tracks an empty beam (no macroparticles), used
        by the structural backfill-span tests.
    controller_call_counter
        If given, a ``{"count": 0}`` dict; every controller update increments
        ``"count"`` so tests can compare controller steps against the
        recorded forward/total coarse-cell counts.
    use_controller
        If False, attach no PI controller, so the generator current stays at
        the constant ``generator_current_bias``. Used by the driven
        steady-state tests, which need the open-loop cavity response.
    detuning_half_bandwidths
        Cavity resonance detuning ``delta_omega`` in units of the cavity
        half-bandwidth ``omega_rf / (2 Q_L)``, so that this number *is*
        ``tan(psi)``. ``0`` (the default) keeps every existing call site on
        resonance and bit-unchanged.
    delta_omega_rf
        Station RF-frequency offset [rad/s], set on every station before
        the run (from turn 0). ``0`` (the default) keeps every existing
        call site bit-unchanged.

    Returns
    -------
    dict
        Per-turn trajectories per station: ``v_min`` (minimum antenna
        voltage magnitude over the forward segment -- the beam-loading sag),
        ``v_last`` (last coarse sample -- the recovered voltage),
        ``i_max_dev`` (maximum generator-current deviation from the bias --
        the loop response), ``v_dev_grid`` (worst relative antenna-voltage
        deviation from ``V_DESIGN`` over the *whole* coarse grid of the
        turn, backfill reconstruction span included),
        ``n_forward``/``n_total`` (forward and total coarse cells per turn);
        plus ``ref_energy`` and ``sigma_dt``.
    """
    from blond import ConstantMagneticCycle

    cycle_probe = ConstantMagneticCycle(
        reference_particle=mu_plus, value=energy, in_unit="total energy"
    )
    t_rev = cycle_probe.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    harmonic = int(HARMONIC - HARMONIC % (2 * n_sections))
    t_rf = t_rev / harmonic

    # Cavity resonance detuning in units of the cavity half-bandwidth
    # omega_rf / (2 Q_L), so tan(psi) == detuning_half_bandwidths.
    omega_rf = 2.0 * np.pi / t_rf
    delta_omega = detuning_half_bandwidths * omega_rf / (2.0 * Q_L)

    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    half_drift = CIRCUMFERENCE / n_sections / 2
    stations = []
    feedbacks = []
    elements = []
    for section_index in range(n_sections):
        profile = StaticProfile.from_rad(
            np.pi * 1.5,
            np.pi * 4.5,
            N_SLICES,
            t_rf,
            section_index=section_index,
        )
        controller = GeneratorCurrentPIController(
            gain_proportional=GAIN_P,
            gain_integral=GAIN_P / (30.0 * t_rf),
            generator_current_bias=I_GEN_BIAS + 0.0j,
            n_delay=N_DELAY,
        )
        if controller_call_counter is not None:
            _orig_update = controller.update_generator_current

            def _counting_update(error, delta_t, _o=_orig_update):
                controller_call_counter["count"] += 1
                return _o(error, delta_t)

            controller.update_generator_current = _counting_update
        if not use_controller:
            controller = None
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            generator_current_bias=I_GEN_BIAS + 0.0j,
            n_cavities=1,
            initial_voltage=V_DESIGN,
            n_rf_periods_per_coarse_grid=1,
            delta_omega=delta_omega,
            controller=controller,
            voltage_setpoint=V_DESIGN + 0.0j,
        )
        if controller_call_counter is not None:
            # These structural tests count per-cell
            # ``controller.update_generator_current`` calls to pin "the PI is
            # stepped on forward cells only". That call structure is specific
            # to the pure-Python reference path; the numba envelope kernel
            # inlines the PI (never calling the controller method), so drive the
            # reference path here. The kernel's equivalent forward-only stepping
            # is pinned instead by the byte-identical coarse grids in
            # test_envelope_kernel (a kernel stepping the PI on the backfill
            # segments would diverge there).
            feedback.use_numba_envelope_kernel = False
        station = SingleHarmonicRFStation(
            voltage=V_DESIGN,
            phi_rf=0.0,
            harmonic=harmonic,
            cavity_feedback=feedback,
            profile=profile,
            section_index=section_index,
        )
        if delta_omega_rf != 0.0:
            # Pre-run configuration; the post-init setter warning is by
            # design and irrelevant here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                station.delta_omega_rf = delta_omega_rf
        stations.append(station)
        feedbacks.append(feedback)
        elements += [
            DriftSimple(
                orbit_length=half_drift,
                momentum_compaction_factor=ALPHA_P,
                section_index=section_index,
            ),
            station,
            DriftSimple(
                orbit_length=half_drift,
                momentum_compaction_factor=ALPHA_P,
                section_index=section_index,
            ),
        ]
    ring.add_elements(elements, reorder=False)

    delta_e_section = delta_e_turn / n_sections
    values = (
        energy + delta_e_section * np.arange(1, n_sections * n_turns + 1)
    ).reshape(n_sections, n_turns, order="F")
    cycle = MagneticCyclePerTurnAllRFStations(
        reference_particle=mu_plus,
        value_init=energy,
        values_after_rf_station_per_turn=values,
        in_unit="total energy",
    )
    sim = Simulation(ring=ring, magnetic_cycle=cycle)

    beam = Beam(intensity=intensity, particle_type=mu_plus)
    beam.reference.total_energy = energy
    if intensity > 0:
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=N_MACROPARTICLES,
                sigma_dt=0.06 * t_rf,
                sigma_dE=None,
                seed=SEED,
                reinsertion=True,
            ),
        )
        # Shift the bunch one RF period into the profile window (the window
        # starts at 0.75 t_rf; the matched bunch is created around dt ~ 0).
        beam._dt.array_local += t_rf
    else:
        # Empty beam: no beam loading, so a matched-bias PI loop should sit
        # at its no-beam steady state (V = V_ss, I_gen = bias) every turn.
        beam.setup_beam(dt=np.array([]), dE=np.array([]))

    rec = {
        "v_min": [],
        "v_last": [],
        "i_max_dev": [],
        "v_dev_grid": [],
        "phi_corr": [],
        "delta_phi_rf": [],
        "ref_energy": [],
        "sigma_dt": [],
        "n_forward": [],
        "n_total": [],
    }

    def callback(_sim, b):
        rec["n_forward"].append(
            [int(f._rf_centers_lengths[-1]) for f in feedbacks]
        )
        rec["n_total"].append([int(len(f._rf_centers)) for f in feedbacks])
        # Only the forward segment of this turn (the last
        # rf_centers_lengths[-1] samples) -- the backfill part repeats the
        # previous turn's no-beam propagation.
        rec["v_min"].append(
            [
                float(
                    np.abs(
                        f.antenna_voltage_coarse_grid[
                            -int(f._rf_centers_lengths[-1]) :
                        ]
                    ).min()
                )
                for f in feedbacks
            ]
        )
        rec["v_last"].append(
            [
                float(np.abs(f.antenna_voltage_coarse_grid[-1]))
                for f in feedbacks
            ]
        )
        rec["i_max_dev"].append(
            [
                float(
                    np.abs(
                        f.generator_current_coarse_grid[
                            -int(f._rf_centers_lengths[-1]) :
                        ]
                        - I_GEN_BIAS
                    ).max()
                )
                for f in feedbacks
            ]
        )
        # Worst antenna-voltage deviation over the WHOLE coarse grid of the
        # turn -- backfill reconstruction span included. With no beam and a
        # regulated loop the correct value is the setpoint on every sample.
        rec["v_dev_grid"].append(
            [
                float(
                    np.abs(f.antenna_voltage_coarse_grid - V_DESIGN).max()
                    / V_DESIGN
                )
                for f in feedbacks
            ]
        )
        # Rigid RF phase the feedback hands the station this turn (the
        # readout is flat over the window here, so the mean is that
        # constant), and the station kick clock it was applied against.
        rec["phi_corr"].append(
            [float(np.mean(f.phase_correction)) for f in feedbacks]
        )
        rec["delta_phi_rf"].append([float(s.delta_phi_rf) for s in stations])
        rec["ref_energy"].append(float(b.reference.total_energy))
        rec["sigma_dt"].append(float(np.std(copy_to_cpu(b.dt.array_local))))

    sim.run_simulation(
        (beam,), n_turns=n_turns, callbacks=callback, show_progressbar=False
    )
    for key, values in rec.items():
        rec[key] = np.array(values)
    return rec


class TestPIBackfillSpanFrameConsistency(unittest.TestCase):
    """
    The PI loop must not act on the backfill reconstruction segments.

    A multi-section feedback rebuilds the previous turn each turn as
    ``no_beam`` backfill segments before the forward pass. The controller
    must be stepped only on the forward (real, current-turn) segment: the
    backfill cells carry a per-segment frame phase (corrected only on the
    last sample), so a controller stepped there under a fast ramp
    integrates frame-rotated errors and injects spurious quadrature
    current.

    Isolation: rather than measuring the (small) frame drift, these tests
    pin the fix *structurally* by counting controller updates. A
    frame-consistent loop must step the controller once per forward
    (real-passage) coarse cell and never on a backfill reconstruction cell,
    so ``controller-call count == sum(n_forward)`` while ``n_total`` is
    strictly larger (the backfill cells exist but must be skipped). This is
    independent of beam loading, so the tests run with a full-intensity
    beam and assert the call count, not a voltage trajectory.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 3

    def test_controller_stepped_only_on_forward_cells(self):
        """
        Two-section fast ramp: controller calls == forward cells, not total.

        With the bug the controller is stepped on every coarse cell,
        including the backfill reconstruction segments (n_total per station),
        double-advancing its delay line / integrator on frame-rotated
        errors; the fix restricts it to the forward segment (n_forward).
        """
        counter = {"count": 0}
        rec = _run_config(
            2,
            self.ENERGY,
            self.DELTA_E_TURN,
            self.N_TURNS,
            controller_call_counter=counter,
        )
        n_forward = int(np.sum(rec["n_forward"]))
        n_total = int(np.sum(rec["n_total"]))
        # Sanity: the backfill segments really are a large fraction.
        self.assertGreater(n_total, 1.5 * n_forward)
        self.assertEqual(
            counter["count"],
            n_forward,
            f"controller stepped {counter['count']} times, expected "
            f"{n_forward} (forward cells); {n_total} total cells exist -- "
            "it is being stepped on the backfill reconstruction segments",
        )

    def test_single_section_controller_skips_turn0_backfill(self):
        """
        Control: single section is stepped only on forward cells too.

        Single-section rings still reconstruct the very first turn by
        backfill (n_total > n_forward on turn 0), so the gate must skip
        that backfill span here as well -- the controller count equals the
        forward cells, not the total.
        """
        counter = {"count": 0}
        rec = _run_config(
            1,
            self.ENERGY,
            self.DELTA_E_TURN,
            self.N_TURNS,
            controller_call_counter=counter,
        )
        self.assertGreater(
            int(np.sum(rec["n_total"])), int(np.sum(rec["n_forward"]))
        )  # turn-0 backfill exists
        self.assertEqual(counter["count"], int(np.sum(rec["n_forward"])))


class TestDrivenSteadyStateFastRamp(unittest.TestCase):
    """
    A driven, beam-free cavity holds its steady state on the fast ramp.

    With the matched generator bias and no beam the coarse recursion has the
    exact fixed point ``V_ss = 2 (R/Q) Q_L I_gen == V_DESIGN``: the per-step
    decay and the per-step drive both scale with ``omega * dt``, so the fixed
    point is independent of the RF frequency *and* of the step size. An
    on-resonance cavity (``delta_omega = 0``) driven by a constant generator
    must therefore sit at ``V_ss`` turn after turn, however fast the ramp
    moves and however many RF stations the ring has. Single section does, to
    ~2e-12.

    Multi-section used to rotate the carried antenna voltage by the per-turn
    grid-vs-carrier phase ``sum_k (omega_k - omega_0) T_seg,k``, which is a
    registration phase of the *piecewise* coarse grid against the single
    forward demodulation carrier. Applying it to the state also hit the
    generator-driven field -- which carries no such error, being re-injected
    on the current grid every cell -- and the constant drive then pulled the
    rotating state back toward the real axis, so ``|V_ant|`` drifted ~3 %
    over 5 turns (~0.6 %/turn, diverging). The phase is now carried on the
    demodulation/readout carrier, where it belongs, leaving the state and
    hence the driven steady state exact.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 5
    # The single-section control holds V_ss to ~2e-12; gate far above that
    # floor and far below the ~3e-2 the state rotation produced.
    GATE = 1e-8

    def _assert_holds_steady_state(self, n_sections: int) -> None:
        """
        Track a driven, beam-free ring and assert ``|V_ant| == V_ss``.

        Parameters
        ----------
        n_sections
            Number of RF stations in the ring.
        """
        rec = _run_config(
            n_sections,
            self.ENERGY,
            self.DELTA_E_TURN,
            self.N_TURNS,
            intensity=0.0,
            use_controller=False,
        )
        deviation = np.abs(rec["v_last"] / V_DESIGN - 1.0)
        self.assertLess(
            float(deviation.max()),
            self.GATE,
            f"{n_sections} section(s): |V_ant| left the driven steady state "
            f"by {float(deviation.max()):.3e} (relative) over "
            f"{self.N_TURNS} turns; per-turn "
            f"{deviation.max(axis=1) if deviation.ndim > 1 else deviation}",
        )

    def test_single_section_holds_steady_state(self):
        """Control: one station is exact on the fast ramp."""
        self._assert_holds_steady_state(1)

    def test_multi_section_holds_steady_state(self):
        """Two stations must be exact too -- the regression under test."""
        self._assert_holds_steady_state(2)

    def test_four_sections_hold_steady_state(self):
        """Four stations: three backfill segments per passage."""
        self._assert_holds_steady_state(4)


class TestDrivenFeedbackIsPhaseNeutralWithoutBeam(unittest.TestCase):
    """
    A driven, beam-free cavity on its setpoint must hand the station NO phase.

    The in-repo counterpart of the RCS example's
    ``test_feedback_is_a_no_op_without_beam``: with the matched generator
    bias and zero intensity the cavity sits exactly on its setpoint (see
    ``TestDrivenSteadyStateFastRamp`` for the magnitude), so the feedback
    must be a no-op -- ``phase_correction == 0`` on every turn. The
    generator drive is locked to the DESIGN frequency, whose per-segment
    values the coarse grid already samples, so the driven field carries no
    grid-vs-carrier registration phase: adding the multi-section
    registration phase ``Psi = sum_k (omega_k - omega_0) T_seg,k`` to the
    generator-driven component at the readout is a bookkeeping error that
    walks the bucket off the design synchronous phase with no beam at all.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 6
    # The residual is FP dust of the fine-grid solve; the bug this pins was
    # ~0.3 rad/turn on this ring.
    TOLERANCE = 1.0e-12

    def _assert_phase_neutral(self, use_controller: bool) -> None:
        """
        Track a driven, beam-free 2-section fast ramp; expect zero phase.

        Parameters
        ----------
        use_controller
            Whether the matched bias is held by a PI loop (True) or fed
            forward as a constant current (False).
        """
        rec = _run_config(
            2,
            self.ENERGY,
            self.DELTA_E_TURN,
            self.N_TURNS,
            intensity=0.0,
            use_controller=use_controller,
        )
        phi = np.abs(np.array(rec["phi_corr"]))
        self.assertLess(
            float(phi.max()),
            self.TOLERANCE,
            "driven beam-free feedback applies a rigid RF phase: per turn "
            f"{np.array(rec['phi_corr'])} rad",
        )

    def test_matched_bias_applies_no_phase(self):
        """Constant matched drive: the headline zero-intensity no-op."""
        self._assert_phase_neutral(use_controller=False)

    def test_pi_loop_applies_no_phase(self):
        """A PI holding the same setpoint must be phase-neutral too."""
        self._assert_phase_neutral(use_controller=True)


class TestDesignLockedDriveWalkOffUnderRFOffset(unittest.TestCase):
    r"""
    Under ``delta_omega_rf`` the design-locked drive walks off the actual RF.

    The klystron drive follows the DESIGN frequency. With a station
    RF-frequency offset the actual RF accumulates the kick-clock slip
    ``int delta_omega_rf dt`` relative to the design clock, so the driven
    (generator) field must appear at MINUS that slip relative to the
    actual RF -- real physics, not a bookkeeping artefact. The station
    applies its kick clock ``delta_phi_rf`` through ``phi_rf`` and the
    live tail of the slip is ``_carrier_slip_gap``, so the anchoring rule

        (net phase relative to actual RF) = -(delta_phi_rf + live gap)

    reduces to ``phase_correction == -delta_phi_rf`` for a beam-free,
    matched-bias cavity (the readout composition subtracts the full slip
    from the generator component and then adds back the live gap).
    """

    ENERGY = 63.0e9
    N_TURNS = 6
    #: RF-frequency offset as a fraction of omega_rf: ~0.016 rad of slip
    #: per turn -- far above the readout's FP floor, far below a wrap.
    OFFSET_FRACTION = 1.0e-7
    TOLERANCE = 1.0e-9

    def test_driven_field_appears_at_minus_the_kick_clock_slip(self):
        """Beam-free driven cavity: ``phase_correction == -delta_phi_rf``."""
        harmonic = int(HARMONIC - HARMONIC % 2)
        from blond import ConstantMagneticCycle

        t_rev = ConstantMagneticCycle(
            reference_particle=mu_plus,
            value=self.ENERGY,
            in_unit="total energy",
        ).get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
        delta_omega_rf = self.OFFSET_FRACTION * 2.0 * np.pi * harmonic / t_rev
        rec = _run_config(
            1,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            intensity=0.0,
            use_controller=False,
            delta_omega_rf=delta_omega_rf,
        )
        phi_corr = np.array(rec["phi_corr"])[:, 0]
        delta_phi_rf = np.array(rec["delta_phi_rf"])[:, 0]
        # The premise has teeth: the kick clock really accumulates.
        self.assertGreater(float(np.abs(delta_phi_rf[-1])), 0.05)
        np.testing.assert_allclose(
            phi_corr,
            -delta_phi_rf,
            atol=self.TOLERANCE,
            err_msg=(
                "the design-locked drive does not appear at minus the "
                "kick-clock slip relative to the actual RF"
            ),
        )


class TestDetunedLoopHoldsSetpointAcrossBackfillSpan(unittest.TestCase):
    r"""
    A detuned, PI-regulated cavity must hold its setpoint all turn long.

    With ``delta_omega != 0`` the matched no-beam drive is no longer the
    feedforward bias but ``I_0 (1 - i tan psi)``,
    ``tan psi = 2 Q_L delta_omega / omega_rf``: cancelling the detuning
    precession needs a reactive standing current, which the PI finds on the
    forward span. A multi-section ring then replays the remaining
    ``(N - 1) / N`` of the turn as no-beam backfill segments, and it must
    replay it with the current the loop actually held. Driving it with the
    bias instead lets the antenna voltage precess for most of every turn.

    The excursion is analytic: the discarded drive is purely reactive, so
    over a backfill span of duration ``T``

    .. math:: |\Delta V| / V_\mathsf{set} \simeq \Delta\omega\, T,

    independent of ``Q_L`` and ``R/Q``. Here (one half-bandwidth of
    detuning, two sections, ``T = t_rev / 2``) that is ``3.2e-2``: 3 % of
    the setpoint every turn, on the very sample that seeds the fine grid
    the bunch is solved on, so it is not self-correcting.

    No beam is tracked on purpose -- without beam loading the correct answer
    is exactly the setpoint on every coarse sample, so the assertion has no
    tolerance budget to hide in.
    """

    N_SECTIONS = 2
    N_TURNS = 5
    ENERGY = 63.0e9  # constant energy: no ramp, no frame slip
    # Skip turn 1: the loop is still converging from ``initial_voltage``.
    SETTLED = slice(1, None)
    TOLERANCE = 1e-6

    def test_detuned_loop_holds_setpoint_over_the_whole_turn(self):
        """Backfill span must not drive the detuned cavity off setpoint."""
        rec = _run_config(
            self.N_SECTIONS,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            intensity=0.0,
            detuning_half_bandwidths=1.0,
        )
        worst = float(rec["v_dev_grid"][self.SETTLED].max())
        self.assertLess(
            worst,
            self.TOLERANCE,
            f"detuned PI loop leaves the setpoint by {worst:.3e} relative; "
            "the no-beam backfill span is driven by the feedforward bias "
            "instead of the current the loop held",
        )

    def test_four_sections_hold_setpoint_over_the_whole_turn(self):
        """
        Four sections: the backfill span is 3/4 of the turn, not 1/2.

        The excursion scales with the backfill-span duration
        ``T = (N - 1) / N * t_rev``, so this is the direct fingerprint of
        the backfill reconstruction rather than of any forward-pass effect.
        """
        rec = _run_config(
            4,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            intensity=0.0,
            detuning_half_bandwidths=1.0,
        )
        worst = float(rec["v_dev_grid"][self.SETTLED].max())
        self.assertLess(
            worst,
            self.TOLERANCE,
            f"detuned PI loop leaves the setpoint by {worst:.3e} relative "
            "over a three-quarter-turn backfill span",
        )

    def test_matched_bias_control_case_still_exact(self):
        """
        Control: on resonance the bias IS the held current.

        This is what proves the detuned failure above comes from the
        detuning and not from a broken fixture: same ring, same loop, same
        assertion, only ``delta_omega = 0``.
        """
        rec = _run_config(
            self.N_SECTIONS,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            intensity=0.0,
            detuning_half_bandwidths=0.0,
        )
        self.assertLess(
            float(rec["v_dev_grid"][self.SETTLED].max()), self.TOLERANCE
        )

    def test_undriven_detuned_cavity_is_left_free_running(self):
        """
        Control: with no controller the detuned cavity must still precess.

        Guards against "fixing" the above by writing a matched current into
        the grid unconditionally -- an unregulated detuned cavity has to be
        left alone to precess away from the setpoint.
        """
        rec = _run_config(
            self.N_SECTIONS,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            intensity=0.0,
            use_controller=False,
            detuning_half_bandwidths=1.0,
        )
        self.assertGreater(float(rec["v_dev_grid"][-1].max()), 0.2)


class TestPIFullTrackingSingleSectionFastRamp(unittest.TestCase):
    """Single section, strong beam loading, fast (transition-adjacent) ramp."""

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 8

    # Pinned per-turn trajectories (regenerate with PI_TRACKING_PRINT_PINS=1).
    PIN_V_MIN = np.array(
        [
            28874968.095003456,
            28844619.779986404,
            28792400.399823453,
            28735253.05501849,
            28688525.171576884,
            28660685.300401166,
            28657677.8111442,
            28680212.677146845,
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            57.50127701204241,
            57.466984197052156,
            57.24471693879749,
            56.88475643990175,
            56.555873171380554,
            56.279113314333316,
            55.961772577977406,
            55.601929261577396,
        ]
    )

    @classmethod
    def setUpClass(cls):
        """Run the tracked simulation once."""
        cls.rec = _run_config(1, cls.ENERGY, cls.DELTA_E_TURN, cls.N_TURNS)
        if PRINT_PINS:
            np.set_printoptions(precision=17)
            print("V_MIN:", repr(cls.rec["v_min"][:, 0]))
            print("I_MAX_DEV:", repr(cls.rec["i_max_dev"][:, 0]))

    def test_reference_follows_energy_program(self):
        """The reference energy gains exactly DELTA_E_TURN per turn."""
        np.testing.assert_allclose(
            self.rec["ref_energy"],
            self.ENERGY + self.DELTA_E_TURN * np.arange(1, self.N_TURNS + 1),
            rtol=1e-12,
        )

    def test_beam_loading_sags_the_voltage(self):
        """The bunch passage visibly sags |V_ant| below the setpoint."""
        sag = 1.0 - self.rec["v_min"][:, 0] / V_DESIGN
        self.assertGreater(float(sag.max()), 0.005)
        self.assertLess(float(sag.max()), 0.2)

    def test_loop_acts_on_the_generator_current(self):
        """The PI response is large compared to the bias current."""
        i_response = self.rec["i_max_dev"][:, 0] / I_GEN_BIAS
        self.assertGreater(float(i_response.max()), 0.1)

    def test_voltage_recovers_by_turn_end(self):
        """The loop restores |V_ant| to the setpoint by the end of a turn."""
        v_dev = np.abs(self.rec["v_last"][:, 0] - V_DESIGN) / V_DESIGN
        self.assertLess(float(v_dev.max()), 1e-3)

    def test_bunch_stays_bounded(self):
        """The bunch length stays finite and bounded (no blow-up)."""
        sigma = self.rec["sigma_dt"]
        self.assertLess(float(sigma[-1]), 3.0 * float(sigma[0]))

    def test_pinned_trajectories(self):
        """Characterization: the exact recorded trajectories."""
        if PRINT_PINS or self.PIN_V_MIN is None:
            self.skipTest("pins not recorded yet")
        np.testing.assert_allclose(
            self.rec["v_min"][:, 0], self.PIN_V_MIN, rtol=1e-6
        )
        np.testing.assert_allclose(
            self.rec["i_max_dev"][:, 0], self.PIN_I_MAX_DEV, rtol=1e-6
        )


class TestPIFullTrackingMultiSectionSlowRamp(unittest.TestCase):
    """
    Two sections, strong beam loading, operating-point (slow) ramp.

    Uses the operating-point ramp so the pinned trajectories characterize a
    representative production regime; the transition-adjacent fast ramp is
    covered by ``TestPIFullTrackingMultiSectionFastRamp``.
    """

    ENERGY = 63.0e9
    DELTA_E_TURN = 4.0e6
    N_TURNS = 6

    # Regenerated when the coarse envelope was split into its generator-
    # and beam-sourced components and the PI error moved to the KICK-frame
    # sum (see ``_update_frame_rotations``): the loop now regulates the
    # applied kick, whose difference from the former raw state is
    # ``V_beam (1 - e^{i Psi})`` with the slow ramp's registration phase
    # ``Psi ~ 7e-6 rad/turn``. That moved |V_ant| by <= 2.4e-6 relative
    # and the current response by <= 1.7e-6 -- marginally beyond the 1e-6
    # pin tolerance, a real (declared) modelling shift, not FP noise.
    # Both stations still hold the setpoint and respond to the loading,
    # which the behavioural tests above assert independently.
    PIN_V_MIN = np.array(
        [
            [29720241.870437603, 29718291.578992896],
            [29714444.558270626, 29708843.53792085],
            [29701466.33210081, 29692091.054452974],
            [29681368.34197527, 29669577.804998245],
            [29657420.28679109, 29645289.890251752],
            [29633652.159150466, 29622668.427521624],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.62027262356375, 56.620266121986276],
            [56.623039259161175, 56.623083234533965],
            [56.62742376709935, 56.62913667190933],
            [56.63700927637203, 56.64423175134397],
            [56.65642904646521, 56.6677698159723],
            [56.69138791226268, 56.716714106921884],
        ]
    )

    @classmethod
    def setUpClass(cls):
        """Run the tracked simulation once."""
        cls.rec = _run_config(2, cls.ENERGY, cls.DELTA_E_TURN, cls.N_TURNS)
        if PRINT_PINS:
            np.set_printoptions(precision=17)
            print("V_MIN_MS:", repr(cls.rec["v_min"]))
            print("I_MAX_DEV_MS:", repr(cls.rec["i_max_dev"]))

    def test_reference_follows_energy_program(self):
        """The reference energy gains exactly DELTA_E_TURN per turn."""
        np.testing.assert_allclose(
            self.rec["ref_energy"],
            self.ENERGY + self.DELTA_E_TURN * np.arange(1, self.N_TURNS + 1),
            rtol=1e-12,
        )

    def test_beam_loading_sags_both_stations(self):
        """The bunch passage sags |V_ant| at both stations."""
        for section in range(2):
            sag = 1.0 - self.rec["v_min"][:, section] / V_DESIGN
            self.assertGreater(float(sag.max()), 0.005, f"section {section}")
            self.assertLess(float(sag.max()), 0.2, f"section {section}")

    def test_loop_acts_on_both_stations(self):
        """Both stations' PI loops respond to the loading."""
        for section in range(2):
            i_response = self.rec["i_max_dev"][:, section] / I_GEN_BIAS
            self.assertGreater(
                float(i_response.max()), 0.1, f"section {section}"
            )

    def test_voltage_recovers_on_both_stations(self):
        """The loops restore |V_ant| to the setpoint by the turn end."""
        v_dev = np.abs(self.rec["v_last"] - V_DESIGN) / V_DESIGN
        self.assertLess(float(v_dev.max()), 1e-3)

    def test_pinned_trajectories(self):
        """Characterization: the exact recorded trajectories."""
        if PRINT_PINS or self.PIN_V_MIN is None:
            self.skipTest("pins not recorded yet")
        np.testing.assert_allclose(
            self.rec["v_min"], self.PIN_V_MIN, rtol=1e-6
        )
        np.testing.assert_allclose(
            self.rec["i_max_dev"], self.PIN_I_MAX_DEV, rtol=1e-6
        )


class TestPIFullTrackingMultiSectionFastRamp(unittest.TestCase):
    """
    Two sections, strong beam loading, transition-adjacent fast ramp.

    This configuration used to be excluded: the multi-section grid-vs-carrier
    registration phase was applied as a rotation of the antenna-voltage
    state, which on the fast ramp dragged the driven field off its steady
    state (see ``TestDrivenSteadyStateFastRamp``) and made a PI
    characterization here meaningless. With that phase carried on the
    demodulation/readout carrier instead, the fast ramp behaves like the
    slow one: both loops restore the setpoint by the end of every turn (to
    ~1e-16 relative here) while the ramp is 5x steeper at 1/16 the energy.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 6

    # Regenerated with the split coarse envelope (generator- vs
    # beam-sourced components; see ``_update_frame_rotations``): these
    # pins previously encoded the driven multi-section readout-phase
    # artefact this configuration exists to expose -- the registration
    # phase ``Psi ~ 0.14 rad/turn/station`` was applied to the
    # generator-driven field too, and the PI partially fought that
    # bookkeeping rotation. With the generator component design-anchored
    # and the PI regulating the kick-frame sum, |V_ant| moved by up to
    # 1.8e-2 relative and the current response by up to ~9 % here.
    # ``TestDrivenFeedbackIsPhaseNeutralWithoutBeam`` pins the fixed
    # zero-intensity behaviour these numbers now build on.
    PIN_V_MIN = np.array(
        [
            [29587394.54086683, 29543774.540832333],
            [29676468.31671796, 29553403.276575282],
            [29624356.981359284, 29496297.608280458],
            [29580726.649029866, 29555686.28042151],
            [29735746.97591483, 29825765.17887941],
            [29969674.09771953, 29941278.22053024],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.682189929467, 56.7590343761462],
            [55.93044248155129, 55.58787727483113],
            [53.1462008548574, 52.87825223329836],
            [50.2186915900871, 51.274645770223536],
            [50.28424633707266, 53.22012677798405],
            [54.379862842346625, 58.098186870389405],
        ]
    )

    @classmethod
    def setUpClass(cls):
        """Run the tracked simulation once."""
        cls.rec = _run_config(2, cls.ENERGY, cls.DELTA_E_TURN, cls.N_TURNS)
        if PRINT_PINS:
            np.set_printoptions(precision=17)
            print("V_MIN_MS_FAST:", repr(cls.rec["v_min"]))
            print("I_MAX_DEV_MS_FAST:", repr(cls.rec["i_max_dev"]))

    def test_reference_follows_energy_program(self):
        """The reference energy gains exactly DELTA_E_TURN per turn."""
        np.testing.assert_allclose(
            self.rec["ref_energy"],
            self.ENERGY + self.DELTA_E_TURN * np.arange(1, self.N_TURNS + 1),
            rtol=1e-12,
        )

    def test_beam_loading_sags_both_stations(self):
        """The bunch passage sags |V_ant| at both stations."""
        for section in range(2):
            sag = 1.0 - self.rec["v_min"][:, section] / V_DESIGN
            self.assertGreater(float(sag.max()), 0.005, f"section {section}")
            self.assertLess(float(sag.max()), 0.2, f"section {section}")

    def test_loop_acts_on_both_stations(self):
        """Both stations' PI loops respond to the loading."""
        for section in range(2):
            i_response = self.rec["i_max_dev"][:, section] / I_GEN_BIAS
            self.assertGreater(
                float(i_response.max()), 0.1, f"section {section}"
            )

    def test_voltage_recovers_on_both_stations(self):
        """The loops restore |V_ant| to the setpoint by the turn end."""
        v_dev = np.abs(self.rec["v_last"] - V_DESIGN) / V_DESIGN
        self.assertLess(float(v_dev.max()), 1e-3)

    def test_bunch_stays_bounded(self):
        """The bunch length stays finite and bounded (no blow-up)."""
        sigma = self.rec["sigma_dt"]
        self.assertLess(float(sigma[-1]), 3.0 * float(sigma[0]))

    def test_pinned_trajectories(self):
        """Characterization: the exact recorded trajectories."""
        if PRINT_PINS or self.PIN_V_MIN is None:
            self.skipTest("pins not recorded yet")
        np.testing.assert_allclose(
            self.rec["v_min"], self.PIN_V_MIN, rtol=1e-6
        )
        np.testing.assert_allclose(
            self.rec["i_max_dev"], self.PIN_I_MAX_DEV, rtol=1e-6
        )


class TestKernelMatchesReferenceEndToEnd(unittest.TestCase):
    """The numba envelope kernel reproduces the reference over a full run."""

    def test_multi_section_kernel_vs_python_bit_identical(self):
        """
        A 2-section multi-turn tracked run is byte-identical either path.

        End-to-end guard for the envelope-kernel bit-identity invariant: it
        drives the real turn loop (reset, backfill reconstruction segments,
        demodulation, forward pass, PI regulation) on the default numba kernel
        and on the pure-Python reference and pins the two byte-for-byte. This
        exercises exactly the multi-section, turn>=1 carried-state backfill
        segment where the kernel's generator-current / beam-current drive must
        match the reference (the isolated regression lives in
        test_envelope_kernel.py; this is the whole-simulation counterpart).
        """
        cls = IQCavityFeedbackTimingClass
        original = cls.use_numba_envelope_kernel
        try:
            cls.use_numba_envelope_kernel = True
            rec_kernel = _run_config(2, 4.0e9, 20.0e6, 4)
            cls.use_numba_envelope_kernel = False
            rec_python = _run_config(2, 4.0e9, 20.0e6, 4)
        finally:
            cls.use_numba_envelope_kernel = original
        for key in ("v_min", "v_last", "i_max_dev"):
            np.testing.assert_array_equal(
                rec_kernel[key],
                rec_python[key],
                err_msg=f"kernel vs python diverged in {key!r}",
            )


if __name__ == "__main__":
    unittest.main()
