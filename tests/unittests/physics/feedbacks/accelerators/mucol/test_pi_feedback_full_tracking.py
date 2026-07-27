"""
PI-controlled cavity feedback inside a real tracked ``Simulation``.

Every other PI test drives the controller on hand-built constant-step
grids (see ``test_generator_current_pi_feedback.py``); here the full chain
runs in anger: a matched ``BiGaussian`` ``mu_plus`` bunch with strong beam
loading is tracked through a real ring with the reverse/forward reference
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
        by the structural reverse-span tests.
    controller_call_counter
        If given, a ``{"count": 0}`` dict; every controller update increments
        ``"count"`` so tests can compare controller steps against the
        recorded forward/total coarse-cell counts.
    use_controller
        If False, attach no PI controller, so the generator current stays at
        the constant ``generator_current_bias``. Used by the driven
        steady-state tests, which need the open-loop cavity response.

    Returns
    -------
    dict
        Per-turn trajectories per station: ``v_min`` (minimum antenna
        voltage magnitude over the forward segment -- the beam-loading sag),
        ``v_last`` (last coarse sample -- the recovered voltage),
        ``i_max_dev`` (maximum generator-current deviation from the bias --
        the loop response), ``n_forward``/``n_total`` (forward and total
        coarse cells per turn); plus ``ref_energy`` and ``sigma_dt``.
    """
    from blond import ConstantMagneticCycle

    cycle_probe = ConstantMagneticCycle(
        reference_particle=mu_plus, value=energy, in_unit="total energy"
    )
    t_rev = cycle_probe.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    harmonic = int(HARMONIC - HARMONIC % (2 * n_sections))
    t_rf = t_rev / harmonic

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
            delta_omega=0.0,
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
            # test_envelope_kernel (a kernel stepping the PI on the reverse
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
        # rf_centers_lengths[-1] samples) -- the reverse part repeats the
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
        rec["ref_energy"].append(float(b.reference.total_energy))
        rec["sigma_dt"].append(float(np.std(b.dt.array_local)))

    sim.run_simulation(
        (beam,), n_turns=n_turns, callbacks=callback, show_progressbar=False
    )
    for key, values in rec.items():
        rec[key] = np.array(values)
    return rec


class TestPIReverseSpanFrameConsistency(unittest.TestCase):
    """
    The PI loop must not act on the reverse reconstruction segments.

    A multi-section feedback rebuilds the previous turn each turn as
    ``no_beam`` reverse segments before the forward pass. The controller
    must be stepped only on the forward (real, current-turn) segment: the
    reverse cells carry a per-segment frame phase (corrected only on the
    last sample), so a controller stepped there under a fast ramp
    integrates frame-rotated errors and injects spurious quadrature
    current.

    Isolation: rather than measuring the (small) frame drift, these tests
    pin the fix *structurally* by counting controller updates. A
    frame-consistent loop must step the controller once per forward
    (real-passage) coarse cell and never on a reverse reconstruction cell,
    so ``controller-call count == sum(n_forward)`` while ``n_total`` is
    strictly larger (the reverse cells exist but must be skipped). This is
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
        including the reverse reconstruction segments (n_total per station),
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
        # Sanity: the reverse segments really are a large fraction.
        self.assertGreater(n_total, 1.5 * n_forward)
        self.assertEqual(
            counter["count"],
            n_forward,
            f"controller stepped {counter['count']} times, expected "
            f"{n_forward} (forward cells); {n_total} total cells exist -- "
            "it is being stepped on the reverse reconstruction segments",
        )

    def test_single_section_controller_skips_turn0_reverse(self):
        """
        Control: single section is stepped only on forward cells too.

        Single-section rings still reconstruct the very first turn in
        reverse (n_total > n_forward on turn 0), so the gate must skip that
        reverse span here as well -- the controller count equals the forward
        cells, not the total.
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
        )  # turn-0 reverse exists
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
        """Four stations: three reverse segments per passage."""
        self._assert_holds_steady_state(4)


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

    # Regenerated when the multi-section grid-vs-carrier registration phase
    # moved from a rotation of the antenna-voltage STATE onto the
    # demodulation/readout carrier (see ``_track`` and
    # ``TestDrivenSteadyStateFastRamp``). The PI loop regulates the raw
    # coarse antenna voltage, so it used to see a state carrying that
    # rotation and now sees the true antenna voltage: |V_ant| itself moves
    # by <1e-6 relative here (the slow ramp's phase is tiny), while the
    # generator-current response -- a small difference of large numbers --
    # shifts by ~2.4e-4. Both stations still hold the setpoint and respond
    # to the loading, which the behavioural tests above assert
    # independently.
    PIN_V_MIN = np.array(
        [
            [29720238.60345596, 29718279.461241655],
            [29714423.366358045, 29708813.574096315],
            [29701429.028963964, 29692046.72468031],
            [29681320.80143154, 29669519.48808996],
            [29657362.949341103, 29645226.38311076],
            [29633591.32297692, 29622598.279163722],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.62027262356374, 56.620271839144614],
            [56.623036747979334, 56.62309322895194],
            [56.62744895816398, 56.62916165129061],
            [56.63706159810912, 56.64423272762598],
            [56.6564447799741, 56.667824214982446],
            [56.69140189912312, 56.716810402245585],
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

    PIN_V_MIN = np.array(
        [
            [29510809.734342266, 29257583.346666165],
            [29313937.061649576, 29025938.910384115],
            [29180931.487604056, 28969679.411779363],
            [29224924.59541966, 29102380.528750226],
            [29452874.896333005, 29360901.638597563],
            [29733785.496557347, 29579358.6584041],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.682189929466844, 56.71161284327756],
            [56.54583554032278, 56.32585826652903],
            [55.91086834159208, 55.60132240962575],
            [55.246952051712114, 55.1607570630149],
            [55.65299112097477, 56.34506291118168],
            [56.753367631546965, 56.896968910363576],
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
        drives the real turn loop (reset, reverse reconstruction segments,
        demodulation, forward pass, PI regulation) on the default numba kernel
        and on the pure-Python reference and pins the two byte-for-byte. This
        exercises exactly the multi-section, turn>=1 carried-state reverse
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
