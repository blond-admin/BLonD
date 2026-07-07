"""
PI-controlled cavity feedback inside a real tracked ``Simulation``.

Every other PI test drives the controller on hand-built constant-step
grids (see ``test_generator_current_pi_feedback.py``); here the full chain
runs in anger: a matched ``BiGaussian`` ``mu_plus`` bunch with strong beam
loading is tracked through a real ring with the reverse/forward reference
tracking, under strong acceleration, with the
:class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController`
regulating the generator current -- single-section on the fast
(transition-adjacent) ramp and multi-section on the validated slow ramp.

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
            [int(f.rf_centers_lengths[-1]) for f in feedbacks]
        )
        rec["n_total"].append([int(len(f.rf_centers)) for f in feedbacks])
        # Only the forward segment of this turn (the last
        # rf_centers_lengths[-1] samples) -- the reverse part repeats the
        # previous turn's no-beam propagation.
        rec["v_min"].append(
            [
                float(
                    np.abs(
                        f.antenna_voltage_coarse_grid[
                            -int(f.rf_centers_lengths[-1]) :
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
                            -int(f.rf_centers_lengths[-1]) :
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

    Isolation: with a matched bias and NO beam, a frame-consistent loop
    sits exactly at its no-beam steady state (V = V_ss, I_gen = bias) on
    every turn -- any drift of the generator current or antenna voltage is
    the reverse-span frame error, not beam loading. Single-section (no
    reverse segments) is the clean control.
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
    representative production regime; the multi-section frame correction in
    the timing class handles the fast ramp too (see
    ``test_multiturn_fast_ramp_multisection``), it is just not needed to make
    this characterization meaningful.
    """

    ENERGY = 63.0e9
    DELTA_E_TURN = 4.0e6
    N_TURNS = 6

    PIN_V_MIN = np.array(
        [
            [29720238.603471544, 29718279.461305067],
            [29714426.67089729, 29708826.90995844],
            [29701438.90212061, 29692066.813471388],
            [29681334.23090612, 29669543.93249241],
            [29657379.276949894, 29645252.965956792],
            [29633606.974730425, 29622620.549507413],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.616838163679994, 56.60653630576703],
            [56.61617122941309, 56.61623073099019],
            [56.620592519533005, 56.62230626917159],
            [56.63020684148947, 56.637435182939264],
            [56.64962620143712, 56.66103878175048],
            [56.68469130159842, 56.7100732336399],
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


if __name__ == "__main__":
    unittest.main()
