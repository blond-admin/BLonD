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
from collections.abc import Callable
from unittest import mock

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
    GeneratorCurrentController,
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
    controller_factory: Callable[[], GeneratorCurrentController] | None = None,
    generator_current_bias: complex = I_GEN_BIAS + 0.0j,
    initial_voltage: float = V_DESIGN,
    per_turn_hook: Callable[[list, list], None] | None = None,
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
    controller_factory
        If given, called once per station to build the controller that
        replaces the PI controller (still subject to ``use_controller``).
    generator_current_bias
        Feedforward generator-current bias [A] of every feedback; the
        matched bias by default.
    initial_voltage
        Initial antenna voltage [V] of every feedback; ``V_DESIGN`` by
        default.
    per_turn_hook
        If given, called at the end of every turn as
        ``per_turn_hook(feedbacks, stations)``, so a test can record state
        that the grids of the next passage overwrite.

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
        if controller_factory is not None:
            controller = controller_factory()
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
            generator_current_bias=generator_current_bias,
            n_cavities=1,
            initial_voltage=initial_voltage,
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
            # reference path here. The kernel steps the PI on the same cells
            # (backfill segments included); that equivalence is pinned by the
            # byte-identical coarse grids in test_envelope_kernel.
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
        "i_backfill_ptp": [],
    }

    def callback(_sim, b):
        rec["n_forward"].append(
            [int(f._rf_centers_lengths[-1]) for f in feedbacks]
        )
        rec["n_total"].append([int(len(f._rf_centers)) for f in feedbacks])
        # Peak-to-peak generator-current magnitude over the BACKFILL span
        # (everything before the forward segment). A controller that only
        # steps on the forward segment leaves a zero-order hold here, so
        # this is exactly 0.0; one that regulates the whole turn does not.
        rec["i_backfill_ptp"].append(
            [
                float(
                    np.ptp(
                        np.abs(
                            f.generator_current_coarse_grid[
                                : -int(f._rf_centers_lengths[-1])
                            ]
                        )
                    )
                )
                if len(f._rf_centers) > int(f._rf_centers_lengths[-1])
                else 0.0
                for f in feedbacks
            ]
        )
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
        if per_turn_hook is not None:
            per_turn_hook(feedbacks, stations)

    sim.run_simulation(
        (beam,), n_turns=n_turns, callbacks=callback, show_progressbar=False
    )
    for key, values in rec.items():
        rec[key] = np.array(values)
    return rec


class TestPIStepsOnEveryTrackedCell(unittest.TestCase):
    """
    The PI loop must act on every tracked coarse cell, backfill included.

    A multi-section feedback rebuilds the interval since its previous
    passage as ``no_beam`` backfill segments before the forward pass. A
    real LLRF regulates continuously, so the controller has to be stepped
    on those cells too -- otherwise it is open-loop for ``(N - 1) / N`` of
    every turn, holding whatever current the forward pass last commanded,
    and the "regulation" of a 16-section ring runs at a 6 % duty cycle.

    Pinned two ways. Structurally, the controller-call count must equal
    the TOTAL number of coarse cells, not just the forward ones. Physically,
    the generator current over the backfill span must not be a constant
    hold: a stepped loop varies it there.

    Both are asserted in the regime where every frame rotation is exactly
    unity (multi-section, constant energy, no RF-frequency offset --
    ``phi_acc = gap = delta_phi_rf = 0``), so the expectation is
    unambiguous. Under a ramp each backfill cell is regulated in the frame
    of the phase accumulated up to it; that frame is pinned by
    ``TestKickFrameVoltageIsContinuousAcrossBackfill``, not here.
    """

    ENERGY = 4.0e9
    N_TURNS = 3

    def test_controller_stepped_on_every_cell_two_sections(self):
        """Two sections, constant energy: calls == total cells."""
        counter = {"count": 0}
        rec = _run_config(
            2,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            controller_call_counter=counter,
        )
        n_forward = int(np.sum(rec["n_forward"]))
        n_total = int(np.sum(rec["n_total"]))
        # Sanity: the backfill segments really are a large fraction.
        self.assertGreater(n_total, 1.5 * n_forward)
        self.assertEqual(
            counter["count"],
            n_total,
            f"controller stepped {counter['count']} times, expected "
            f"{n_total} (every tracked cell); only {n_forward} forward "
            "cells were stepped -- the loop is open-loop on the backfill "
            "span",
        )

    def test_controller_stepped_on_every_cell_four_sections(self):
        """Four sections: the backfill span is 3/4 of the turn."""
        counter = {"count": 0}
        rec = _run_config(
            4,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            controller_call_counter=counter,
        )
        self.assertEqual(counter["count"], int(np.sum(rec["n_total"])))

    def test_single_section_turn0_backfill_is_stepped_too(self):
        """
        Single section: the turn-0 backfill cells are stepped as well.

        A single-section ring reconstructs its very first turn by backfill
        (``n_total > n_forward`` on turn 0); those cells are tracked, so the
        controller must step on them like on any other.
        """
        counter = {"count": 0}
        rec = _run_config(
            1,
            self.ENERGY,
            0.0,
            self.N_TURNS,
            controller_call_counter=counter,
        )
        self.assertGreater(
            int(np.sum(rec["n_total"])), int(np.sum(rec["n_forward"]))
        )
        self.assertEqual(counter["count"], int(np.sum(rec["n_total"])))

    def test_generator_current_is_regulated_on_the_backfill_span(self):
        """
        The backfill-span generator current is not a zero-order hold.

        With the beam loading of the forward passage still decaying into
        the backfill span, a loop that steps there must move the current;
        a forward-only loop leaves it constant, so the peak-to-peak
        magnitude over that span is exactly ``0.0``.
        """
        rec = _run_config(2, self.ENERGY, 0.0, self.N_TURNS)
        # Turn 0's backfill is the pre-fill; look at the settled turns.
        ptp = np.array(rec["i_backfill_ptp"][1:])
        self.assertGreater(
            float(ptp.max()),
            0.0,
            "generator current is constant over the backfill span: the "
            "controller is not being stepped there",
        )


class _ZeroDriveRecordingController(GeneratorCurrentController):
    """
    Controller that commands no generator current and records its errors.

    Zero gains and zero bias by construction: every command is exactly
    ``0 + 0j``. It does not advertise ``supports_envelope_scan``, so the
    feedback drives it cell by cell on the reference path, and every error
    reaches :meth:`update_generator_current` in the order the cells are
    tracked.
    """

    def __init__(self) -> None:
        self.errors: list[complex] = []

    def update_generator_current(
        self, error: complex, delta_t: float
    ) -> complex:
        """
        Record the error and command no current.

        Parameters
        ----------
        error
            Antenna-voltage error of this cell [V].
        delta_t
            Time step of this cell [s]; unused.

        Returns
        -------
        generator_current
            Exactly ``0 + 0j`` [A].
        """
        self.errors.append(complex(error))
        return 0.0 + 0.0j


def _design_omega_rf(total_energy: float, harmonic: int) -> float:
    """
    Design RF angular frequency of the test ring at a reference energy.

    Parameters
    ----------
    total_energy
        Reference total energy [eV].
    harmonic
        RF harmonic.

    Returns
    -------
    omega_rf
        ``2 pi harmonic / t_rev`` [rad/s].
    """
    from blond import ConstantMagneticCycle

    t_rev = ConstantMagneticCycle(
        reference_particle=mu_plus, value=total_energy, in_unit="total energy"
    ).get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    return 2.0 * np.pi * harmonic / t_rev


class TestKickFrameVoltageIsContinuousAcrossBackfill(unittest.TestCase):
    r"""
    The kick-frame antenna voltage must not jump where nothing happens.

    The controller regulates, and the station applies, the antenna voltage
    in the KICK frame. With beam but no generator drive, on resonance and
    without an RF-frequency offset, that voltage is the beam-induced
    envelope rotated by the grid-vs-carrier phase accumulated so far, and
    between two adjacent coarse cells without a beam deposit only two
    things may change it:

    - the cavity decay, ``|V_next| / |V_prev| = exp(-omega dt / (2 Q_L))``
      (the exact propagator, on resonance);
    - the smooth drift of the accumulated phase, ``(omega_carrier -
      omega_k) dt``, of the backfill segment ``k`` being replayed against
      the previous passage's forward carrier.

    It is checked at every place where the grid changes hands -- (a) the
    last forward cell of passage ``m`` to the first backfill cell of
    passage ``m + 1``, (b) a backfill segment boundary, (c) the last
    backfill cell to the first forward cell, which
    ``forbid_charge_in_first_coarse_cell`` keeps charge-free -- and at
    every step inside a backfill segment, from passage 1 on (passage 0
    carries no beam-induced voltage before its own deposit).

    **Gates.** A coarse step is at most ``1.5 t_rf``: a segment's unfilled
    tail (at most one period) plus the next segment's first local centre
    (half a period into it). The frequency spread over one passage
    interval is below the one-turn change ``omega(E + dE_turn) -
    omega(E)``, largest at the injection energy: ``2.8e4`` rad/s on this
    fast ramp. So a phase step is below ``3.3e-5`` rad and a magnitude
    step below ``1.5 pi / Q_L = 3.7e-6``; the gates sit ``GATE_FACTOR``
    above both, at ``3.3e-4`` rad and ``3.7e-5``.

    **What it catches.** Rotating the whole backfill span of passage
    ``m + 1`` with that passage's FINAL accumulated phase, while the last
    forward cell of passage ``m`` carries passage ``m``'s, makes the
    voltage jump at (a) by the whole per-passage increment -- measured
    0.136 .. 0.140 rad on two sections and 0.203 .. 0.211 rad on four,
    400 to 650 times the phase gate. A frame rotation never changes
    ``|V|``, so that defect leaves the magnitude alone; the magnitude gate
    guards the per-cell rotation against changing it.

    **Fixture.** A zero-gain, zero-bias controller and a zero feedforward
    bias and initial voltage keep the generator-sourced component exactly
    zero, so the composed sum IS the beam-sourced component; the
    controller records every error, so the kick-frame voltage of each
    tracked cell is ``pi_setpoint - error`` (the actuator rotation is
    exactly unity without an RF-frequency offset). The coarse grid runs on
    the exact exponential propagator, so between deposit-free cells the
    magnitude changes by exactly the decay; the frame rotation is applied
    outside the propagator.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 3
    GATE_FACTOR = 10.0
    # The recovered voltage carries rounding of order |V_set| * eps ~ 7e-9
    # V; a deposit leaves ~1.6 MV, so every checked cell must hold far more
    # than that floor for its phase to mean anything.
    MIN_CHECKED_VOLTAGE = 1.0e5

    @classmethod
    def _gates(cls, n_sections: int) -> tuple[float, float]:
        """
        Phase and magnitude gates for adjacent cells without a deposit.

        Parameters
        ----------
        n_sections
            Number of RF stations, which fixes the harmonic.

        Returns
        -------
        phase_gate, magnitude_gate
            Largest admissible ``|angle(V_next / V_prev)|`` [rad] and
            ``| |V_next| / |V_prev| - 1 |``.
        """
        harmonic = int(HARMONIC - HARMONIC % (2 * n_sections))
        omega_rf = _design_omega_rf(cls.ENERGY, harmonic)
        largest_step = 1.5 * 2.0 * np.pi / omega_rf
        frequency_spread = (
            _design_omega_rf(cls.ENERGY + cls.DELTA_E_TURN, harmonic)
            - omega_rf
        )
        phase_bound = frequency_spread * largest_step
        decay_bound = omega_rf * largest_step / (2.0 * Q_L)
        return cls.GATE_FACTOR * phase_bound, cls.GATE_FACTOR * decay_bound

    def _record(self, n_sections: int, delta_e_turn: float) -> tuple:
        """
        Track the ring and record every error and every passage's grid.

        Parameters
        ----------
        n_sections
            Number of RF stations.
        delta_e_turn
            Reference energy gain per turn [eV].

        Returns
        -------
        controllers, passages
            One recording controller per station, and per turn one record
            per station of that passage's grid and composition.
        """
        controllers = []
        passages = []

        def build_controller():
            controller = _ZeroDriveRecordingController()
            controllers.append(controller)
            return controller

        def record_passage(feedbacks, stations):
            passages.append(
                [
                    {
                        "segment_lengths": np.array(
                            feedback.rf_centers_lengths
                        ),
                        "sum_is_beam_component": bool(
                            np.array_equal(
                                feedback.antenna_voltage_coarse_grid,
                                feedback.antenna_voltage_beam_coarse_grid,
                            )
                        ),
                        "generator_silent": not (
                            np.any(feedback.antenna_voltage_gen_coarse_grid)
                            or np.any(feedback.generator_current_coarse_grid)
                        ),
                        "edge_beam_currents": (
                            complex(
                                feedback.beam_current_forward_coarse_grid[0]
                            ),
                            complex(
                                feedback.beam_current_forward_coarse_grid[-1]
                            ),
                        ),
                        "pi_setpoint": complex(feedback.pi_setpoint),
                        "delta_phi_rf": float(station.delta_phi_rf),
                    }
                    for feedback, station in zip(feedbacks, stations)
                ]
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run_config(
                n_sections,
                self.ENERGY,
                delta_e_turn,
                self.N_TURNS,
                controller_factory=build_controller,
                generator_current_bias=0.0 + 0.0j,
                initial_voltage=0.0,
                per_turn_hook=record_passage,
            )
        return controllers, passages

    def _assert_fixture_premises(self, controllers, passages) -> None:
        """
        Assert the recorded errors are the kick-frame beam voltage per cell.

        Parameters
        ----------
        controllers
            The recording controllers, one per station.
        passages
            The per-turn passage records of :meth:`_record`.
        """
        for station_index, controller in enumerate(controllers):
            records = [turn[station_index] for turn in passages]
            n_cells = sum(int(np.sum(r["segment_lengths"])) for r in records)
            with self.subTest(station=station_index, premise="alignment"):
                # One error per tracked cell, none skipped: the errors line
                # up with the cells passage by passage.
                self.assertEqual(len(controller.errors), n_cells)
            for passage, record in enumerate(records):
                with self.subTest(station=station_index, passage=passage):
                    self.assertTrue(record["generator_silent"])
                    self.assertTrue(record["sum_is_beam_component"])
                    self.assertEqual(record["pi_setpoint"], V_DESIGN + 0.0j)
                    # exp(+i delta_phi_rf) short-circuits to exactly 1.
                    self.assertEqual(record["delta_phi_rf"], 0.0)
                    self.assertEqual(record["edge_beam_currents"], (0j, 0j))

    @staticmethod
    def _later_cells_by_boundary(segment_lengths_per_passage) -> dict:
        """
        Later cell of every checked pair of adjacent cells, by kind.

        Parameters
        ----------
        segment_lengths_per_passage
            ``rf_centers_lengths`` of each passage of one station.

        Returns
        -------
        dict
            Kind of step -> whole-run indices of the later cell of each
            pair (the earlier one is the index before it).
        """
        later_cells = {
            "passage": [],
            "backfill segment": [],
            "backfill to forward": [],
            "inside backfill segment": [],
        }
        passage_start = 0
        for passage, lengths in enumerate(segment_lengths_per_passage):
            n_backfill = int(np.sum(lengths[:-1]))
            if passage >= 1:
                later_cells["passage"].append([passage_start])
                segment_start = passage_start
                for length in lengths[:-1]:
                    later_cells["inside backfill segment"].append(
                        np.arange(segment_start + 1, segment_start + length)
                    )
                    if segment_start > passage_start:
                        later_cells["backfill segment"].append([segment_start])
                    segment_start += int(length)
                if n_backfill > 0:
                    later_cells["backfill to forward"].append(
                        [passage_start + n_backfill]
                    )
            passage_start += int(np.sum(lengths))
        return {
            kind: np.concatenate(cells).astype(int)
            if cells
            else np.zeros(0, dtype=int)
            for kind, cells in later_cells.items()
        }

    def _assert_continuous(self, n_sections: int, delta_e_turn: float) -> dict:
        """
        Track, check the premises and gate every checked step.

        Parameters
        ----------
        n_sections
            Number of RF stations.
        delta_e_turn
            Reference energy gain per turn [eV].

        Returns
        -------
        dict
            Station index -> number of checked pairs per kind, for the
            callers' non-vacuity checks.
        """
        controllers, passages = self._record(n_sections, delta_e_turn)
        self._assert_fixture_premises(controllers, passages)
        phase_gate, magnitude_gate = self._gates(n_sections)
        checked = {}
        for station_index, controller in enumerate(controllers):
            voltage = V_DESIGN - np.array(controller.errors)
            later_cells = self._later_cells_by_boundary(
                [turn[station_index]["segment_lengths"] for turn in passages]
            )
            checked[station_index] = {
                kind: len(cells) for kind, cells in later_cells.items()
            }
            for kind, later in later_cells.items():
                if len(later) == 0:
                    continue
                ratio = voltage[later] / voltage[later - 1]
                phase_step = np.abs(np.angle(ratio))
                magnitude_step = np.abs(np.abs(ratio) - 1.0)
                worst = int(np.argmax(phase_step))
                with self.subTest(station=station_index, step=kind):
                    self.assertGreater(
                        float(np.abs(voltage[np.r_[later, later - 1]]).min()),
                        self.MIN_CHECKED_VOLTAGE,
                    )
                    self.assertLess(
                        float(phase_step.max()),
                        phase_gate,
                        f"{n_sections} section(s), station {station_index}: "
                        f"the kick-frame voltage turns by "
                        f"{float(phase_step.max()):.4e} rad at a "
                        f"'{kind}' step (cell {int(later[worst])}; all such "
                        f"steps: {np.array2string(phase_step[:6])}), gate "
                        f"{phase_gate:.3e} rad",
                    )
                    self.assertLess(
                        float(magnitude_step.max()),
                        magnitude_gate,
                        f"{n_sections} section(s), station {station_index}: "
                        f"|V| steps by {float(magnitude_step.max()):.4e} at a "
                        f"'{kind}' step, gate {magnitude_gate:.3e}",
                    )
        return checked

    def test_two_sections_fast_ramp(self):
        """Two sections: one backfill segment per passage."""
        checked = self._assert_continuous(2, self.DELTA_E_TURN)
        for counts in checked.values():
            self.assertEqual(counts["passage"], self.N_TURNS - 1)
            self.assertEqual(counts["backfill to forward"], self.N_TURNS - 1)

    def test_four_sections_fast_ramp(self):
        """Four sections: three backfill segments per passage."""
        checked = self._assert_continuous(4, self.DELTA_E_TURN)
        for counts in checked.values():
            self.assertEqual(counts["passage"], self.N_TURNS - 1)
            self.assertEqual(
                counts["backfill segment"], 2 * (self.N_TURNS - 1)
            )
            self.assertEqual(counts["backfill to forward"], self.N_TURNS - 1)

    def test_constant_energy_control_four_sections(self):
        """
        Control: without a ramp the accumulated phase is exactly zero.

        Same ring, recording and gates; only ``delta_e_turn = 0``. The
        voltage must be continuous whatever the rotation bookkeeping, so a
        failure of the ramped cases comes from the ramp, not the fixture.
        """
        self._assert_continuous(4, 0.0)

    def test_single_section_control_fast_ramp(self):
        """
        Control: a single section accumulates no phase, ramp or not.

        Its later passages have no backfill span at all, so only the
        passage boundary is checked, on the same fast ramp.
        """
        checked = self._assert_continuous(1, self.DELTA_E_TURN)
        self.assertEqual(checked[0]["passage"], self.N_TURNS - 1)


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


class TestReflectedPowerIsTotalOnEveryCell(unittest.TestCase):
    r"""
    A driven, beam-free cavity reflects its whole forward power on every cell.

    With no beam, on resonance and with the matched generator bias the
    cavity holds ``V_ss = 2 (R/Q) Q_L I_gen`` on the fast ramp (see
    ``TestDrivenSteadyStateFastRamp``), and a superconducting cavity has
    nowhere else to put the power: ``reflected_power() /
    generator_power()`` must be 1 on every coarse cell of every passage,
    the backfill span included (``TestReflectedPower`` pins that limit on
    single samples).

    The composed grid carries the generator component rotated per cell --
    a backfill cell by the phase accumulated up to it, the forward span by
    the passage's -- so the default readout must subtract the generator
    current with the same per-cell rotation. With the passage's rotation on
    a backfill cell instead, the two differ by the phase ``delta`` still to
    accumulate there, and the ratio becomes ``|2 exp(i delta) - 1|**2 ~
    1 + 2 delta**2``. The same readout with the passage's rotation passed
    explicitly shows that such cells exist, so a pass is not vacuous.

    Measured with the passage's rotation as the default, on the first
    backfill cell of every station's second and later passages: 3.8e-2 ..
    3.9e-2 on two sections and 8.4e-2 .. 8.9e-2 on four -- ``2 delta**2``
    of the 0.14 and 0.21 rad passage increments that
    ``TestKickFrameVoltageIsContinuousAcrossBackfill`` measures.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 3
    #: Measured on this ramp: the default readout sits within 2.2e-15 of 1
    #: on every cell; the passage's rotation on a backfill cell moves it by
    #: ``2 delta**2``, i.e. by 1e-2 already at 0.07 rad.
    GATE = 1.0e-8
    #: Smallest deviation the passage's rotation must produce somewhere on
    #: the backfill span of a ramped multi-section run (non-vacuity).
    MIN_PASSAGE_ROTATION_ERROR = 1.0e-3

    def _record(self, n_sections: int, delta_e_turn: float) -> list:
        """
        Track a driven, beam-free ring and record the reflection ratios.

        Parameters
        ----------
        n_sections
            Number of RF stations.
        delta_e_turn
            Reference energy gain per turn [eV].

        Returns
        -------
        list
            One record per passage of every station: the station index,
            the forward cell count, the ratio of the default readout and
            the ratio with the passage's generator frame rotation on every
            cell.
        """
        records = []

        def record_passage(feedbacks, stations):
            for station_index, feedback in enumerate(feedbacks):
                forward_power = feedback.generator_power()
                passage_rotation = feedback._generator_frame_rotation
                records.append(
                    {
                        "station": station_index,
                        "n_forward": int(feedback.rf_centers_lengths[-1]),
                        "ratio": feedback.reflected_power() / forward_power,
                        "ratio_passage_rotation": (
                            feedback.reflected_power(
                                generator_frame_rotation=passage_rotation
                            )
                            / forward_power
                        ),
                    }
                )

        _run_config(
            n_sections,
            self.ENERGY,
            delta_e_turn,
            self.N_TURNS,
            intensity=0.0,
            use_controller=False,
            per_turn_hook=record_passage,
        )
        return records

    def _assert_total_reflection(
        self, n_sections: int, delta_e_turn: float
    ) -> float:
        """
        Gate the default readout on every cell of every passage.

        Parameters
        ----------
        n_sections
            Number of RF stations.
        delta_e_turn
            Reference energy gain per turn [eV].

        Returns
        -------
        float
            Largest deviation from 1 of the ratio with the passage's
            rotation over all backfill cells, for the non-vacuity checks.
        """
        records = self._record(n_sections, delta_e_turn)
        worst_passage_rotation_error = 0.0
        for index, record in enumerate(records):
            n_cells = len(record["ratio"])
            n_backfill = n_cells - record["n_forward"]
            deviation = np.abs(record["ratio"] - 1.0)
            forward_deviation = np.abs(
                record["ratio_passage_rotation"][n_backfill:] - 1.0
            )
            with self.subTest(
                n_sections=n_sections,
                station=record["station"],
                record=index,
            ):
                self.assertLess(
                    float(deviation.max()),
                    self.GATE,
                    f"{n_sections} section(s): reflected / forward power "
                    f"is off 1 by {float(deviation.max()):.4e} at cell "
                    f"{int(np.argmax(deviation))} of {n_cells} "
                    f"({n_backfill} backfill cells)",
                )
                # The forward span is composed with the passage's rotation,
                # so there both readouts must agree with total reflection.
                self.assertLess(float(forward_deviation.max()), self.GATE)
            if n_backfill > 0:
                worst_passage_rotation_error = max(
                    worst_passage_rotation_error,
                    float(
                        np.abs(
                            record["ratio_passage_rotation"][:n_backfill] - 1.0
                        ).max()
                    ),
                )
        return worst_passage_rotation_error

    def test_two_sections_fast_ramp(self):
        """Two sections: the backfill span is half of every passage."""
        worst = self._assert_total_reflection(2, self.DELTA_E_TURN)
        self.assertGreater(worst, self.MIN_PASSAGE_ROTATION_ERROR)

    def test_four_sections_fast_ramp(self):
        """Four sections: three backfill segments per passage."""
        worst = self._assert_total_reflection(4, self.DELTA_E_TURN)
        self.assertGreater(worst, self.MIN_PASSAGE_ROTATION_ERROR)

    def test_constant_energy_control_four_sections(self):
        """
        Control: without a ramp every rotation is exactly unity.

        The passage's rotation is then right on every backfill cell too, so
        a failure of the ramped cases comes from the ramp, not the fixture.
        """
        worst = self._assert_total_reflection(4, 0.0)
        self.assertLess(worst, self.GATE)


class TestTrackReadsTheForwardSegmentPhase(unittest.TestCase):
    """
    Each passage takes its accumulated phase from its forward segment.

    On a driven, beam-free fast ramp every passage must hand the
    demodulation and the readout the kick-clock gap plus the forward
    segment's ``accumulated_phase``, and that phase must grow from one
    passage to the next by exactly the increment of this passage's backfill
    segments against the previous passage's forward carrier. The run is
    beam-free because the phase bookkeeping is the only thing under test.
    """

    ENERGY = 4.0e9
    DELTA_E_TURN = 20.0e6
    N_TURNS = 4

    def _record_passages(self, n_sections: int) -> dict:
        """
        Track the ring and record every passage of every station.

        Parameters
        ----------
        n_sections
            Number of RF stations in the ring.

        Returns
        -------
        dict
            Per station index, one ``(segments, kick_clock_slip_gap,
            carrier_slip_gap)`` tuple per passage, in passage order.
        """
        passages: dict = {}
        original_track = IQCavityFeedbackTimingClass._track

        def recording_track(feedback, beam):
            original_track(feedback, beam)
            passages.setdefault(feedback.section_index, []).append(
                (
                    tuple(feedback._segments),
                    feedback._kick_clock_slip_gap,
                    feedback._carrier_slip_gap,
                )
            )

        with mock.patch.object(
            IQCavityFeedbackTimingClass, "_track", recording_track
        ):
            _run_config(
                n_sections,
                self.ENERGY,
                self.DELTA_E_TURN,
                self.N_TURNS,
                intensity=0.0,
                use_controller=False,
            )
        return passages

    def test_carrier_phase_is_the_forward_segment_phase(self):
        passages = self._record_passages(2)
        for section, records in passages.items():
            for index, (segments, kick_gap, carrier_gap) in enumerate(records):
                with self.subTest(section=section, passage=index):
                    self.assertEqual(
                        carrier_gap, kick_gap + segments[-1].accumulated_phase
                    )

    def test_forward_phase_grows_by_the_backfill_increment(self):
        passages = self._record_passages(2)
        n_checked = 0
        for section, records in passages.items():
            for index in range(1, len(records)):
                previous_forward = records[index - 1][0][-1]
                segments = records[index][0]
                backfill = segments[:-1]
                increments = (
                    previous_forward.omega
                    - np.array([segment.omega for segment in backfill])
                ) * np.array([segment.duration for segment in backfill])
                with self.subTest(section=section, passage=index):
                    self.assertEqual(
                        segments[-1].accumulated_phase,
                        previous_forward.accumulated_phase
                        + float(np.sum(increments)),
                    )
                n_checked += 1
        self.assertGreater(n_checked, 0)
        # The fast ramp really accumulates a phase, so the pin is not
        # satisfied by zeros.
        self.assertGreater(
            max(
                abs(records[-1][0][-1].accumulated_phase)
                for records in passages.values()
            ),
            0.0,
        )

    def test_single_section_accumulates_exactly_zero(self):
        passages = self._record_passages(1)
        for records in passages.values():
            for segments, _, _ in records:
                for segment in segments:
                    self.assertEqual(segment.accumulated_phase, 0.0)


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

    # Regenerated for the timestamped fine-grid handoff and bin-centred
    # sampling (2026-09-10). The independent sag/recovery gates are unchanged.
    # Regenerated 2026-09-11 for the exact exponential coarse step (forward
    # Euler retired): i_max_dev moved by <= 1.22e-6 relative and v_min by
    # <= 5.6e-8. Patching the Euler step back in reproduced the previous
    # pins exactly, so the move is the Euler truncation alone.
    # Regenerate with PI_TRACKING_PRINT_PINS=1.
    PIN_V_MIN = np.array(
        [
            28874969.432303112,
            28844507.26413691,
            28792090.37687803,
            28734780.130662423,
            28687939.987144604,
            28660078.762072034,
            28657066.26975915,
            28679681.376705285,
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            57.50120699475231,
            57.467029820637215,
            57.24485871262254,
            56.88472234385635,
            56.55476562737862,
            56.27678988883557,
            55.958367705224816,
            55.5968395759544,
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
    # Regenerated 2026-09-02 for the PI-on-every-tracked-cell change: the
    # loop now regulates over the backfill span rather than holding the
    # forward pass's last command. On this slow ramp the move is only
    # ~1.8e-6 relative (just over the 1e-6 pin tolerance; the backfill
    # cells then still took the passage's frame rotation, per cell only
    # since 2026-09-11); the fast-ramp class shows the ~0.75 % move where
    # the effect is large.
    # Regenerated for bin-centred fine sampling (2026-09-10): voltage moves
    # by <= 1.28e-5 relative; the physical validation gates remain unchanged.
    # Regenerated 2026-09-11 for the exact exponential coarse step (forward
    # Euler retired): i_max_dev moved by <= 1.22e-6 relative and v_min by
    # <= 1.5e-8. Patching the Euler step back in reproduced the previous
    # pins to <= 9.3e-10, so the move is the Euler truncation.
    PIN_V_MIN = np.array(
        [
            [29720242.194671847, 29718281.72720342],
            [29714414.203314707, 29708787.932021037],
            [29701375.86495218, 29691965.543755144],
            [29681187.610761583, 29669361.537674204],
            [29657143.780837268, 29644959.143025886],
            [29633286.627059486, 29622239.902350448],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.6202036789715, 56.6201998723662],
            [56.6231898063932, 56.62319859551216],
            [56.62746066373908, 56.629245415704844],
            [56.637496965218205, 56.644769493713795],
            [56.657129613371076, 56.6686142098662],
            [56.69236043931437, 56.717281304850594],
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
    # Regenerated again for the registration-phase reference fix (the
    # increment is now referred to the PREVIOUS passage's design carrier;
    # see ``RFCenterSegment.accumulated_phase``). See
    # ``test_pinned_trajectories`` for the size of that move.
    # Regenerated once more (2026-09-02) for the PI-on-every-tracked-cell
    # change: the loop now regulates over the backfill span instead of
    # holding the forward pass's last command there, which on this fast
    # ramp moved |V_ant| by ~0.75 % relative -- the physics of the change,
    # not FP noise. ``TestPIStepsOnEveryTrackedCell`` pins the stepping
    # itself; the four behavioural gates in this class are independent of
    # these numbers.
    # Regenerated for bin-centred fine sampling (2026-09-10): voltage moves
    # by <= 2.19e-5 relative; the physical validation gates remain unchanged.
    # Regenerated (2026-09-11) for the per-cell backfill frame rotations and
    # the removal of the forward-Euler coarse step: ``v_min`` moved by at
    # most 3.46e-6 relative (101.7 V, turn 5 / section 1) and ``i_max_dev``
    # by at most 1.03e-4 (5.8 mA, turn 4 / section 0). The exact coarse step
    # accounts for 4.0e-8 / 1.22e-6 of that, including the whole turn-0 move;
    # the rest is the per-cell rotation, which has no phase to act on before
    # turn 1. The physical validation gates remain unchanged.
    PIN_V_MIN = np.array(
        [
            [29587395.018831506, 29543722.390324343],
            [29455155.753527947, 29333766.51191205],
            [29203703.009337213, 29095079.58707814],
            [29028538.370241813, 29012956.799688347],
            [29048306.225948744, 29130971.649211258],
            [29248433.286558207, 29377387.0737793],
        ]
    )
    PIN_I_MAX_DEV = np.array(
        [
            [56.68212090950253, 56.75903304832277],
            [56.846225008249874, 56.86330624651038],
            [56.77700979679049, 56.607720984164736],
            [56.39824534451811, 56.20995488847329],
            [56.082228114374274, 56.05750952171812],
            [56.1316647038373, 56.260469707261294],
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
        """
        Characterization: the exact recorded trajectories.

        MOVED by the registration-phase reference fix. The per-passage
        increment of the accumulated phase (stored per segment as
        ``RFCenterSegment.accumulated_phase`` since 2026-09-11, without
        moving any number) is now referred to
        the PREVIOUS passage's forward-segment design carrier,
        ``sum_k (omega_prev - omega_k) T_seg,k``, instead of to this
        passage's carrier with the opposite sign. This configuration --
        DRIVEN, two sections, accelerating on the fast ramp -- is exactly
        the one that change acts on, so the trajectories really do differ:
        the two expressions agree only to first order in the design-
        frequency programme (they differ by its second difference, which
        vanishes identically for a linear ramp), and the surviving
        curvature term feeds through the demodulation/readout carrier into
        the beam-loading sag the PI then regulates against.

        Measured size of the move: ``v_min`` shifted by at most 4913.32 V
        on ~2.98e7 V (1.65e-4 relative; worst at turn 4 / section 1,
        29825765.18 -> 29830678.50 V) and ``i_max_dev`` by at most
        0.0492457 A on ~50.2 A (9.81e-4 relative; worst at turn 3 /
        section 0, 50.21869159 -> 50.16944594 A). Turn 0 is unchanged to
        roundoff (1.3e-16 relative) in both, as it must be: the first
        passage of a station has no previous carrier and so contributes
        exactly zero.

        The physics gates in this class (sag, loop response, setpoint
        recovery, bounded bunch) all still hold at their existing
        thresholds -- this is a bookkeeping-phase correction, not a change
        of regime.
        """
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
