"""
Two simultaneous counter-rotating beams through the cavity feedback.

The muon-collider RCS accelerates a co-rotating mu+ and a counter-rotating
mu- beam in the same ring; every RF station sees BOTH beams' gap currents,
which -- with the direction-signed charge -- add constructively, and both
beams must see the same cavity. ``MainloopCounterRotatingBeams`` tracks
``beams[0]`` through the elements in forward order and ``beams[1]`` in
reverse order, so each station's feedback is tracked once per beam per turn
at the beams' true (generally offset) arrival times.

Two regimes, cleanly split by the station azimuth:

* **Offset passages** (any even section count with the half-drift / station /
  half-drift layout: the stations sit away from the beams' meeting points,
  and the two arrivals at each station are ``T_rev / 2`` apart). The
  feedback's per-passage grid machinery handles the alternating arrivals
  natively: each ``_track`` spans exactly the half turn to that beam's next
  passage, the envelope paces at the physical rate and carries each beam's
  loading into the other's passage. :class:`TestTwoBeamOffsetPassages` pins
  this against the two-beam multi-pass convolution reference.

* **Simultaneous passages** (a station at a meeting azimuth, e.g. the single
  mid-ring station of a one-section layout). The per-passage machinery would
  silently serialize the two coincident arrivals one full projection window
  apart -- the envelope then runs at twice the physical rate and the summed
  loading is wrong (measured ~47 % L2 on the first turn, decaying to ~10 %).
  The feedback detects this and raises ``NotImplementedError``
  (:class:`TestSimultaneousPassageGuard`); integrating two coincident
  currents (deposit-sum + envelope rewind) is a known open extension.

The convolution reference for two counter-rotating beams needs the
counter-rotating shunt of an *asymmetric* fundamental mode: ``R_CR = -R``
(the parameter is the shunt the counter-rotating witness experiences, its
direction sign included -- the sign is a property of the mode's field
symmetry; the beam-current signs are carried by the signed charges).
"""

import unittest
import warnings
from unittest import mock

import numpy as np

from blond import (
    Beam,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    mu_minus,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.impedances.solvers import MultiPassResonatorSolver

# Import the reference *module* (not the class) -- binding the TestCase class
# to a module-level name would make pytest re-collect its whole suite here.
from . import test_mtw_vs_nondriven_feedback as _mtw_reference
from .support import rel_err

N_SLICES = 1024
N_TURNS = 3


def _base():
    """
    Return the reference machinery class via attribute access.

    A module-level binding of the ``unittest.TestCase`` subclass would make
    pytest re-collect its whole suite under this module.

    Returns
    -------
    type
        ``TestMultiTurnFeedbackVsConvolution``.
    """
    return _mtw_reference.TestMultiTurnFeedbackVsConvolution


def _build_two_beam_simulation(
    n_sections: int,
    mode: str,
    allow_delta_t_zero: bool = False,
    acceleration: bool = False,
    fast_ramp: bool = False,
    n_turns: int = N_TURNS,
    delta_omega_rf: float = 0.0,
):
    """
    Build the two-beam ring: half-drift / station / half-drift per section.

    Parameters
    ----------
    n_sections
        Number of RF stations per turn.
    mode
        ``"fb"`` (feedback with beam), ``"fb_reference"`` (feedback, both
        beams at zero intensity) or ``"mtw"`` (multi-pass convolution
        wakefield with an asymmetric-fundamental-mode counter-rotating
        shunt).
    allow_delta_t_zero
        ``"mtw"`` only: passed to the ``MultiPassResonatorSolver``, allowing
        two beams to deposit at the same reference time (needed for a
        meeting-azimuth station where both arrive simultaneously).
    acceleration
        If True, use the reference module's accelerating magnetic cycle
        (the reference energy is raised at every RF station each turn) so
        t_rev and the RF carrier slip turn over turn. Default False keeps
        the static cycle of the offset-passage baseline.
    fast_ramp
        If True, use the transition-adjacent fast frame-slip regime
        (implies acceleration): the injection energy and per-turn gain of
        ``TestMultiTurnFeedbackVsConvolution``. The ``"mtw"`` convolution
        then retunes on resonance each pass (``delta_f = 0``), the two-beam
        counterpart of the single-beam retuning convolution.
    n_turns
        Number of turns the accelerating cycle is built for (must match the
        run length). Ignored for the static cycle.
    delta_omega_rf
        Static RF-frequency offset [rad/s] applied to the feedback stations
        (``station.delta_omega_rf``) from turn 0. The ``"mtw"`` convolution
        follows the actual RF through a static resonator retuning
        ``delta_f = delta_omega_rf / (2 pi)`` (the mapping the single-beam
        ``delta_omega_rf`` tests validate). Mutually exclusive with
        ``fast_ramp`` here. Default 0.0 leaves the baseline unchanged.

    Returns
    -------
    sim
        The Simulation.
    beams
        ``(mu_plus co-rotating, mu_minus counter-rotating)``.
    collected
        Per section, the RF station (feedback modes) or the WakeField.
    """
    harmonic, t_rf = _base()._calc_multiturn_harmonic_and_t_rf(
        n_sections, fast_ramp=fast_ramp
    )
    energy, _ = _base()._regime(fast_ramp)
    half_drift = _base().MULTITURN_CIRCUMFERENCE / n_sections / 2

    # ``mtw`` solver options: on the fast ramp retune on resonance each pass
    # (delta_f = 0), mirroring the single-beam fast-ramp convolution
    # reference and the feedback's always-on-resonance cavity. On the static
    # cycle no retuning (delta_f absent) reproduces the baseline exactly.
    solver_kwargs = {
        "decay_fraction_threshold": 1e-12,
        "allow_delta_t_zero": allow_delta_t_zero,
    }
    if fast_ramp:
        solver_kwargs["delta_f"] = 0.0
    # A static RF-frequency offset enters the convolution reference as a
    # resonator retuning delta_f = delta_omega_rf / (2 pi) (the same mapping
    # the single-beam delta_omega_rf tests validate against this solver), and
    # the feedback side as ``station.delta_omega_rf`` below. Not combined with
    # fast_ramp here (the two frequency mechanisms are tested separately).
    if delta_omega_rf != 0.0:
        solver_kwargs["delta_f"] = delta_omega_rf / (2 * np.pi)

    ring = Ring(
        circumference=_base().MULTITURN_CIRCUMFERENCE,
        check_section_indices=False,
    )
    elements = []
    collected = []
    for sec in range(n_sections):
        profile = _mtw_reference.make_noisy_profile(
            t_rf, N_SLICES, section_index=sec
        )
        profile.active = False  # frozen histogram drives both beams

        if mode == "mtw":
            wakefield = WakeField(
                sources=(
                    Resonators(
                        shunt_impedances=(
                            _base().MULTITURN_R_OVER_Q * _base().MULTITURN_Q_L
                        ),
                        center_frequencies=1.0 / t_rf,
                        quality_factors=_base().MULTITURN_Q_L,
                        # Asymmetric fundamental mode: R_CR = -R (the
                        # parameter is the shunt the counter-rotating
                        # witness experiences, its direction sign included;
                        # sources and kicks carry the signed charges).
                        shunt_impedances_counter_witness=(
                            -_base().MULTITURN_R_OVER_Q * _base().MULTITURN_Q_L
                        ),
                    ),
                ),
                solver=MultiPassResonatorSolver(**solver_kwargs),
                profile=profile,
            )
            station = SingleHarmonicRFStation(
                voltage=_base().MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=harmonic,
                local_wakefield=wakefield,
                profile=profile,
                section_index=sec,
            )
            collected.append(wakefield)
        else:
            feedback = IQCavityFeedbackTimingClass(
                profile=profile,
                R_over_Q=_base().MULTITURN_R_OVER_Q,
                Q_L=_base().MULTITURN_Q_L,
                generator_current_bias=0.0,
                n_cavities=1,
                initial_voltage=_base().MULTITURN_V_DESIGN,
                n_rf_periods_per_coarse_grid=1,
                delta_omega=0.0,
            )
            station = SingleHarmonicRFStation(
                voltage=_base().MULTITURN_V_DESIGN,
                phi_rf=0.0,
                harmonic=harmonic,
                cavity_feedback=feedback,
                profile=profile,
                section_index=sec,
            )
            # RF-frequency offset on every feedback station, before the run so
            # it is active from turn 0 (matches the mtw resonator's static
            # delta_f). Set pre-run, while the station count is not yet known,
            # so the multi-station setter guard does not fire; suppress its
            # once-per-turn advisory warning.
            if delta_omega_rf != 0.0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    station.delta_omega_rf = delta_omega_rf
            collected.append(station)

        elements += [
            DriftSimple(
                orbit_length=half_drift,
                momentum_compaction_factor=_base().MULTITURN_ALPHA_P,
                section_index=sec,
            ),
            station,
            DriftSimple(
                orbit_length=half_drift,
                momentum_compaction_factor=_base().MULTITURN_ALPHA_P,
                section_index=sec,
            ),
        ]

    ring.add_elements(elements, reorder=False)
    sim = Simulation(
        ring=ring,
        magnetic_cycle=_base()._multiturn_cycle(
            n_sections, acceleration, fast_ramp=fast_ramp, n_turns=n_turns
        ),
    )

    intensity = 0.0 if mode == "fb_reference" else _base().MULTITURN_INTENSITY
    beam = Beam(intensity=intensity, particle_type=mu_plus)
    beam.reference.total_energy = energy
    beam.setup_beam(dt=np.array([]), dE=np.array([]))
    beam_cr = Beam(
        intensity=intensity,
        particle_type=mu_minus,
        is_counter_rotating=True,
    )
    beam_cr.reference.total_energy = energy
    beam_cr.setup_beam(dt=np.array([]), dE=np.array([]))

    return sim, (beam, beam_cr), collected


def _run_two_beam_case(
    n_sections: int,
    mode: str,
    allow_delta_t_zero: bool = False,
    acceleration: bool = False,
    fast_ramp: bool = False,
    n_turns: int = N_TURNS,
    delta_omega_rf: float = 0.0,
) -> list:
    """
    Run a two-beam Simulation and collect a voltage per turn per section.

    Parameters
    ----------
    n_sections
        Number of RF stations per turn.
    mode
        See :func:`_build_two_beam_simulation`.
    allow_delta_t_zero
        See :func:`_build_two_beam_simulation`.
    acceleration
        See :func:`_build_two_beam_simulation`.
    fast_ramp
        See :func:`_build_two_beam_simulation`.
    n_turns
        Number of turns to simulate (and to build the accelerating cycle
        for). Default is the module-level ``N_TURNS``.
    delta_omega_rf
        See :func:`_build_two_beam_simulation`.

    Returns
    -------
    list
        ``[turn][section]`` voltage arrays: the wakefield induced voltage for
        ``"mtw"``, the station gap voltage otherwise -- each reflecting the
        state after the LAST beam passage of that turn at that station.
    """
    sim, beams, collected = _build_two_beam_simulation(
        n_sections,
        mode,
        allow_delta_t_zero=allow_delta_t_zero,
        acceleration=acceleration,
        fast_ramp=fast_ramp,
        n_turns=n_turns,
        delta_omega_rf=delta_omega_rf,
    )
    per_turn = []

    def collect(simulation, beam_in_callback):
        if mode == "mtw":
            per_turn.append(
                [
                    copy_to_cpu(wakefield.induced_voltage)
                    for wakefield in collected
                ]
            )
        else:
            per_turn.append(
                [
                    copy_to_cpu(station.calc_gap_voltage_with_feedbacks())
                    for station in collected
                ]
            )

    with warnings.catch_warnings():
        # The counter-rotating mainloop itself warns "Untested code" and
        # about callbacks receiving only the first beam; both are expected.
        warnings.simplefilter("ignore")
        sim.run_simulation(
            beams, n_turns=n_turns, callbacks=collect, show_progressbar=False
        )
    return per_turn


class TestTwoBeamOffsetPassages(unittest.TestCase):
    """
    Offset (interleaved) two-beam passages: feedback vs convolution.

    Two sections: each station sees mu+ and mu- ``T_rev / 2`` apart -- the
    true arrival pattern of counter-rotating beams in the symmetric ring.
    The feedback's beam-induced part (two-beam gap voltage minus the two-beam
    zero-intensity reference, by linearity) must match the two-beam
    multi-pass convolution at every station and turn. Measured floors are
    0.13 % (turn 0) falling to 0.04 %; the gate is 0.5 %.
    """

    _cache: dict = {}

    @classmethod
    def _two_beam(cls, mode: str) -> list:
        """
        Run (once per mode) and cache the two-section two-beam case.

        Parameters
        ----------
        mode
            See :func:`_build_two_beam_simulation`.

        Returns
        -------
        list
            ``[turn][section]`` collected voltage arrays.
        """
        if mode not in cls._cache:
            cls._cache[mode] = _run_two_beam_case(2, mode)
        return cls._cache[mode]

    def test_feedback_matches_two_beam_convolution(self):
        """
        The two-beam feedback matches the two-beam convolution per station.

        This pins the empirically-correct interleaved regime: the grid paces
        at the physical rate (two half-turn passages per turn) and each
        beam's carried loading appears in the other's passage.
        """
        convolution = self._two_beam("mtw")
        gap_beam = self._two_beam("fb")
        gap_reference = self._two_beam("fb_reference")

        for turn_i in range(N_TURNS):
            for sec_i in range(2):
                induced = (
                    gap_beam[turn_i][sec_i] - gap_reference[turn_i][sec_i]
                )
                self.assertLess(
                    rel_err(induced, convolution[turn_i][sec_i]),
                    0.005,
                    f"turn {turn_i} section {sec_i}",
                )

    def test_two_beam_loading_exceeds_single_beam(self):
        """
        Guard against a vacuous comparison: the second beam adds loading.

        The two-beam convolution voltage must differ from the single-beam
        run of the same ring (cached by the reference module) by well more
        than the comparison gate -- otherwise the equality above could hold
        with the counter-rotating beam silently ignored.
        """
        convolution_two_beam = self._two_beam("mtw")
        convolution_one_beam = _base()._run_multiturn_case("mtw", 2, False)

        last = N_TURNS - 1
        difference = rel_err(
            convolution_two_beam[last][0], convolution_one_beam[last][0]
        )
        self.assertGreater(difference, 0.02)

    def test_both_stations_carry_comparable_loading(self):
        """
        Both stations see the full two-beam loading (symmetric ring).

        The per-station peak voltages agree to a few percent (the profiles
        differ only by their per-section noise seed).
        """
        convolution = self._two_beam("mtw")
        peak_0 = float(np.max(np.abs(convolution[-1][0])))
        peak_1 = float(np.max(np.abs(convolution[-1][1])))
        self.assertGreater(peak_0, 0.0)
        self.assertLess(abs(peak_1 - peak_0) / peak_0, 0.05)


# Turns for the accelerating two-beam runs: five (matching the reference
# module's ``FAST_N_TURNS``) gives the per-turn drift check real teeth --
# on the fast frame-slip ramp a sign/ordering error in the accel frame-slip
# correction composed with the reverse traversal would grow several
# percentage points per turn, so five turns make it unmistakable.
ACCEL_N_TURNS = 5

# Static RF-frequency offset for the two-beam delta_omega_rf coverage (the
# same 2000 rad/s the single-beam multi-section delta_omega_rf test uses), run
# over five turns so any per-turn slip-anchor error accumulates visibly.
DELTA_OMEGA_RF = 2.0e3
DELTA_OMEGA_RF_N_TURNS = 5


class TestTwoBeamAcceleratingOffsetPassages(unittest.TestCase):
    """
    Offset two-beam passages under acceleration: feedback vs convolution.

    Coverage for the composition never exercised by
    :class:`TestTwoBeamOffsetPassages` (static cycle only): the single-beam
    accelerating multi-section frame-slip correction (which removes, before
    seeding the forward segment, the carried-envelope phase error the *other*
    stations' mid-turn grid re-seeding accrues under the slipping RF carrier)
    composed with the per-beam **reverse** traversal of the two-beam
    counter-rotating mainloop.

    Two sections on the transition-adjacent fast ramp (~4 GeV, gamma_t ~ 31:
    the RF frame slips ~0.09 t_rf per turn, orders of magnitude more than at
    the 63 GeV operating point) -- an even section count so the stations sit
    off the beams' meeting azimuths and each station still sees the two
    arrivals ``T_rev / 2`` apart. The feedback's beam-induced part (two-beam
    gap voltage minus the two-beam zero-intensity reference, by linearity of
    the cavity equation, which also cancels the common acceleration kick)
    must track the two-beam multi-pass **retuning** convolution
    (``delta_f = 0``, the accelerating counterpart of the static reference)
    with no per-turn-growing error.

    Measured (deterministic, both sections identical): 0.13 % on turn 0
    falling to 0.025 % on turn 4 -- the relative error *shrinks* as the
    carried wake accumulates (the discretization floor stays roughly
    constant while the beam loading builds up), the opposite of the ramping
    signature a mis-composed frame-slip correction would show. A sign or
    ordering error in that composition produced multi-percent-per-turn
    growth in the single-beam multi-section case (turn 4 reached ~29 % /
    ~57 % for 2 / 4 sections without the correction).
    """

    _cache: dict = {}

    @classmethod
    def _accel_two_beam(cls, mode: str) -> list:
        """
        Run (once per mode) and cache the accelerating two-section case.

        Parameters
        ----------
        mode
            See :func:`_build_two_beam_simulation`.

        Returns
        -------
        list
            ``[turn][section]`` collected voltage arrays.
        """
        if mode not in cls._cache:
            cls._cache[mode] = _run_two_beam_case(
                2,
                mode,
                acceleration=True,
                fast_ramp=True,
                n_turns=ACCEL_N_TURNS,
            )
        return cls._cache[mode]

    @classmethod
    def _beam_induced_per_turn(cls) -> list:
        """
        Feedback beam-induced voltage ``[turn][section]`` under the ramp.

        The two-beam beam run's gap voltage minus the two-beam
        zero-intensity reference, isolating the beam-induced part by
        linearity (and cancelling the common acceleration kick).

        Returns
        -------
        list
            ``[turn][section]`` beam-induced voltage arrays.
        """
        gap_beam = cls._accel_two_beam("fb")
        gap_reference = cls._accel_two_beam("fb_reference")
        return [
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

    def test_accel_feedback_matches_two_beam_convolution(self):
        """
        The accelerating two-beam feedback matches the two-beam convolution.

        Pins the accel frame-slip x reverse-traversal composition: on the
        fast ramp the coarse grid is rebuilt each turn from backfill segments
        across the other station (each re-seeded at its own past-station RF
        frequency), and the two counter-rotating beams traverse the ring in
        opposite element orders. The beam-induced gap voltage must still
        match the retuning convolution at every station and turn, to the
        same order of tolerance as the static offset-passage test (0.5 %;
        measured max ~0.13 %).
        """
        convolution = self._accel_two_beam("mtw")
        induced_per_turn = self._beam_induced_per_turn()

        # Non-degenerate: the carried wake carries real voltage by the last
        # turn (guards against a silently-zeroed comparison).
        self.assertGreater(float(np.max(np.abs(convolution[-1][0]))), 0.0)

        for turn_i in range(ACCEL_N_TURNS):
            for sec_i in range(2):
                self.assertLess(
                    rel_err(
                        induced_per_turn[turn_i][sec_i],
                        convolution[turn_i][sec_i],
                    ),
                    0.005,
                    f"turn {turn_i} section {sec_i}",
                )

    def test_accel_error_does_not_grow_per_turn(self):
        """
        The carried two-beam wake error stays bounded, not ramping per turn.

        The structural failure the static offset-passage test cannot catch:
        a sign or ordering error in the accel frame-slip correction composed
        with the reverse traversal would leave a per-turn-growing phase error
        in the carried two-beam wake. This fits the worst-section relative
        error against the turn number (skipping the turn-0 single-pass
        transient) and requires a non-positive-trending slope, plus a
        last-turn error no larger than turn 0 -- i.e. the error does not grow
        as the wake is carried turn over turn.

        Measured slope over turns 1..4 is ~-0.011 pp/turn (the error
        *shrinks*); a mis-composed correction would instead ramp at several
        percentage points per turn.
        """
        convolution = self._accel_two_beam("mtw")
        induced_per_turn = self._beam_induced_per_turn()

        worst_per_turn = np.array(
            [
                max(
                    rel_err(
                        induced_per_turn[turn_i][sec_i],
                        convolution[turn_i][sec_i],
                    )
                    for sec_i in range(2)
                )
                for turn_i in range(ACCEL_N_TURNS)
            ]
        )

        # Bounded: every turn stays within the static-order gate.
        self.assertLess(
            float(worst_per_turn.max()),
            0.005,
            f"unbounded error {worst_per_turn}",
        )

        # Non-growing: fit the post-transient trend (turns 1..end). A real
        # frame-slip x reverse-traversal composition error ramps positively;
        # the healthy behaviour is flat-to-decreasing.
        turns = np.arange(ACCEL_N_TURNS)
        slope_pp_per_turn = np.polyfit(
            turns[1:], 100.0 * worst_per_turn[1:], 1
        )[0]
        self.assertLess(
            slope_pp_per_turn,
            0.05,
            f"per-turn error ramps {slope_pp_per_turn:.4g} pp/turn "
            f"(worst_per_turn={worst_per_turn})",
        )

        # Explicitly non-ramping: the carried-wake error at the last turn is
        # no larger than the turn-0 single-pass floor.
        self.assertLessEqual(
            float(worst_per_turn[-1]),
            float(worst_per_turn[0]),
            f"error grew across the run {worst_per_turn}",
        )


class TestTwoBeamDeltaOmegaRfOffsetPassages(unittest.TestCase):
    """
    Offset two-beam passages with an RF-frequency offset: feedback vs conv.

    The last untested corner of the demodulation slip anchor. The
    ``delta_omega_rf`` demod-anchoring (design-clock coarse grid plus the
    accumulated slip ``-(delta_phi_rf + live gap)`` carried as a constant
    demodulation phase) is validated for the *forward* stream both alone
    (``test_multiturn_delta_omega_rf_*``) and, here, its lone counter-rotating
    equivalent -- but a lone beam is tracked *forward* by ``MainloopSingleBeam``
    regardless of direction. Only the *two-beam*
    ``MainloopCounterRotatingBeams`` walks the counter-rotating beam through the
    ring in **reverse** element order, so whether the slip anchor carries the
    right sign/value for the reverse stream is exercised for the first time
    here.

    Two sections on the **static** cycle (no acceleration -- the frame slip is
    purely the RF-frequency offset, isolating it from the ramp frame-slip that
    :class:`TestTwoBeamAcceleratingOffsetPassages` covers) with
    ``delta_omega_rf = 2000`` rad/s. Even section count keeps the stations off
    the meeting azimuths (arrivals ``T_rev / 2`` apart). The feedback's
    beam-induced part (two-beam gap voltage minus the two-beam zero-intensity
    reference, both with the offset, so the offset's rotation of the empty-
    cavity voltage cancels by linearity) must track the two-beam multi-pass
    **retuning** convolution (``delta_f = delta_omega_rf / (2 pi)``) with no
    per-turn-growing error. A reverse-stream slip-anchor sign/value error would
    leave a per-turn-growing phase error in the carried two-beam wake that the
    lone-beam and zero-offset tests structurally cannot catch.
    """

    _cache: dict = {}

    @classmethod
    def _dw_two_beam(cls, mode: str) -> list:
        """
        Run (once per mode) and cache the offset two-section case.

        Parameters
        ----------
        mode
            See :func:`_build_two_beam_simulation`.

        Returns
        -------
        list
            ``[turn][section]`` collected voltage arrays.
        """
        if mode not in cls._cache:
            cls._cache[mode] = _run_two_beam_case(
                2,
                mode,
                delta_omega_rf=DELTA_OMEGA_RF,
                n_turns=DELTA_OMEGA_RF_N_TURNS,
            )
        return cls._cache[mode]

    @classmethod
    def _beam_induced_per_turn(cls) -> list:
        """
        Feedback beam-induced voltage ``[turn][section]`` with the offset.

        The two-beam beam run's gap voltage minus the two-beam zero-intensity
        reference, isolating the beam-induced part by linearity (and
        cancelling the offset's common rotation of the empty-cavity voltage).

        Returns
        -------
        list
            ``[turn][section]`` beam-induced voltage arrays.
        """
        gap_beam = cls._dw_two_beam("fb")
        gap_reference = cls._dw_two_beam("fb_reference")
        return [
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

    def test_delta_omega_rf_feedback_matches_two_beam_convolution(self):
        """
        The offset two-beam feedback matches the two-beam retuning conv.

        Pins the reverse-stream slip anchor: both counter-rotating beams
        traverse the ring in opposite element orders while the demodulation is
        anchored to the station's accumulated ``delta_omega_rf`` slip. The
        beam-induced gap voltage must still match the retuning convolution at
        every station and turn, to the same order of tolerance as the static
        offset-passage test (0.5 %).
        """
        convolution = self._dw_two_beam("mtw")
        induced_per_turn = self._beam_induced_per_turn()

        # Non-degenerate: the carried wake carries real voltage by the last
        # turn (guards against a silently-zeroed comparison).
        self.assertGreater(float(np.max(np.abs(convolution[-1][0]))), 0.0)

        for turn_i in range(DELTA_OMEGA_RF_N_TURNS):
            for sec_i in range(2):
                self.assertLess(
                    rel_err(
                        induced_per_turn[turn_i][sec_i],
                        convolution[turn_i][sec_i],
                    ),
                    0.005,
                    f"turn {turn_i} section {sec_i}",
                )

    def test_delta_omega_rf_error_does_not_grow_per_turn(self):
        """
        The offset two-beam wake error stays bounded, not ramping per turn.

        The structural failure the lone-beam and zero-offset tests cannot
        catch: a sign or value error in the reverse-stream slip anchor would
        leave a per-turn-growing phase error in the carried two-beam wake as
        the accumulated ``delta_omega_rf`` slip builds up. Fits the
        worst-section relative error against the turn number (skipping the
        turn-0 single-pass transient) and requires a non-positive-trending
        slope, plus a last-turn error no larger than turn 0.
        """
        convolution = self._dw_two_beam("mtw")
        induced_per_turn = self._beam_induced_per_turn()

        worst_per_turn = np.array(
            [
                max(
                    rel_err(
                        induced_per_turn[turn_i][sec_i],
                        convolution[turn_i][sec_i],
                    )
                    for sec_i in range(2)
                )
                for turn_i in range(DELTA_OMEGA_RF_N_TURNS)
            ]
        )

        # Bounded: every turn stays within the static-order gate.
        self.assertLess(
            float(worst_per_turn.max()),
            0.005,
            f"unbounded error {worst_per_turn}",
        )

        # Non-growing: a reverse-stream slip-anchor error ramps positively as
        # the accumulated offset slip builds; the healthy behaviour is
        # flat-to-decreasing over the post-transient turns.
        turns = np.arange(DELTA_OMEGA_RF_N_TURNS)
        slope_pp_per_turn = np.polyfit(
            turns[1:], 100.0 * worst_per_turn[1:], 1
        )[0]
        self.assertLess(
            slope_pp_per_turn,
            0.05,
            f"per-turn error ramps {slope_pp_per_turn:.4g} pp/turn "
            f"(worst_per_turn={worst_per_turn})",
        )

        # Explicitly non-ramping: last-turn error no larger than turn 0.
        self.assertLessEqual(
            float(worst_per_turn[-1]),
            float(worst_per_turn[0]),
            f"error grew across the run {worst_per_turn}",
        )


class TestSimultaneousPassageGuard(unittest.TestCase):
    """
    A station at a meeting azimuth refuses simultaneous two-beam passages.

    With one section the single mid-ring station sits exactly at the beams'
    meeting point: both arrive at the same reference time, which the
    per-passage grid machinery would silently serialize one full projection
    window apart (envelope at twice the physical rate, ~47 % L2 waveform
    error on the first turn). The feedback must fail loudly instead.
    """

    def test_single_section_two_beam_raises(self):
        """The coincident second passage raises ``NotImplementedError``."""
        with self.assertRaises(NotImplementedError) as ctx:
            _run_two_beam_case(1, "fb")
        self.assertIn("simultaneously", str(ctx.exception))

    def test_single_section_convolution_reference_needs_delta_t_zero(self):
        """
        The convolution route for a meeting-azimuth station works with the flag.

        Both halves of the escape hatch the feedback's error message points
        to are pinned: without ``allow_delta_t_zero`` the solver's
        monotonic-clock assertion rejects the coincident passage, and *with*
        it the same two-beam convolution runs and produces finite induced
        voltage.
        """
        with self.assertRaises(AssertionError):
            _run_two_beam_case(1, "mtw")

        per_turn = _run_two_beam_case(1, "mtw", allow_delta_t_zero=True)
        self.assertEqual(len(per_turn), N_TURNS)
        peak = float(np.max(np.abs(per_turn[-1][0])))
        self.assertTrue(np.isfinite(peak))
        self.assertGreater(peak, 0.0)


class TestBackfillWalkDirectionConsistency(unittest.TestCase):
    """
    The backfill reference walk never runs with a mismatched direction.

    ``get_time_omega_array_backfill`` takes its element *order* from
    the previously tracked beam (``_last_tracked_beam_state_frwrd``, used at
    ``rf_center_grid.py:276-286`` and for the ``until_index`` at ``:342-344``)
    but hands ``beam.is_counter_rotating`` -- the *current* beam -- to
    ``track_reference``. In a single-beam run the two always agree, so the
    distinction is invisible; only a two-beam run can make them differ.

    At ``n_sections == 2`` they stay safe because the interval to backfill
    is empty: the forward projection spans one section, ``t_rev / n``, while
    the other beam reaches station ``i`` a gap ``|n - 2 i - 1| / n * t_rev``
    later. The two coincide exactly when ``|n - 2 i - 1| == 1``, which for
    ``n == 2`` holds at BOTH stations -- so the reference times match to the
    bit and the early return at ``rf_center_grid.py:617`` fires before the
    walk is entered.

    That is NOT a structural invariant of the projection endpoint: it is a
    property of two sections. ``|n - 2 i - 1|`` is odd, so beyond ``n == 2``
    only the two middle stations (``i == n / 2 - 1`` and ``i == n / 2``)
    satisfy it -- 2 of 16 for the muon-collider RCS1 layout, where the walk
    is therefore entered with a mismatched direction at 14 stations every
    turn. That is safe because the walk now derives its element order, its
    indices and its per-station section remap from the SAME carried flag
    (``backfill_is_counter_rotating``); it was not safe while the element
    order came from the carried beam and the ``track_reference`` direction
    from the current one.

    These tests pin both halves: the empty-interval early return at
    ``n == 2``, and the fact that the walk really is entered with a
    mismatch beyond it, so a future change to the projection endpoint or to
    the
    mainloop's element order cannot silently start walking the elements of
    one beam while tracking the reference of the other. Measured (static
    case): 10 of 12 backfill calls carry a direction mismatch and none
    reaches the walk; with the early return removed all 10 do.

    The invariant is pinned across all three two-beam regimes exercised in
    this module (static, accelerating fast ramp, ``delta_omega_rf``),
    because each moves the reference clock differently and the time
    equality must hold bit-exactly in every one.

    A note on why the *outcome* cannot be compared instead: forcing the
    walk to run in the empty-interval situation produces identical arrays
    for the matched and the mismatched element order, because the ring
    layout is symmetric (half-drift / station / half-drift). Only the
    "never entered" property and its exact-time-equality gate are
    observable, so that is what is asserted.
    """

    #: config name -> extra kwargs for :func:`_run_two_beam_case`.
    _configs: dict = {
        "static": {},
        "accelerating": {"acceleration": True, "fast_ramp": True},
        "delta_omega_rf": {"delta_omega_rf": DELTA_OMEGA_RF},
    }
    _records_cache: dict = {}

    @classmethod
    def _recorded_two_beam_run(cls, config: str, n_sections: int = 2) -> dict:
        """
        Run (once per config) the two-beam case with the walk instrumented.

        Parameters
        ----------
        config
            Key into ``_configs``: which two-beam regime to run.

        Returns
        -------
        dict
            ``"calls"``: per backfill call
            ``(last_tracked_direction, current_direction, time_gap)`` with
            ``time_gap = beam.reference.time - reference_state.time`` [s],
            recorded *before* the early returns run; ``"walks"``: the same
            pair for each call that actually entered the element walk.
        """
        cache_key = (config, n_sections)
        if cache_key in cls._records_cache:
            return cls._records_cache[cache_key]

        calls: list[tuple[bool | None, bool, float]] = []
        walks: list[tuple[bool | None, bool]] = []

        original_call = (
            IQCavityFeedbackTimingClass.calculate_rf_centers_for_backfill
        )
        original_walk = (
            IQCavityFeedbackTimingClass.get_time_omega_array_backfill
        )

        def recording_call(self, beam):
            calls.append(
                (
                    self._last_tracked_beam_state_frwrd,
                    beam.is_counter_rotating,
                    float(
                        beam.reference.time
                        - self._reference_state_until_tracked.time
                    ),
                )
            )
            return original_call(self, beam)

        def recording_walk(self, beam):
            walks.append(
                (self._last_tracked_beam_state_frwrd, beam.is_counter_rotating)
            )
            return original_walk(self, beam)

        with (
            mock.patch.object(
                IQCavityFeedbackTimingClass,
                "calculate_rf_centers_for_backfill",
                recording_call,
            ),
            mock.patch.object(
                IQCavityFeedbackTimingClass,
                "get_time_omega_array_backfill",
                recording_walk,
            ),
        ):
            _run_two_beam_case(n_sections, "fb", **cls._configs[config])

        cls._records_cache[cache_key] = {
            "calls": calls,
            "walks": walks,
        }
        return cls._records_cache[cache_key]

    @staticmethod
    def _mismatched(records):
        """
        Filter records where the stored and current directions differ.

        Parameters
        ----------
        records
            Tuples whose first two entries are
            ``(last_tracked_direction, current_direction)``.

        Returns
        -------
        list
            The records with ``last is not None and last != current``.
        """
        return [
            record
            for record in records
            if record[0] is not None and record[0] != record[1]
        ]

    def test_walk_is_entered_with_a_mismatch_beyond_two_sections(self):
        """
        Beyond two sections the mismatched walk IS entered, and is safe.

        The empty-interval coincidence needs ``|n - 2 i - 1| == 1``, which
        every station satisfies only at ``n_sections == 2``. At four and six
        sections most stations see the two beams a different gap apart than
        the one section the forward projection spans, so the backfill walk
        runs with ``_last_tracked_beam_state_frwrd`` different from the
        beam being tracked.

        This is the configuration the class docstring used to call
        unreachable. It is reached, and it is correct: the walk takes its
        element order, its indices and its per-station section remap from
        the one carried flag, so the run completes and the physics
        comparisons elsewhere in this module hold.
        """
        for n_sections in (4, 6):
            with self.subTest(n_sections=n_sections):
                records = self._recorded_two_beam_run("static", n_sections)
                self.assertGreater(
                    len(self._mismatched(records["calls"])),
                    0,
                    "no backfill call carried a direction mismatch",
                )
                self.assertGreater(
                    len(self._mismatched(records["walks"])),
                    0,
                    "the mismatched backfill walk was never entered at "
                    f"{n_sections} sections, so the empty-interval "
                    "coincidence is wider than the |n - 2 i - 1| == 1 "
                    "geometry predicts "
                    f"(walks: {records['walks']})",
                )

    def test_backfill_walk_never_entered_with_mismatched_direction(self):
        """
        At ``n_sections == 2`` no mismatched call reaches the walk.

        Both halves matter: the first assertion proves the two-beam run
        actually exercises the dangerous configuration (it would fail if the
        beams stopped alternating), the second proves the empty-interval
        early return holds. All three two-beam regimes are covered.

        Scoped to two sections on purpose -- see
        :meth:`test_walk_is_entered_with_a_mismatch_beyond_two_sections`
        for why that is a property of ``n_sections == 2``, not a structural
        invariant of the projection endpoint.
        """
        for config in self._configs:
            with self.subTest(config=config):
                records = self._recorded_two_beam_run(config)
                self.assertGreater(
                    len(self._mismatched(records["calls"])),
                    0,
                    "no backfill call carried a direction "
                    "mismatch, so this test no longer exercises the "
                    "two-beam configuration it guards "
                    f"(calls: {records['calls']})",
                )
                self.assertEqual(
                    self._mismatched(records["walks"]),
                    [],
                    "the backfill walk ran with the element order of one "
                    "beam and the reference direction of another; the "
                    "empty-backfill invariant no longer holds "
                    f"(walk entries: {records['walks']})",
                )

    def test_mismatched_calls_are_gated_by_exact_time_equality(self):
        """
        Every mismatched-direction call carries a bit-exact zero time gap.

        This pins the *mechanism* behind the previous test: the forward
        projection has already advanced the reference to the other beam's
        arrival time exactly, so ``beam.reference.time ==
        _reference_state_until_tracked.time`` and the early return fires
        before the walk. A merely-approximate equality (a nonzero gap below
        some tolerance) would NOT be safe -- the ``==`` in the production
        early return is exact -- so the assertion is ``gap == 0.0``, not
        ``assertAlmostEqual``.

        The companion assertion shows the gap is a live measurement, not a
        quantity that is trivially always zero: the first passage of each
        station (``last is None``, nothing projected yet) has a genuinely
        nonzero gap and legitimately walks.
        """
        for config in self._configs:
            with self.subTest(config=config):
                records = self._recorded_two_beam_run(config)
                mismatched_gaps = [
                    gap for _, _, gap in self._mismatched(records["calls"])
                ]
                self.assertGreater(len(mismatched_gaps), 0)
                self.assertEqual(
                    [gap for gap in mismatched_gaps if gap != 0.0],
                    [],
                    "a mismatched-direction call reached the backfill "
                    "machinery with a nonzero backfill interval; the "
                    "exact-time-equality gate no longer protects the "
                    "mixed-direction walk "
                    f"(all mismatched gaps: {mismatched_gaps})",
                )
                first_passage_gaps = [
                    gap for last, _, gap in records["calls"] if last is None
                ]
                self.assertTrue(
                    any(gap != 0.0 for gap in first_passage_gaps),
                    "expected the first passages (nothing projected yet) "
                    "to show a nonzero gap; the gap recording appears "
                    f"inert (first-passage gaps: {first_passage_gaps})",
                )


if __name__ == "__main__":
    unittest.main()
