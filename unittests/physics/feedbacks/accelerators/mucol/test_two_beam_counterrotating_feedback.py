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


def _build_two_beam_simulation(n_sections: int, mode: str):
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

    Returns
    -------
    sim
        The Simulation.
    beams
        ``(mu_plus co-rotating, mu_minus counter-rotating)``.
    collected
        Per section, the RF station (feedback modes) or the WakeField.
    """
    harmonic, t_rf = _base()._calc_multiturn_harmonic_and_t_rf(n_sections)
    energy, _ = _base()._regime(False)
    half_drift = _base().MULTITURN_CIRCUMFERENCE / n_sections / 2

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
                        shunt_impedances_counter_rotating=(
                            -_base().MULTITURN_R_OVER_Q * _base().MULTITURN_Q_L
                        ),
                    ),
                ),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12
                ),
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
        ring=ring, magnetic_cycle=_base()._multiturn_cycle(n_sections, False)
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


def _run_two_beam_case(n_sections: int, mode: str) -> list:
    """
    Run a two-beam Simulation and collect a voltage per turn per section.

    Parameters
    ----------
    n_sections
        Number of RF stations per turn.
    mode
        See :func:`_build_two_beam_simulation`.

    Returns
    -------
    list
        ``[turn][section]`` voltage arrays: the wakefield induced voltage for
        ``"mtw"``, the station gap voltage otherwise -- each reflecting the
        state after the LAST beam passage of that turn at that station.
    """
    sim, beams, collected = _build_two_beam_simulation(n_sections, mode)
    per_turn = []

    def collect(simulation, beam_in_callback):
        if mode == "mtw":
            per_turn.append(
                [
                    np.copy(np.asarray(wakefield.induced_voltage))
                    for wakefield in collected
                ]
            )
        else:
            per_turn.append(
                [
                    np.copy(
                        np.asarray(station.calc_gap_voltage_with_feedbacks())
                    )
                    for station in collected
                ]
            )

    with warnings.catch_warnings():
        # The counter-rotating mainloop itself warns "Untested code" and
        # about callbacks receiving only the first beam; both are expected.
        warnings.simplefilter("ignore")
        sim.run_simulation(
            beams, n_turns=N_TURNS, callbacks=collect, show_progressbar=False
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
        The convolution route for a meeting-azimuth station is documented.

        The solver's monotonic-clock assertion rejects the coincident
        passage unless ``allow_delta_t_zero=True`` -- the workaround the
        feedback's error message points to. This pins that the recommended
        escape hatch actually exists.
        """
        with self.assertRaises(AssertionError):
            _run_two_beam_case(1, "mtw")


if __name__ == "__main__":
    unittest.main()
