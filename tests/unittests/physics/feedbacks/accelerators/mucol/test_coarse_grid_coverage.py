"""
Every cell of the coarse antenna-voltage grids is written on every passage.

:meth:`IQCavityFeedbackCoarseGrid.reset_arrays` sizes the three antenna-
voltage grids for a new passage without initialising them: the backfill
replay and the forward span together write every cell before anything
reads it. These tests pin that coverage on real multi-station tracking, by
poisoning the grids with NaN right after they are sized and checking that
no NaN survives a passage -- on both the compiled and the reference path,
open loop and with the generator loop. A self-check skips the backfill
replay and confirms the harness does find the cells left unwritten.
"""

import unittest
import warnings

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_plus,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackCoarseGrid
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)

#: RCS1-like cavity and ring, as in the sibling readout tests.
R_OVER_Q = 518.0
Q_L = 1.29e4
ALPHA_P = 10.395e-4
CIRCUMFERENCE = 5990.0
ENERGY = 63e9
HARMONIC = 2590
INTENSITY = 2.7e12
V_DESIGN = 30e6
N_SLICES = 256
N_MACROPARTICLES = 1000
N_TURNS = 2
N_STATIONS = 2

#: The three grids ``reset_arrays`` leaves uninitialised.
ANTENNA_GRIDS = (
    "antenna_voltage_coarse_grid",
    "antenna_voltage_gen_coarse_grid",
    "antenna_voltage_beam_coarse_grid",
)


class _StopTracking(Exception):
    """Ends a self-check run once it has what it needs."""


def _feedback(profile, with_controller):
    """
    Operating-point feedback on ``profile``.

    Parameters
    ----------
    profile
        The station's profile.
    with_controller
        Attach a PI generator loop.

    Returns
    -------
    feedback
        The feedback.
    """
    bias = V_DESIGN / (2.0 * R_OVER_Q * Q_L)
    controller = (
        GeneratorCurrentPIController(
            gain_proportional=1.0e-8,
            gain_integral=1.0e-10,
            generator_current_bias=bias,
        )
        if with_controller
        else None
    )
    return IQCavityFeedbackCoarseGrid(
        profile=profile,
        R_over_Q=R_OVER_Q,
        Q_L=Q_L,
        generator_current_bias=bias,
        n_cavities=1,
        initial_voltage=V_DESIGN,
        n_rf_periods_per_coarse_grid=1,
        delta_omega=0.0,
        controller=controller,
    )


def _poison_after_reset(feedback):
    """
    Make ``reset_arrays`` fill the antenna grids with NaN.

    Parameters
    ----------
    feedback
        Feedback to instrument.
    """
    reset_arrays = feedback.reset_arrays

    def poisoned(n_backfill_cells=0):
        reset_arrays(n_backfill_cells=n_backfill_cells)
        for name in ANTENNA_GRIDS:
            getattr(feedback, name)[:] = np.nan

    feedback.reset_arrays = poisoned


def _track_two_stations(
    use_kernel, with_controller, skip_backfill_replay=False
):
    """
    Track a two-station ring and count unwritten cells after each passage.

    Parameters
    ----------
    use_kernel
        Run the compiled coarse recursion (else the reference path).
    with_controller
        Attach a PI generator loop to each feedback.
    skip_backfill_replay
        Self-check: leave the backfill replay out and stop at the first
        passage that had one to replay.

    Returns
    -------
    unwritten
        One entry per checked passage: ``(n_backfill_cells, n_unwritten)``.
    """
    cycle = ConstantMagneticCycle(
        reference_particle=mu_plus, value=ENERGY, in_unit="total energy"
    )
    t_rev = cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    t_rf = t_rev / HARMONIC

    elements = []
    feedbacks = []
    unwritten = []
    for section_index in range(N_STATIONS):
        profile = StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, N_SLICES, t_rf
        )
        feedback = _feedback(profile, with_controller)
        feedback.use_numba_envelope_kernel = use_kernel
        _poison_after_reset(feedback)
        feedbacks.append(feedback)
        station = SingleHarmonicRFStation(
            voltage=V_DESIGN / N_STATIONS,
            phi_rf=0.0,
            harmonic=HARMONIC,
            cavity_feedback=feedback,
            section_index=section_index,
        )
        elements += [
            DriftSimple(
                orbit_length=CIRCUMFERENCE / N_STATIONS,
                momentum_compaction_factor=ALPHA_P,
                section_index=section_index,
            ),
            station,
        ]

    for feedback in feedbacks:
        track = feedback._track
        replay = feedback._replay_backfill_span

        def checked_track(beam, feedback=feedback, track=track):
            track(beam)
            n_unwritten = sum(
                int(np.isnan(getattr(feedback, name)).sum())
                for name in ANTENNA_GRIDS
            )
            n_backfill = len(feedback._rf_centers) - int(
                feedback._rf_centers_lengths[-1]
            )
            unwritten.append((n_backfill, n_unwritten))
            if skip_backfill_replay and n_backfill > 0:
                raise _StopTracking

        def skipped_replay(n_backfill_centers, replay=replay):
            if not skip_backfill_replay:
                replay(n_backfill_centers=n_backfill_centers)

        feedback._track = checked_track
        feedback._replay_backfill_span = skipped_replay

    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    ring.add_elements(elements, reorder=False)
    simulation = Simulation(ring=ring, magnetic_cycle=cycle)
    beam = Beam(intensity=INTENSITY, particle_type=mu_plus)
    beam.reference.total_energy = ENERGY
    simulation.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            n_macroparticles=N_MACROPARTICLES,
            sigma_dt=0.06 * t_rf,
            sigma_dE=1.5e7,
            seed=7,
            reinsertion=True,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            simulation.run_simulation(
                (beam,), n_turns=N_TURNS, show_progressbar=False
            )
        except _StopTracking:
            pass
    return unwritten


class TestEveryAntennaCellIsWrittenEachPassage(unittest.TestCase):
    """No cell of a freshly sized antenna grid survives a passage unset."""

    def test_every_cell_is_written(self):
        for use_kernel in (True, False):
            for with_controller in (False, True):
                with self.subTest(
                    use_kernel=use_kernel, with_controller=with_controller
                ):
                    unwritten = _track_two_stations(
                        use_kernel, with_controller
                    )
                    self.assertEqual(len(unwritten), N_TURNS * N_STATIONS)
                    self.assertTrue(
                        any(n_backfill > 0 for n_backfill, _ in unwritten),
                        msg="no passage replayed a backfill span",
                    )
                    self.assertEqual(
                        [n_unwritten for _, n_unwritten in unwritten],
                        [0] * len(unwritten),
                    )

    def test_the_harness_finds_cells_left_unwritten(self):
        unwritten = _track_two_stations(
            use_kernel=True, with_controller=False, skip_backfill_replay=True
        )
        n_backfill, n_unwritten = unwritten[-1]
        self.assertGreater(n_backfill, 0)
        # Each of the three grids keeps every skipped backfill cell unset.
        self.assertGreaterEqual(n_unwritten, 3 * n_backfill)


if __name__ == "__main__":
    unittest.main()
