from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

import blond
from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    DynamicProfileConstNBins,
    InductiveImpedance,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.experimental.simulation.warmup import warmup
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import (
    ContinuousMultiTurnTimeDomainSolver,
    InductiveImpedanceSolver,
)

resonator_data = np.loadtxt(
    os.path.join(
        os.path.dirname(blond.__file__),
        "examples/scripts/resources/EX_05_new_HQ_table.txt",
    ),
    comments="!",
)
R_SHUNT = resonator_data[:, 2] * 10**6
F_RES = resonator_data[:, 0] * 10**9
Q_FACTOR = resonator_data[:, 1] * 100


class TestWarmup(unittest.TestCase):
    def setUp(self):
        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicRFStation()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf_design = 0

        n_turns = 100
        magnetic_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 460e9, n_turns),
            reference_particle=proton,
            in_unit="total energy",
        )

        drift1 = DriftSimple(orbit_length=26658.883)
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        profile1 = DynamicProfileConstNBins(n_bins=50)
        wakefield1 = WakeField(
            sources=(InductiveImpedance(Z_over_n=1.0),),
            solver=InductiveImpedanceSolver(),
            profile=profile1,
        )

        # A profile not attached to any WakeField - just an observability
        # element in the ring.
        standalone_profile = StaticProfile(
            cut_left=0, cut_right=12e-9, n_bins=20
        )

        beam1 = Beam(intensity=1e9, particle_type=proton)
        beam1.setup_beam(
            dt=np.linspace(1e-9, 10e-9, 100),
            dE=np.linspace(-1e6, 1e6, 100),
            reference_time=0,
            reference_total_energy=450e9,
        )
        self.simulation = Simulation.from_locals(locals())
        self.simulation._beams = (beam1,)
        self.beam = beam1
        self.wakefield = wakefield1
        self.profile = profile1
        self.standalone_profile = standalone_profile

    def test_warmup_bunch_shape_unchanged(self):
        dt_before = copy_to_cpu(self.beam._dt.array_local.copy())
        dE_before = copy_to_cpu(self.beam._dE.array_local.copy())
        flags_before = copy_to_cpu(self.beam._flags.array_local.copy())
        ids_before = copy_to_cpu(self.beam._ids.array_local.copy())

        warmup(
            self.simulation,
            self.beam,
            n_turns=5,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_array_equal(
            copy_to_cpu(self.beam._dt.array_local), dt_before
        )
        np.testing.assert_array_equal(
            copy_to_cpu(self.beam._dE.array_local), dE_before
        )
        np.testing.assert_array_equal(
            copy_to_cpu(self.beam._flags.array_local), flags_before
        )
        np.testing.assert_array_equal(
            copy_to_cpu(self.beam._ids.array_local), ids_before
        )

    def test_warmup_reference_unchanged_after_call(self):
        # `finalize()` synchronizes `beam.reference.total_energy` with the
        # magnetic cycle's turn-0 value the first time it runs (independent
        # of `warmup()`); capture "before" after that has already happened,
        # matching what `warmup()` itself sees internally.
        self.simulation.finalize(beams=(self.beam,), n_turns=5)
        time_before = self.beam.reference.time
        total_energy_before = self.beam.reference.total_energy

        warmup(
            self.simulation,
            self.beam,
            n_turns=5,
            show_progressbar=False,
            verbose=False,
        )

        self.assertEqual(self.beam.reference.time, time_before)
        self.assertEqual(self.beam.reference.total_energy, total_energy_before)

    def test_warmup_turn_and_section_counter_unchanged(self):
        turn_before = self.simulation.turn_counter.value
        section_before = self.simulation.section_counter.value

        warmup(
            self.simulation,
            self.beam,
            n_turns=7,
            show_progressbar=False,
            verbose=False,
        )

        self.assertEqual(self.simulation.turn_counter.value, turn_before)
        self.assertEqual(self.simulation.section_counter.value, section_before)

    def test_warmup_profile_not_recomputed_after_first_turn(self):
        # The number of `profile.track()` calls made while warming up for a
        # single turn should not grow with `n_turns` - the profile is only
        # computed on the first warmup turn, then held static.
        original_active = self.profile.active

        # Patch the instance's own `track`, not the shared class method -
        # multiple `ProfileBaseClass` instances exist in this fixture, and
        # patching the class would redirect every instance's `.track()`
        # call to whichever instance's bound method was used as `wraps`.
        with patch.object(
            self.profile, "track", wraps=self.profile.track
        ) as tracked:
            warmup(
                self.simulation,
                self.beam,
                n_turns=1,
                show_progressbar=False,
                verbose=False,
            )
        calls_for_one_turn = tracked.call_count

        with patch.object(
            self.profile, "track", wraps=self.profile.track
        ) as tracked:
            warmup(
                self.simulation,
                self.beam,
                n_turns=20,
                show_progressbar=False,
                verbose=False,
            )
        calls_for_many_turns = tracked.call_count

        self.assertGreater(calls_for_one_turn, 0)
        self.assertEqual(calls_for_many_turns, calls_for_one_turn)
        self.assertEqual(self.profile.active, original_active)

    def test_warmup_standalone_profile_not_recomputed_after_first_turn(self):
        # A profile that isn't attached to any WakeField must also be
        # frozen after the first warmup turn.
        original_active = self.standalone_profile.active

        with patch.object(
            self.standalone_profile,
            "track",
            wraps=self.standalone_profile.track,
        ) as tracked:
            warmup(
                self.simulation,
                self.beam,
                n_turns=1,
                show_progressbar=False,
                verbose=False,
            )
        calls_for_one_turn = tracked.call_count

        with patch.object(
            self.standalone_profile,
            "track",
            wraps=self.standalone_profile.track,
        ) as tracked:
            warmup(
                self.simulation,
                self.beam,
                n_turns=20,
                show_progressbar=False,
                verbose=False,
            )
        calls_for_many_turns = tracked.call_count

        self.assertGreater(calls_for_one_turn, 0)
        self.assertEqual(calls_for_many_turns, calls_for_one_turn)
        self.assertEqual(self.standalone_profile.active, original_active)

    def test_warmup_zero_turns_is_noop(self):
        dt_before = copy_to_cpu(self.beam._dt.array_local.copy())
        turn_before = self.simulation.turn_counter.value

        warmup(
            self.simulation,
            self.beam,
            n_turns=0,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_array_equal(
            copy_to_cpu(self.beam._dt.array_local), dt_before
        )
        self.assertEqual(self.simulation.turn_counter.value, turn_before)


class TestWarmupEquilibratesSolverState(unittest.TestCase):
    """Warmup should fill up a multi-turn wakefield solver's memory."""

    def setUp(self):
        circumference = 6911.56
        transition_gamma = 22.82177322938192
        harmonic = 4620
        sync_momentum = 25.92e9

        ring = Ring(circumference=circumference)
        rf_station = SingleHarmonicRFStation(
            harmonic=harmonic, voltage=0.9e6, phi_rf=0.0
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=transition_gamma
            ),
            orbit_length=circumference,
        )
        t_rev = magnetic_cycle.get_t_rev_init(
            circumference, particle_type=proton
        )

        self.n_wake_turns = 5
        profile = StaticProfile.from_rad(0, 2 * np.pi, 2**8, t_period=t_rev)
        wakefield = WakeField(
            sources=(Resonators(R_SHUNT, F_RES, Q_FACTOR),),
            solver=ContinuousMultiTurnTimeDomainSolver(
                n_turns=self.n_wake_turns
            ),
            profile=profile,
        )

        beam = Beam(intensity=1e10, particle_type=proton)
        ring.add_elements((wakefield, drift, rf_station))
        simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4, seed=1, n_macroparticles=1000
            ),
            beam=beam,
        )

        self.simulation = simulation
        self.beam = beam
        self.wakefield = wakefield

    def test_warmup_fills_continuous_multiturn_wake_deque(self):
        solver = self.wakefield.solver
        self.assertEqual(len(solver._previous_wakes), 0)

        warmup(
            self.simulation,
            self.beam,
            n_turns=self.n_wake_turns,
            show_progressbar=False,
            verbose=False,
        )

        self.assertEqual(len(solver._previous_wakes), self.n_wake_turns)


if __name__ == "__main__":
    unittest.main()
