import logging
import unittest
from copy import deepcopy

import numpy as np
import pytest

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    ConstantMagneticCycle,
    DriftSimple,
    EmptyBeam,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
    uranium_29,
)
from blond.core.backends.backend import Numpy32Bit, backend
from blond.core.base import DynamicParameter
from blond.core.beam.beams import ProbeBeam
from blond.cycles.magnetic_cycle import (
    ConstantMagneticCycle,
    MagneticCyclePerTurn,
)
from blond.testing.mocks import simulation_mock


class TestDriftIntegration(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    def test_exec(self):
        circumference = 26658.883

        logging.basicConfig(level=logging.INFO)
        ring = Ring(circumference=circumference)

        cavity1 = SingleHarmonicRFStation(section_index=0)
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf_design = 0
        cavity2 = SingleHarmonicRFStation(section_index=1)
        cavity2.harmonic = 35640
        cavity2.voltage = 6e6
        cavity2.phi_rf_design = 0

        N_TURNS = int(1e3)
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=0,
        )

        drift2 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=1,
        )

        drift3 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=1,
        )
        momentum_compaction_factor_ = momentum_compaction_factor(
            transition_gamma=55.759505
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor_
        drift2.momentum_compaction_factor = momentum_compaction_factor_
        drift3.momentum_compaction_factor = momentum_compaction_factor_

        beam1 = Beam(intensity=1e9, particle_type=proton)

        sim = Simulation.from_locals(locals())
        sim.ring.assert_circumference()

    @pytest.mark.backend_mutation
    def test_check_circumference(self):
        circumference = 26658.883

        logging.basicConfig(level=logging.INFO)
        ring = Ring(circumference=circumference)

        cavity1 = SingleHarmonicRFStation(section_index=0)
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf_design = 0

        N_TURNS = int(1e3)
        energy_cycle = ConstantMagneticCycle(
            value=450e9,
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=circumference / 2,
            section_index=0,
        )
        beam = EmptyBeam(proton)
        momentum_compaction_factor_ = 1 / 55.759505**2
        drift1.momentum_compaction_factor = momentum_compaction_factor_
        sim = Simulation.from_locals(locals())
        with self.assertRaisesRegex(AssertionError, "but should be"):
            sim.ring.assert_circumference()
        with self.assertRaisesRegex(AssertionError, "but should be"):
            sim.finalize(beams=beam, n_turns=N_TURNS)
        sim.check_circumference = "warn"
        with self.assertWarnsRegex(UserWarning, "but should be"):
            sim.finalize(beams=beam, n_turns=N_TURNS)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            sim.check_circumference = "some typo"
            sim.finalize(beams=beam, n_turns=N_TURNS)
        sim.check_circumference = "ignore"
        sim.finalize(beams=beam, n_turns=N_TURNS)

        self.assertAlmostEqual(
            sim.ring.momentum_compaction_factor,
            momentum_compaction_factor_,
        )

    def test_add_observable(self):
        drift1 = DriftSimple.headless(
            momentum_compaction_factor(transition_gamma=12), orbit_length=12
        )

        beam = EmptyBeam(particle_type=uranium_29, reference_total_energy=12)
        observable_1 = BeamObservationOncePerTurn(each_turn_i=1)
        drift1.add_observable(beam, observable_1)
        with self.assertRaisesRegex(ValueError, "already set"):
            drift1.add_observable(
                beam, BeamObservationOncePerTurn(each_turn_i=1)
            )

    def test_track_with_observable(self):
        drift1 = DriftSimple.headless(
            momentum_compaction_factor(transition_gamma=12), orbit_length=12
        )

        beam = EmptyBeam(particle_type=uranium_29, reference_total_energy=12)
        observable_1 = BeamObservationOncePerTurn(each_turn_i=1)
        observable_1._simulation = simulation_mock
        observable_1._simulation.turn_i = DynamicParameter(1)
        observable_1.on_run_simulation(
            simulation=simulation_mock, beam=beam, n_turns=2
        )
        drift1.add_observable(beam, observable_1)
        with self.assertRaisesRegex(ValueError, "already set"):
            drift1.add_observable(
                beam, BeamObservationOncePerTurn(each_turn_i=1)
            )

        drift1.track(beam=beam)

    def test_momentum_compaction_splitting(self):
        circumference = 1
        beam1 = ProbeBeam(
            dE=np.linspace(-10, 10, 5),
            intensity=1e9,
            particle_type=proton,
        )

        def run_two_drifts():
            ring = Ring(circumference=circumference)

            energy_cycle = ConstantMagneticCycle(
                value=450e9,
                reference_particle=proton,
            )

            drift1 = DriftSimple(
                orbit_length=circumference * 1 / 4,
                momentum_compaction_factor=3,  # intentionally different
                section_index=0,
            )

            drift2 = DriftSimple(
                orbit_length=circumference * 3 / 4,
                momentum_compaction_factor=4,  # intentionally different
                section_index=1,
            )
            rf = SingleHarmonicRFStation(
                voltage=0, phi_rf=0, harmonic=1, section_index=0
            )
            rf2 = SingleHarmonicRFStation(
                voltage=0, phi_rf=0, harmonic=1, section_index=1
            )

            sim = Simulation.from_locals(locals())
            sim.ring.assert_circumference()

            sim.print_one_turn_execution_order()
            p = deepcopy(beam1)
            sim.run_simulation(beams=p, n_turns=5)

            global_momentum_compaction_factor = (
                sim.ring.momentum_compaction_factor
            )  # this is tested

            return p._dt.array_local, global_momentum_compaction_factor

        def run_combined_drifts(global_momentum_compaction_factor):
            ring = Ring(circumference=circumference)

            energy_cycle = ConstantMagneticCycle(
                value=450e9,
                reference_particle=proton,
            )

            drift1 = DriftSimple(
                orbit_length=circumference,
                momentum_compaction_factor=global_momentum_compaction_factor,
                section_index=0,
            )

            rf = SingleHarmonicRFStation(voltage=0, phi_rf=0, harmonic=1)

            sim = Simulation.from_locals(locals())
            sim.ring.assert_circumference()
            p = deepcopy(beam1)
            sim.run_simulation(beams=p, n_turns=5)
            return p._dt.array_local

        dt1, global_momentum_compaction_factor = run_two_drifts()
        dt2 = run_combined_drifts(global_momentum_compaction_factor)
        np.testing.assert_allclose(dt1, dt2)
