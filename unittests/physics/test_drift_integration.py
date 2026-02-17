import logging
import unittest

import numpy as np
import pytest
from blond.core.backends.backend import Numpy32Bit, backend
from blond.core.base import DynamicParameter
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.testing.mocks import simulation_mock

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    ConstantMagneticCycle,
    DriftSimple,
    EmptyBeam,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
    uranium_29,
)


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
        cavity1.phi_rf = 0
        cavity2 = SingleHarmonicRFStation(section_index=1)
        cavity2.harmonic = 35640
        cavity2.voltage = 6e6
        cavity2.phi_rf = 0

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
        drift1.transition_gamma = 55.759505
        drift2.transition_gamma = 55.759505
        drift3.transition_gamma = 55.759505
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
        cavity1.phi_rf = 0

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
        drift1.momentum_compaction_factor = 1 / 55.759505**2
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

    def test_add_observable(self):
        drift1 = DriftSimple.headless(transition_gamma=12, orbit_length=12)

        beam = EmptyBeam(particle_type=uranium_29, reference_total_energy=12)
        observable_1 = BeamObservationOncePerTurn(each_turn_i=1)
        drift1.add_observable(beam, observable_1)
        with self.assertRaisesRegex(ValueError, "already set"):
            drift1.add_observable(
                beam, BeamObservationOncePerTurn(each_turn_i=1)
            )

    def test_track_with_observable(self):
        drift1 = DriftSimple.headless(transition_gamma=12, orbit_length=12)

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
