import unittest
from copy import copy
from unittest.mock import MagicMock, Mock

import numpy as np

from blond.core.base import DynamicParameter
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import proton
from blond.physics.energy_reference_kick import ReferenceEnergyChange
from blond.testing.mocks import cycle_const_mock, simulation_mock


class TestEnergyReferenceKick(unittest.TestCase):
    def setUp(self):
        self.simulation = simulation_mock
        self.simulation.turn_i = Mock(DynamicParameter)
        self.simulation.magnetic_cycle = cycle_const_mock

        # Bypass __init__ TypeError by manually setting magnetic_cycle after creation
        self.energy_kick = ReferenceEnergyChange(section_index=0)
        self.energy_kick._magnetic_cycle = self.simulation.magnetic_cycle
        self.energy_kick._turn_i = self.simulation.turn_i
        self.energy_kick._ring = self.simulation.ring

    def test_init_raises_typeerror(self):
        """Test that using an invalid magnetic cycle raises a TypeError when required."""
        simulation = MagicMock()
        simulation.turn_i = Mock(DynamicParameter)
        simulation.magnetic_cycle = object()  # invalid type
        simulation.ring = MagicMock()

        kick = ReferenceEnergyChange(section_index=0)
        with self.assertRaises(TypeError):
            kick.on_init_simulation(simulation)

    def test_track_updates_beam_energy(self):
        total_energy = 1e12
        self.simulation.magnetic_cycle.get_target_total_energy.return_value = (
            total_energy
        )
        beam = ProbeBeam(
            dE=[0], particle_type=proton, reference_total_energy=0.5e12
        )
        self.simulation.turn_i.value = 5

        self.energy_kick.schedule_active = False  # No schedules applied

        original_ref_energy = copy(beam.reference_total_energy)
        original_dE = np.copy(beam._dE)

        self.energy_kick.track(beam)

        target_energy = self.simulation.magnetic_cycle.get_target_total_energy(
            turn_i=5,
            section_i=self.energy_kick.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        expected_change = target_energy - original_ref_energy

        self.assertEqual(
            beam.reference_total_energy, original_ref_energy + expected_change
        )
        np.testing.assert_allclose(beam._dE, original_dE - expected_change)


if __name__ == "__main__":
    unittest.main()
