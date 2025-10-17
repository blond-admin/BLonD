import unittest
from unittest.mock import MagicMock, PropertyMock

from blond._core.beam.base import BeamBaseClass
from blond._core.beam.particle_types import proton
from blond.cycles.magnetic_cycle import MagneticCycleByTime
from blond.physics.energy_reference_kick import EnergyReferenceKick


class DummyMagneticCycleByTime(MagneticCycleByTime):
    def __init__(self):
        # Pass dummy data to satisfy base class
        super().__init__(
            reference_particle=proton,
            base_time=[0],
            base_values=[1e9],  # initial energy
        )

    def get_target_total_energy(self, *, turn_i, section_i, reference_time, particle_type):
        return 1e9 + turn_i * 1e6

class DummyTurn:
    def __init__(self):
        self.value = 0


class DummyBeam(BeamBaseClass):
    """Minimal Beam implementation for testing EnergyReferenceKick."""
    def __init__(self):
        self.reference_total_energy = 1e9
        self._dE = 0.0
        self.reference_time = 0.0

    @property
    def particle_type(self):
        return proton

    def plot_hist2d(self, *args, **kwargs):
        pass

    def ratio(self, *args, **kwargs):
        return 1.0

    def setup_beam(self, *args, **kwargs):
        pass


class TestEnergyReferenceKick(unittest.TestCase):

    def setUp(self):
        self.simulation = MagicMock()
        self.simulation.turn_i = DummyTurn()
        self.simulation.magnetic_cycle = DummyMagneticCycleByTime()
        self.simulation.ring = MagicMock()

        # Bypass __init__ TypeError by manually setting magnetic_cycle after creation
        self.energy_kick = EnergyReferenceKick(section_index=0)
        self.energy_kick._magnetic_cycle = self.simulation.magnetic_cycle
        self.energy_kick._turn_i = self.simulation.turn_i
        self.energy_kick._ring = self.simulation.ring

    def test_init_raises_typeerror(self):
        """Test that using an invalid magnetic cycle raises a TypeError when required."""
        simulation = MagicMock()
        simulation.turn_i = DummyTurn()
        simulation.magnetic_cycle = object()  # invalid type
        simulation.ring = MagicMock()

        kick = EnergyReferenceKick(section_index=0)
        with self.assertRaises(TypeError):
            kick.on_init_simulation(simulation)

    def test_track_updates_beam_energy(self):
        beam = DummyBeam()
        type(beam).particle_type = PropertyMock(return_value=proton)
        self.simulation.turn_i.value = 5

        self.energy_kick.schedule_active = False  # No schedules applied

        original_ref_energy = beam.reference_total_energy
        original_dE = beam._dE

        self.energy_kick.track(beam)

        target_energy = self.simulation.magnetic_cycle.get_target_total_energy(
            turn_i=5,
            section_i=self.energy_kick.section_index,
            reference_time=beam.reference_time,
            particle_type=beam.particle_type,
        )
        expected_change = target_energy - original_ref_energy

        self.assertEqual(beam.reference_total_energy, original_ref_energy + expected_change)
        self.assertEqual(beam._dE, original_dE - expected_change)

    def test_track_with_schedule_applies_schedules(self):
        beam = DummyBeam()
        self.simulation.turn_i.value = 3

        self.energy_kick.schedule_active = True

        called = {}

        def fake_apply_schedules(turn_i, reference_time):
            called['called'] = True
            called['turn_i'] = turn_i
            called['reference_time'] = reference_time

        self.energy_kick.apply_schedules = fake_apply_schedules

        self.energy_kick.track(beam)

        self.assertTrue(called.get('called', False))
        self.assertEqual(called.get('turn_i'), 3)
        self.assertEqual(called.get('reference_time'), beam.reference_time)


if __name__ == "__main__":
    unittest.main()
