import unittest
from copy import copy

from blond.core.ring.beam_physics_relevant_elements import (
    BeamPhysicsRelevantElements,
)
from blond.core.simulation.intensity_effect_manager import (
    IntensityEffectManager,
)
from blond.testing.mocks import (
    simulation_mock,
    static_profile_mock,
    wakefield_profile_mock,
)


class TestIntensityEffectManager(unittest.TestCase):
    def setUp(self):
        self.wake = wakefield_profile_mock
        self.profile = static_profile_mock
        wakefield_profile_mock.profile = copy(static_profile_mock)
        simulation_mock.ring.elements.elements = [
            static_profile_mock,
            wakefield_profile_mock,
        ]
        simulation_mock.ring.elements.get_elements = (
            lambda x, recursive: BeamPhysicsRelevantElements.get_elements(
                self=simulation_mock.ring.elements,
                class_=x,
                recursive=recursive,
            )
        )
        self.intensity_effect_manager = IntensityEffectManager(
            simulation=simulation_mock
        )

    def test___init__(self):
        pass  # done by setup

    def test_has_wakefields(self):
        self.assertTrue(self.intensity_effect_manager.has_wakefields())

    def test_is_active_wakefields(self):
        self.intensity_effect_manager.set_wakefields(active=True)
        active = self.intensity_effect_manager.is_active_wakefields()
        self.assertEqual(active, True)

        self.intensity_effect_manager.set_wakefields(active=False)
        active = self.intensity_effect_manager.is_active_wakefields()
        self.assertEqual(active, False)

    def test_set_wakefields(self):
        self.intensity_effect_manager.set_wakefields(active=True)
        self.assertTrue(self.wake.active)
        self.intensity_effect_manager.set_wakefields(active=False)
        self.assertFalse(self.wake.active)

    def test_set_profiles(self):
        self.intensity_effect_manager.set_profiles(active=True)
        self.assertTrue(self.wake.profile.active)
        self.assertTrue(self.profile.active)

        self.intensity_effect_manager.set_profiles(active=False)
        self.assertFalse(self.wake.profile.active)
        self.assertFalse(self.profile.active)

    def test_is_active_profiles(self):
        self.intensity_effect_manager.set_profiles(active=True)
        active = self.intensity_effect_manager.is_active_profiles()
        self.assertEqual(active, True)

        self.intensity_effect_manager.set_profiles(active=False)
        active = self.intensity_effect_manager.is_active_profiles()
        self.assertEqual(active, False)


if __name__ == "__main__":
    unittest.main()
