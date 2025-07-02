import unittest
from functools import cached_property
from typing import Optional

from blond3 import Simulation
from blond3._core.base import (
    BeamPhysicsRelevant,
    DynamicParameter,
    HasPropertyCache,
    MainLoopRelevant,
    Preparable,
)
from blond3._core.beam.base import BeamBaseClass


class BeamPhysicsRelevantTester(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0, name: Optional[str] = None):
        super().__init__(section_index=section_index, name=name)

    def track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class TestBeamPhysicsRelevant(unittest.TestCase):
    def setUp(self):
        self.beam_physics_relevant = BeamPhysicsRelevantTester(
            section_index=10, name="Simon"
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_section_index(self):
        self.assertEqual(10, self.beam_physics_relevant.section_index)

    @unittest.skip("Abstract method")
    def test_track(self):
        # self.beam_physics_relevant.track(beam=None)
        pass


class TestDynamicParameter(unittest.TestCase):
    def setUp(self):
        self.dynamic_parameter = DynamicParameter(value_init=5)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_change(self):
        array = [0]

        def callback(newvalue):
            array[0] = newvalue

        self.dynamic_parameter.on_change(callback=callback)
        self.dynamic_parameter.value = 10
        self.assertEqual(array[0], 10)


class HasPropertyCacheHelper(HasPropertyCache):
    def __init__(self):
        self.foo = 1

    @cached_property
    def bar(self):
        return self.foo

    def invalidate_cache(self):
        self._invalidate_cache(("bar",))


class TestHasPropertyCache(unittest.TestCase):
    def setUp(self):
        self.has_property_cache = HasPropertyCacheHelper()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_invalidate_cache(self):
        self.has_property_cache.foo = 22
        self.assertEqual(22, self.has_property_cache.bar)
        self.has_property_cache.foo = 11
        self.assertEqual(22, self.has_property_cache.bar)  # cache still active
        self.has_property_cache.invalidate_cache()
        self.assertEqual(11, self.has_property_cache.bar)


class MainLoopRelevantHelper(MainLoopRelevant):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class TestMainLoopRelevant(unittest.TestCase):
    def setUp(self):
        self.main_loop_relevant = MainLoopRelevantHelper()
        self.main_loop_relevant.each_turn_i = 10

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_is_active_this_turn(self):
        self.assertFalse(self.main_loop_relevant.is_active_this_turn(turn_i=1))
        self.assertTrue(self.main_loop_relevant.is_active_this_turn(turn_i=0))
        self.assertTrue(self.main_loop_relevant.is_active_this_turn(turn_i=10))
        self.assertTrue(self.main_loop_relevant.is_active_this_turn(turn_i=20))


class PreparableHelper(Preparable):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        pass


class TestPreparable(unittest.TestCase):
    def setUp(self):
        self.preparable = PreparableHelper()

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip("Abstract methods")
    def test_on_init_simulation(self):
        pass

    @unittest.skip("Abstract methods")
    def test_on_run_simulation(self):
        pass


if __name__ == "__main__":
    unittest.main()
