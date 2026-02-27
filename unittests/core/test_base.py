import unittest
from functools import cached_property

import numpy as np
import scipy

from blond import Simulation
from blond.core.base import (
    BeamObservationElement,
    BeamPhysicsRelevant,
    DynamicParameter,
    HasPropertyCache,
    MainLoopRelevant,
    Preparable,
    Schedulable,
    ScheduledArray,
    ScheduledInterpolation,
    get_scheduler,
)
from blond.core.beam.base import BeamBaseClass
from blond.handle_results.helpers import callers_relative_path


class BeamPhysicsRelevantTester(BeamPhysicsRelevant):
    def __init__(self, section_index: int = 0, name: str | None = None):
        super().__init__(section_index=section_index, name=name)

    def _track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
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


class TestScheduledInterpolation(unittest.TestCase):
    def setUp(
        self,
    ):  # TODO this testcase must be changed, when fixing the ISSUE #110
        t_arr = np.linspace(0, 10)
        vals = np.linspace(-10, 0)
        self.scheduled_constant = ScheduledInterpolation(
            times=t_arr, values=vals
        )
        np.testing.assert_allclose(
            self.scheduled_constant.get_scheduled(1, 1.0),
            np.interp(1.0, t_arr, vals),
        )
        np.testing.assert_allclose(
            self.scheduled_constant.get_scheduled(5, 1.0),
            np.interp(1.0, t_arr, vals),
        )
        np.testing.assert_allclose(
            self.scheduled_constant.get_scheduled(5, 1.0),
            np.interp(1.0, t_arr, vals),
        )

    def test_init(self):
        pass

    def test_init_other1(self):
        t_arr = np.linspace(0, 10)
        vals = np.linspace(-10, 0)
        scheduler = ScheduledInterpolation(
            times=t_arr,
            values=vals,
            interpolator=scipy.interpolate.Akima1DInterpolator,
            method="makima",
        )
        scheduler.get_scheduled(5, 1.0)  # should not crash

        def test_init_other2(self):
            t_arr = np.linspace(0, 10)
            vals = np.linspace(-10, 0)
            scheduler = ScheduledInterpolation(
                times=t_arr,
                values=vals,
                interpolator=scipy.interpolate.PchipInterpolator,
            )
            scheduler.get_scheduled(5, 1.0)  # should not crash


class BeamObservationElementTester(BeamObservationElement):
    def __init__(self, section_index: int = 0, name: str | None = None):
        super().__init__(section_index=section_index, name=name)

    def _track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
    ) -> None:
        pass


class TestBeamObservationElement(unittest.TestCase):
    def setUp(self):
        self.beam_observation_element = BeamObservationElementTester(
            section_index=10, name="Elle"
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_section_index(self):
        self.assertEqual(10, self.beam_observation_element.section_index)

    @unittest.skip("Abstract method")
    def test_track(self):
        # self.beam_observation_element.track(beam=None)
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
        self,
        simulation: Simulation,
        n_turns: int,
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
        self,
        simulation: Simulation,
        n_turns: int,
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


class TestFunctions(unittest.TestCase):
    def test_get_scheduler_1(self):
        sched1 = get_scheduler(
            np.ones(10),
        )
        sched2 = get_scheduler(
            (np.ones(10), np.ones(10)),
        )
        self.assertEqual(type(sched1), ScheduledArray)
        self.assertEqual(type(sched2), ScheduledInterpolation)
        with self.assertRaises(TypeError):
            get_scheduler(
                "a string",
            )
        with self.assertRaises(TypeError):
            get_scheduler(np.ones(10), mode="not_in_the_mode_today")


class TestSchedulable(unittest.TestCase):
    def setUp(self):
        self.schedulable = Schedulable(intended_for_scheduling=["voltage"])
        with self.assertRaisesRegex(AssertionError, "doesnt exist"):
            self.schedulable.schedule_from_file(
                attribute="voltage",
                filename=callers_relative_path(
                    "schedulable_testfile.txt", stacklevel=1
                ),
            )
        self.schedulable.voltage = None

        self.schedulable.schedule_from_file(
            attribute="voltage",
            filename=callers_relative_path(
                "schedulable_testfile.txt", stacklevel=1
            ),
        )

    def test___init__(self):
        pass

    def test_schedule(self):
        schedulable = Schedulable(intended_for_scheduling=[])
        schedulable.voltage = None
        with self.assertRaises(TypeError):
            schedulable.schedule("voltage", 1)
        schedulable.schedule("voltage", np.ones(10))
        schedulable.schedule(
            "voltage",
            ScheduledInterpolation(np.linspace(0, 10, 20), np.ones(20)),
        )
        schedulable.schedule(
            "voltage",
            ScheduledArray(np.ones(10)),
        )
        schedulable.schedule("voltage", np.ones(10))


if __name__ == "__main__":
    unittest.main()
