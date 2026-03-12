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
    ScheduledConstant,
    UnsafeUserElement,
    UserDefinedElement,
    get_scheduler,
)
from blond.core.backends import backend
from blond.core.beam.base import BeamBaseClass
from blond.handle_results.helpers import callers_relative_path
from blond.testing.backend_testing import multi_backend_testcase


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
        with self.assertRaises(ValueError):
            get_scheduler(
                "a string",
            )
        with self.assertRaises(TypeError):
            get_scheduler(np.ones(10), mode="not_in_the_mode_today")


class TestSchedulable(unittest.TestCase):
    def setUp(self):
        self.schedulable = Schedulable()
        self.schedulable._add_intended_schedule("voltage")
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

    @multi_backend_testcase
    def test_schedule(self):
        schedulable = Schedulable()
        schedulable.voltage = None
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


class TestMultiSchedules(unittest.TestCase):

    def setUp(self):
        self.turn_based_np = np.arange(10)
        self.turn_based_list = np.linspace(0, 5, 10).tolist()
        self.turn_based_tuple = tuple(v*2 for v in range(10))

        self.all_turn_based = [self.turn_based_np, self.turn_based_list,
                               self.turn_based_tuple]

        self.time_based_np = np.array([[0, 5], [1, 2]])
        self.time_based_list = [[0, 5], [10, 20]]
        self.time_based_tuple = ((0, 5), (30, 31))

        self.all_time_based = [self.time_based_np, self.time_based_list,
                               self.time_based_tuple]

        self.all_elements = self.all_turn_based + self.all_time_based

        self.scheduled_array = ScheduledArray(np.arange(10)**2)
        self.scheduled_time = ScheduledInterpolation(np.array([0, 5]),
                                                     np.array([100, 200]))

    @multi_backend_testcase
    def test_get_scheduler_constant(self):
        scheduler = get_scheduler(0)
        self.assertIsInstance(scheduler, ScheduledConstant)
        self.assertEqual(scheduler.get_scheduled(0, 0), 0)
        self.assertEqual(scheduler.get_scheduled(1, 1), 0)

    @multi_backend_testcase
    def test_get_scheduler_array(self):
        for element in self.all_turn_based:
            scheduler = get_scheduler(element)
            self.assertIsInstance(scheduler, ScheduledArray)
            self.assertIsInstance(scheduler.values, backend.backend.ndarray)

            for turn in range(10):
                self.assertEqual(element[turn], scheduler.get_scheduled(turn, None))

    @multi_backend_testcase
    def test_get_scheduler_interpolate(self):

        for element in self.all_time_based:
            scheduler = get_scheduler(element)
            self.assertIsInstance(scheduler, ScheduledInterpolation)

            for time in np.linspace(0, 5, 10):
                target = np.interp(time, element[0], element[1])
                self.assertEqual(target, scheduler.get_scheduled(None, time))

    @multi_backend_testcase
    def test_schedule_singles(self):

        schedulable = Schedulable()
        schedulable.test = 0

        for element in self.all_elements:
            scheduler = get_scheduler(element)
            schedulable.schedule("test", element)

            for turn, time in enumerate(np.linspace(0, 5, 10)):
                target = scheduler.get_scheduled(turn, time)
                schedulable.apply_schedules(turn, time)
                self.assertEqual(target, schedulable.test)

    @multi_backend_testcase
    def test_schedule_multi(self):
        schedulable = Schedulable()
        schedulable.test = 0

        schedulable.schedule("test", *self.all_elements, 0)

        all_schedulers = []
        for element in self.all_elements + [0]:
            all_schedulers.append(get_scheduler(element))

        for turn, time in enumerate(np.linspace(0, 5, 10)):
            target = np.array(
                [s.get_scheduled(turn, time) for s in all_schedulers]
            )

            schedulable.apply_schedules(turn, time)
            value = schedulable.test
            if isinstance(backend.backend, backend.CupyBackend):
                value = value.get()

            np.testing.assert_array_equal(target, value)


class TestUnsafeUserElement(unittest.TestCase):
    def test_init(self):
        class InvalidElement:
            def not_track(self): ...

        class ValidElement:
            def track(self): ...

        element = InvalidElement()
        with self.assertRaises(TypeError):
            UnsafeUserElement(element)

        element = ValidElement()
        with self.assertWarns(Warning):
            wrapper = UnsafeUserElement(element)

        self.assertTrue(wrapper._element is element)
        self.assertIsInstance(wrapper, UserDefinedElement)
        self.assertNotIsInstance(element, UserDefinedElement)
        self.assertEqual(wrapper.section_index, 0)
        self.assertTrue(wrapper.active)

    def test_track(self):
        called_with = []

        class Element:
            def track(self, beam):
                called_with.append(beam)

        element = Element()
        wrapper = UnsafeUserElement(element)

        call_args = [1, 2, 3]
        for arg in call_args:
            wrapper.track(arg)

        self.assertListEqual(called_with, call_args)


if __name__ == "__main__":
    unittest.main()
