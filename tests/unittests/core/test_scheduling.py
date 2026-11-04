import unittest

import numpy as np
import scipy

from blond.core.scheduling import (
    ScheduledArray,
    ScheduledBaseClass,
    ScheduledFunctional,
    ScheduledInterpolation,
    get_scheduler,
)


class TestScheduledArray(unittest.TestCase):
    def test_get_scheduled_indexes_by_turn(self):
        scheduler = ScheduledArray(np.arange(10.0))
        self.assertIsInstance(scheduler, ScheduledBaseClass)
        self.assertEqual(scheduler.get_scheduled(3, 100.0), 3.0)


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

    def test_stores_input(self):
        t_arr = np.linspace(0, 10, 5)
        vals = np.linspace(-10, 0, 5)
        scheduler = ScheduledInterpolation(
            times=t_arr,
            values=vals,
            interpolator=scipy.interpolate.Akima1DInterpolator,
            method="makima",
        )
        np.testing.assert_array_equal(scheduler.times, t_arr)
        np.testing.assert_array_equal(scheduler.values, vals)
        self.assertIs(
            scheduler.interpolator_type, scipy.interpolate.Akima1DInterpolator
        )
        self.assertEqual(scheduler.interpolator_kwargs, {"method": "makima"})

    def test_default_interpolator_type_and_kwargs(self):
        scheduler = ScheduledInterpolation(
            times=np.array([0.0, 1.0]), values=np.array([1.0, 2.0])
        )
        self.assertIs(scheduler.interpolator_type, scipy.interpolate.interp1d)
        self.assertEqual(scheduler.interpolator_kwargs, {})

    def test_stores_copies(self):
        t_arr = np.array([0.0, 1.0, 2.0])
        vals = np.array([1.0, 2.0, 3.0])
        kwargs = {"kind": "linear"}
        scheduler = ScheduledInterpolation(times=t_arr, values=vals, **kwargs)
        t_arr[0] = 99.0
        vals[0] = 99.0
        kwargs["kind"] = "cubic"
        np.testing.assert_array_equal(scheduler.times, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(scheduler.values, [1.0, 2.0, 3.0])
        self.assertEqual(scheduler.interpolator_kwargs, {"kind": "linear"})

    def test_stores_lists_as_arrays(self):
        scheduler = ScheduledInterpolation(times=[0.0, 1.0], values=[1.0, 2.0])
        self.assertIsInstance(scheduler.times, np.ndarray)
        self.assertIsInstance(scheduler.values, np.ndarray)


class TestScheduledFunctional(unittest.TestCase):
    def test_calls_function_with_turn_and_time(self):
        received = {}

        def func(turn_i, reference_time):
            received["turn_i"] = turn_i
            received["reference_time"] = reference_time
            return 42.0

        scheduler = ScheduledFunctional(func)
        value = scheduler.get_scheduled(turn_i=5, reference_time=1.5)

        self.assertEqual(value, 42.0)
        self.assertEqual(received, {"turn_i": 5, "reference_time": 1.5})

    def test_passes_arguments_by_keyword(self):
        # Function only uses reference_time but must accept both keywords.
        scheduler = ScheduledFunctional(
            lambda turn_i, reference_time: 2.0 * reference_time
        )
        self.assertEqual(scheduler.get_scheduled(0, 3.0), 6.0)


class TestFunctions(unittest.TestCase):
    def test_get_scheduler_1(self):
        sched1 = get_scheduler(
            np.ones(10),
        )
        sched2 = get_scheduler(
            (np.ones(10), np.ones(10)),
        )
        sched3 = get_scheduler(lambda turn_i, reference_time: 1.0)
        self.assertEqual(type(sched1), ScheduledArray)
        self.assertEqual(type(sched2), ScheduledInterpolation)
        self.assertEqual(type(sched3), ScheduledFunctional)
        self.assertEqual(sched3.get_scheduled(0, 0.0), 1.0)
        with self.assertRaises(TypeError):
            get_scheduler(
                "a string",
            )
        with self.assertRaises(TypeError):
            get_scheduler(np.ones(10), mode="not_in_the_mode_today")


if __name__ == "__main__":
    unittest.main()
