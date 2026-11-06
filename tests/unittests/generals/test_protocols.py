import unittest

from blond.testing.backend_testing import BLonDTestCase


class TestProtocols(BLonDTestCase):
    def test_no_crash(self):
        from blond.generals import protocols  # NOQA

    def test_schedule_function_is_generic(self):
        from typing import get_args

        from blond.generals.protocols import ScheduleFunction

        self.assertEqual(get_args(ScheduleFunction[float]), (float,))
