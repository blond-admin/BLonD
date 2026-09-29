from unittest import mock

import pytest

from blond.core.backends.backend import Numpy64Bit, Specials, backend
from blond.testing.backend_testing import BLonDTestCase


class TestSpecialsFlush(BLonDTestCase):
    def tearDown(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

    def test_every_specials_has_flush(self) -> None:
        backend.set_specials("python")
        self.assertIsNone(Specials.flush())
        self.assertIsNone(backend.specials.flush())

    @pytest.mark.backend_mutation
    def test_set_specials_flushes_previous(self) -> None:
        backend.set_specials("python")
        with mock.patch.object(
            type(backend.specials), "flush", create=True
        ) as flush:
            backend.set_specials("cpp")
        flush.assert_called_once_with()

    @pytest.mark.backend_mutation
    def test_change_backend_flushes_previous(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("python")

        # A same-class change is a no-op; force a real change via a
        # subclass so the old specials must be flushed.
        class OtherNumpy(Numpy64Bit):
            pass

        with mock.patch.object(
            type(backend.specials), "flush", create=True
        ) as flush:
            backend.change_backend(OtherNumpy)
        flush.assert_called()
