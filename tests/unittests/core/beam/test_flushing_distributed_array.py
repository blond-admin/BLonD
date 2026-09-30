import unittest
from copy import copy, deepcopy
from unittest import mock

import numpy as np

from blond import Beam, proton
from blond.core.backends.backend import backend
from blond.core.beam.flushing_distributed_array import (
    FlushingCoordinates,
    FlushingDistributedArray,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.testing.backend_testing import BLonDTestCase


class TestFlushingDistributedArray(BLonDTestCase):
    """Backend-agnostic: counts calls to the active ``specials.flush``."""

    def setUp(self) -> None:
        self.values = backend.array(
            np.linspace(-1.0, 2.0, 16), dtype=backend.float
        )
        self.array = FlushingDistributedArray(self.values)
        patcher = mock.patch.object(backend.specials, "flush")
        self.flush = patcher.start()
        self.addCleanup(patcher.stop)

    def test_array_local_read_flushes(self) -> None:
        self.assertIs(self.array.array_local, self.values)
        self.flush.assert_called_once_with()

    def test_array_local_write_flushes_before_replacing(self) -> None:
        replacement = backend.zeros(3, dtype=backend.float)
        self.array.array_local = replacement
        self.flush.assert_called_once_with()
        self.assertIs(self.array.array_local_without_flush, replacement)

    def test_array_local_without_flush_does_not_flush(self) -> None:
        self.assertIs(self.array.array_local_without_flush, self.values)
        self.flush.assert_not_called()

    def test_sizes_do_not_flush(self) -> None:
        # Queued kernels never change the particle count, and the RF
        # station asks for it between queued kicks every turn.
        self.assertEqual(self.array.local_size, 16)
        self.assertEqual(self.array.global_size, 16)
        self.flush.assert_not_called()

    def test_every_data_reading_method_flushes(self) -> None:
        readers = {
            "min": lambda: self.array.min(),
            "max": lambda: self.array.max(),
            "mean": lambda: self.array.mean(),
            "std": lambda: self.array.std(),
            "sum": lambda: self.array.sum(),
            "histogram": lambda: self.array.histogram(4, range=(-1, 2)),
            "mpi_gather": lambda: self.array.mpi_gather(),
            "mpi_scatter": lambda: self.array.mpi_scatter(),
            "copy_as_numpy": lambda: self.array.copy_as_numpy(),
        }
        for name, read in readers.items():
            with self.subTest(method=name):
                self.flush.reset_mock()
                read()
                if name == "mpi_scatter" and not self.array.is_distributed:
                    continue  # returns before touching the data
                self.assertGreater(self.flush.call_count, 0)

    def test_copies_flush_and_stay_flushing(self) -> None:
        for make_copy in (copy, deepcopy):
            with self.subTest(copy=make_copy.__name__):
                self.flush.reset_mock()
                copied = make_copy(self.array)
                self.flush.assert_called_once_with()
                self.assertIsInstance(copied, FlushingDistributedArray)
                np.testing.assert_array_equal(
                    copied.copy_as_numpy(), self.array.copy_as_numpy()
                )

    def test_from_distributed_array_shares_the_array(self) -> None:
        plain = DistributedArray(self.values)
        converted = FlushingDistributedArray.from_distributed_array(plain)
        self.assertIsInstance(converted, FlushingDistributedArray)
        self.assertIs(converted.array_local_without_flush, self.values)
        self.assertEqual(converted.is_distributed, plain.is_distributed)


class _CoordinateHolder:
    """Bare-bones owner of a single `FlushingCoordinates` attribute."""

    _dt = FlushingCoordinates()


class TestFlushingCoordinatesFirstAssignment(BLonDTestCase):
    """`FlushingCoordinates.__set__` should only flush a real replacement."""

    def test_creating_a_beam_does_not_flush(self) -> None:
        # Beam.__init__ sets _dE/_dt to None for the first
        # time; there is nothing queued against them yet, so creating a
        # beam mid-turn must not split a pending batch.
        with mock.patch.object(backend.specials, "flush") as flush:
            Beam(intensity=1e11, particle_type=proton)
        flush.assert_not_called()

    def test_first_real_assignment_does_not_flush(self) -> None:
        holder = _CoordinateHolder()
        holder._dt = None  # as BeamBaseClass.__init__ does
        # Pre-wrap the value while flushing is un-patched: constructing a
        # fresh `FlushingDistributedArray` legitimately flushes once (it
        # is a new object, not a replacement) -- that is not what this
        # test is about. Assigning it directly (already the flushing
        # type) skips that wrapping step.
        new_value = FlushingDistributedArray(np.zeros(10))
        with mock.patch.object(backend.specials, "flush") as flush:
            holder._dt = new_value
        flush.assert_not_called()

    def test_replacing_existing_coordinates_still_flushes(self) -> None:
        holder = _CoordinateHolder()
        holder._dt = FlushingDistributedArray(np.zeros(10))
        new_value = FlushingDistributedArray(np.ones(10))
        with mock.patch.object(backend.specials, "flush") as flush:
            holder._dt = new_value
        flush.assert_called_once_with()


class TestBeamCoordinateStorage(BLonDTestCase):
    """Every way of setting dt/dE ends as the flushing type."""

    COORDINATES = ("_dt", "_dE")

    def setUp(self) -> None:
        self.beam = Beam(intensity=1e11, particle_type=proton)
        self.beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 10), dE=np.linspace(-1e6, 1e6, 10)
        )

    def _other_beam(self) -> Beam:
        other = Beam(intensity=1e10, particle_type=proton)
        other.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 1), dE=np.linspace(-1e6, 1e6, 1)
        )
        return other

    def _assert_all_flushing(self, beam: Beam) -> None:
        for name in self.COORDINATES:
            self.assertIsInstance(
                getattr(beam, name), FlushingDistributedArray, name
            )

    def test_unset_coordinates_are_none(self) -> None:
        beam = Beam(intensity=1e11, particle_type=proton)
        for name in self.COORDINATES:
            self.assertIsNone(getattr(beam, name), name)

    def test_every_assignment_path_keeps_the_flushing_type(self) -> None:
        values = np.linspace(0.0, 1.0, 10)
        paths = {
            "setup_beam": lambda beam: None,
            "add_beam": lambda beam: beam.add_beam(self._other_beam()),
            "add_particles": lambda beam: beam.add_particles(
                DistributedArray(values), DistributedArray(values)
            ),
            "copy_coordinates_from": lambda beam: beam.copy_coordinates_from(
                self._other_beam()
            ),
            "purge_flagged_entries": lambda beam: beam.purge_flagged_entries(),
            "sort_by_dt": lambda beam: beam.sort_by_dt(),
            "plain_assignment": lambda beam: setattr(
                beam, "_dE", DistributedArray(values)
            ),
        }
        for name, change in paths.items():
            with self.subTest(path=name):
                beam = deepcopy(self.beam)
                change(beam)
                self._assert_all_flushing(beam)

    def test_deepcopy_of_beam_keeps_the_flushing_type(self) -> None:
        self._assert_all_flushing(deepcopy(self.beam))

    def test_raw_array_assignment_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.beam._dt = np.zeros(10)


if __name__ == "__main__":
    unittest.main()
