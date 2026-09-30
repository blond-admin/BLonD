import pathlib
from copy import deepcopy

import numpy as np
import pytest

from blond import Beam, proton
from blond.core.backends.backend import Numpy64Bit, backend
from blond.core.beam.flags import BeamFlags
from blond.generals.distributed.distributed_array import DistributedArray
from blond.testing.backend_testing import BLonDTestCase

KICK = dict(
    voltage=8e3, omega_rf=2e7, phi_rf=0.1, charge=1.0, acceleration_kick=12.0
)
ROOT = pathlib.Path(__file__).resolve().parents[4] / "blond"


@pytest.mark.backend_mutation
class TestKernelCallAccessors(BLonDTestCase):
    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        self.beam = Beam(intensity=1e11, particle_type=proton)
        self.beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-1e6, 1e6, 100),
        )

    def tearDown(self) -> None:
        backend.specials.flush()
        backend.set_specials("python")

    def _queue_kick(self) -> np.ndarray:
        before = self.beam.kernel_call_dE.copy()
        backend.specials.kick_single_harmonic(
            dt=self.beam.kernel_call_dt, dE=self.beam.kernel_call_dE, **KICK
        )
        return before

    def test_kernel_call_accessors_do_not_flush(self) -> None:
        before = self._queue_kick()
        np.testing.assert_array_equal(self.beam.kernel_call_dE, before)
        self.assertGreater(backend.specials.kernel_call_queue.n_bytes, 0)

    def test_data_accessors_flush(self) -> None:
        for read in (
            lambda: self.beam.read_partial_dE(),
            lambda: self.beam.write_partial_dE(),
            lambda: self.beam.dE.array_local,
            lambda: self.beam.dE_max,
        ):
            before = self._queue_kick()
            read()
            self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
            self.assertFalse(np.array_equal(self.beam.kernel_call_dE, before))

    def test_copy_sees_pending_kicks(self) -> None:  # Review Focus 5
        self._queue_kick()
        copied = deepcopy(self.beam.dE)  # flushing accessor, as muon code
        np.testing.assert_array_equal(
            copied.array_local, self.beam.kernel_call_dE
        )

    def test_add_particles_flushes_pending_kernel_calls(self) -> None:
        before = self._queue_kick()
        new_dt = DistributedArray(np.linspace(2e-9, 3e-9, 10))
        new_dE = DistributedArray(np.zeros(10))

        self.beam.add_particles(new_dt, new_dE)

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(
            np.array_equal(self.beam.kernel_call_dE[:100], before)
        )

    def test_add_beam_flushes_pending_kernel_calls_on_both_beams(
        self,
    ) -> None:
        before_self = self._queue_kick()

        other = Beam(intensity=1e10, particle_type=proton)
        other.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 10),
            dE=np.linspace(-1e6, 1e6, 10),
        )
        before_other = other.kernel_call_dE.copy()
        backend.specials.kick_single_harmonic(
            dt=other.kernel_call_dt, dE=other.kernel_call_dE, **KICK
        )

        self.beam.add_beam(other)

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(
            np.array_equal(self.beam.kernel_call_dE[:100], before_self)
        )
        self.assertFalse(
            np.array_equal(self.beam.kernel_call_dE[100:], before_other)
        )

    def test_iadd_flushes_pending_kernel_calls(self) -> None:
        before = self._queue_kick()

        other = Beam(intensity=1e10, particle_type=proton)
        other.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 10),
            dE=np.linspace(-1e6, 1e6, 10),
        )

        self.beam += other

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(
            np.array_equal(self.beam.kernel_call_dE[:100], before)
        )

    def test_purge_flagged_entries_flushes_pending_kernel_calls(self) -> None:
        before = self._queue_kick()

        self.beam.purge_flagged_entries(flag=BeamFlags.LOST.value)

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(np.array_equal(self.beam.kernel_call_dE, before))

    def test_setup_beam_flushes_pending_kernel_calls(self) -> None:
        self._queue_kick()
        self.assertGreater(backend.specials.kernel_call_queue.n_bytes, 0)

        self.beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 50),
            dE=np.linspace(-1e6, 1e6, 50),
        )

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)


@pytest.mark.backend_mutation
class TestCoordinateStorageFlushes(BLonDTestCase):
    """
    Reads through the Beam's coordinate storage see queued kernel calls.

    No Beam method has to remember a flush: any read of ``_dt``/``_dE``
    (as a future Beam method or a script would do) flushes by itself.
    """

    def setUp(self) -> None:
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp_deferred")
        self.beam = Beam(intensity=1e11, particle_type=proton)
        self.beam.setup_beam(
            dt=np.linspace(-1e-9, 1e-9, 100),
            dE=np.linspace(-1e6, 1e6, 100),
        )

    def tearDown(self) -> None:
        backend.specials.flush()
        backend.set_specials("python")

    def _queue_kick(self, beam: Beam | None = None) -> np.ndarray:
        beam = self.beam if beam is None else beam
        before = beam.kernel_call_dE.copy()
        backend.specials.kick_single_harmonic(
            dt=beam.kernel_call_dt, dE=beam.kernel_call_dE, **KICK
        )
        self.assertGreater(backend.specials.kernel_call_queue.n_bytes, 0)
        return before

    def _assert_flushed(self, before: np.ndarray) -> None:
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(np.array_equal(self.beam.kernel_call_dE, before))

    def test_direct_array_local_read_flushes(self) -> None:
        for name in ("_dt", "_dE", "_flags", "_ids"):
            with self.subTest(coordinate=name):
                before = self._queue_kick()
                getattr(self.beam, name).array_local  # noqa: B018
                self._assert_flushed(before)

    def test_distributed_statistics_see_flushed_data(self) -> None:
        for name in ("min", "max", "mean", "std", "sum"):
            with self.subTest(statistic=name):
                self._queue_kick()
                value = getattr(self.beam._dE, name)()
                self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
                expected = getattr(np, name)(self.beam.kernel_call_dE)
                self.assertAlmostEqual(value, expected, delta=1e-6)

    def test_distributed_histogram_sees_flushed_data(self) -> None:
        self._queue_kick()
        counts = self.beam._dE.histogram(10, range=(-2e6, 2e6)).copy()
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        expected, _ = np.histogram(
            self.beam.kernel_call_dE, bins=10, range=(-2e6, 2e6)
        )
        np.testing.assert_array_equal(counts, expected)

    def test_gather_and_host_copy_see_flushed_data(self) -> None:
        for read in (
            lambda: self.beam._dE.mpi_gather(),
            lambda: self.beam._dE.copy_as_numpy(),
            lambda: deepcopy(self.beam._dE).array_local_without_flush,
            lambda: deepcopy(self.beam).kernel_call_dE,
        ):
            before = self._queue_kick()
            copied = read()
            self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
            self.assertFalse(np.array_equal(copied, before))

    def test_replacing_coordinates_applies_queued_calls_first(self) -> None:
        old_dE = self.beam.kernel_call_dE
        before = self._queue_kick()

        self.beam._dE = DistributedArray(np.zeros(100))

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(np.array_equal(old_dE, before))

    def test_replacing_array_local_applies_queued_calls_first(self) -> None:
        old_dE = self.beam.kernel_call_dE
        before = self._queue_kick()

        self.beam._dE.array_local = np.zeros(100)

        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(np.array_equal(old_dE, before))

    def test_every_assignment_path_keeps_reads_flushing(self) -> None:
        def other_beam() -> Beam:
            other = Beam(intensity=1e11, particle_type=proton)
            other.setup_beam(
                dt=np.linspace(-1e-9, 1e-9, 100),
                dE=np.linspace(-1e6, 1e6, 100),
            )
            return other

        values = np.linspace(0.0, 1.0, 100)
        paths = {
            "setup_beam": lambda: self.beam.setup_beam(
                dt=values.copy(), dE=values.copy()
            ),
            "add_beam": lambda: self.beam.add_beam(other_beam()),
            "add_particles": lambda: self.beam.add_particles(
                DistributedArray(values.copy()),
                DistributedArray(values.copy()),
            ),
            "copy_coordinates_from": lambda: self.beam.copy_coordinates_from(
                other_beam()
            ),
            "plain_assignment": lambda: setattr(
                self.beam, "_dE", DistributedArray(values.copy())
            ),
        }
        for name, change in paths.items():
            with self.subTest(path=name):
                change()
                before = self._queue_kick()
                self.beam._dE.array_local  # noqa: B018
                self._assert_flushed(before)

    def test_deepcopied_beam_reads_flush(self) -> None:
        copied = deepcopy(self.beam)
        before = copied.kernel_call_dE.copy()
        self._queue_kick(copied)
        copied._dE.array_local  # noqa: B018
        self.assertEqual(backend.specials.kernel_call_queue.n_bytes, 0)
        self.assertFalse(np.array_equal(copied.kernel_call_dE, before))

    def test_size_queries_do_not_flush(self) -> None:
        # The RF station asks for `common_array_size` between the queued
        # kick and drift of every turn; flushing there would stop fusion.
        self._queue_kick()
        self.assertEqual(self.beam.common_array_size, 100)
        self.assertEqual(self.beam.n_macroparticles_partial(), 100)
        self.assertEqual(self.beam._dt.local_size, 100)
        self.assertGreater(backend.specials.kernel_call_queue.n_bytes, 0)


class TestRawCoordinateAccessStaysInBeam(BLonDTestCase):
    def test_raw_accessor_only_used_by_beam_base(self) -> None:
        # `_dt`/`_dE` flush on read, so direct access is safe; the one
        # way around the flush must stay behind `kernel_call_dt/dE`.
        allowed = (
            "core/beam/base.py",
            "core/beam/flushing_distributed_array.py",
        )
        hits = []
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(("legacy",) + allowed):
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if "array_local_without_flush" in line:
                    hits.append(f"{relative}:{number}: {line.strip()}")
        self.assertEqual(
            hits,
            [],
            "use `kernel_call_dt`/`kernel_call_dE`:\n" + "\n".join(hits),
        )
