import pathlib
import re
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


class TestNoDirectCoordinateAccess(BLonDTestCase):
    def test_no_direct_dt_dE_outside_core_beam(self) -> None:
        pattern = re.compile(r"\._d(t|E)\b")
        allowed = ("legacy", "experimental", "core/beam", "core/backends")
        hits = []
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(allowed):
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line) and "`" not in line:
                    hits.append(f"{relative}:{number}: {line.strip()}")
        self.assertEqual(
            hits, [], "use the Beam accessors:\n" + "\n".join(hits)
        )
