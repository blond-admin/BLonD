import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blond.core.backends.mpi_distributed.helpers import (
    MPI_RANK,
    distributed_arange,
    mpi_aware_random_generator_cpu,
    mpi_is_distributed,
    mpi_is_root,
)
from blond.testing.backend_testing import BLonDTestCase


@pytest.mark.mpi
class TestCallablesWithMPI(BLonDTestCase):
    def setUp(self):
        is_distributed = mpi_is_distributed()
        if not is_distributed:
            self.skipTest("Only with MPI")

    def test_mpi_local_size(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_local_size

        with self.assertWarnsRegex(
            UserWarning, "Because MPI is used, `global_size`"
        ):
            local_n = mpi_local_size(
                global_size=13,
                warning_hint="global_size",
            )  # assume `mpirun -n 2`
            self.assertEqual(local_n, 6)

    def test_distributed_arange(self):
        from blond.core.backends.mpi_distributed.helpers import (
            distributed_arange,
        )

        da = distributed_arange(12, dtype=np.int32)
        if da._rank == 0:
            np.testing.assert_allclose(
                da.copy_as_numpy(),
                np.arange(0, 12),
                err_msg=f"{da._rank=} {da._size=}",
            )
        elif da._rank == 1:
            np.testing.assert_allclose(
                da.copy_as_numpy(),
                np.arange(12, 12 + 12),
                err_msg=f"{da._rank=} {da._size=}",
            )

    def test_distributed_zeros(self):
        from blond.core.backends.mpi_distributed.helpers import (
            distributed_zeros,
        )

        da = distributed_zeros(12, dtype=np.int32)
        if da._rank == 0 or da._rank == 1:
            np.testing.assert_array_equal(
                da.copy_as_numpy(),
                np.zeros(12),
                err_msg=f"{da._rank=} {da._size=}",
            )

    def test_mpi_is_root(self):
        da = distributed_arange(12, dtype=np.int32)
        if da._rank == 0:
            self.assertTrue(mpi_is_root())
        if da._rank == 1:
            self.assertFalse(mpi_is_root())

    def test_mpi_aware_random_generator_cpu(self):
        seed = 1
        size_global = 12
        size_local = 6  # assume `mpirun -n 2`
        # not distributed
        random_generator_not_distributed = np.random.default_rng(
            seed=seed,
        )
        array_expected = random_generator_not_distributed.standard_normal(
            size=size_global
        )

        # distributed
        random_generator_distributed = mpi_aware_random_generator_cpu(
            seed=seed, n_forward_per_rank=size_local
        )
        array_local = random_generator_distributed.standard_normal(
            size=size_local
        )

        # compare

        if MPI_RANK == 0:
            np.testing.assert_equal(array_expected[0:6], array_local)
        if MPI_RANK == 1:
            np.testing.assert_equal(array_expected[6:12], array_local)


class TestCallablesNoMPI(BLonDTestCase):
    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Only without MPI")

    def test_mpi_is_root(self):
        self.assertTrue(mpi_is_root())

    def test_mpi_local_size(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop(
                "blond.core.backends.mpi_distributed.helpers", None
            )
            from blond.core.backends.mpi_distributed.helpers import (
                mpi_local_size,
            )

            self.assertEqual(
                10, mpi_local_size(global_size=10, warning_hint="")
            )

    def test_distributed_arange(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop(
                "blond.core.backends.mpi_distributed.helpers", None
            )
            from blond.core.backends.mpi_distributed.helpers import (
                distributed_arange,
            )

            da = distributed_arange(12, dtype=np.int32)
            np.testing.assert_allclose(da.copy_as_numpy(), np.arange(0, 12))

    def test_mpi_is_distributed_size_one(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 1):
            result = mpi_is_distributed()
        self.assertFalse(result)

    def test_mpi_is_distributed_size_one_returns_bool(self):
        """With MPI size 1 the result is `False`, not an implicit `None`."""
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 1):
            result = mpi_is_distributed()
        self.assertIs(result, False)

    def test_mpi_is_distributed_size_two(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 2):
            result = mpi_is_distributed()
        self.assertIs(result, True)

    def test_mpi_is_root_on_rank_zero(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_RANK", 0):
            result = mpi_is_root()
        self.assertIs(result, True)

    def test_mpi_is_root_on_non_root_rank(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_RANK", 1):
            result = mpi_is_root()
        self.assertIs(result, False)


def _current_helpers():
    """
    Return the helpers module currently in `sys.modules`.

    Other tests pop and re-import it, so a reference taken at
    import time can be stale.
    """
    return importlib.import_module(
        "blond.core.backends.mpi_distributed.helpers"
    )


class TestMpiRunOnRoot(BLonDTestCase):
    """`mpi_run_on_root` runs on rank 0 and fails on all ranks together."""

    def _patch_rank(self, rank: int, size: int, broadcast_result=None):
        helpers = _current_helpers()
        comm = MagicMock()
        comm.bcast.return_value = broadcast_result
        patches = (
            patch.object(helpers, "MPI_RANK", rank),
            patch.object(helpers, "MPI_SIZE", size),
            patch.object(helpers, "MPI_COMM_WORLD", comm),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return helpers, comm

    def test_serial_runs_function_without_communication(self):
        helpers, comm = self._patch_rank(rank=0, size=1)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_called_once_with()
        comm.bcast.assert_not_called()

    def test_serial_reraises_original_exception(self):
        helpers, _ = self._patch_rank(rank=0, size=1)
        function = MagicMock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(OSError, "disk full"):
            helpers.mpi_run_on_root(function)

    def test_root_broadcasts_success(self):
        helpers, comm = self._patch_rank(rank=0, size=2)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_called_once_with()
        comm.bcast.assert_called_once_with(None, root=0)

    def test_root_broadcasts_failure_and_reraises(self):
        helpers, comm = self._patch_rank(rank=0, size=2)
        function = MagicMock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(OSError, "disk full"):
            helpers.mpi_run_on_root(function)
        comm.bcast.assert_called_once()
        message = comm.bcast.call_args.args[0]
        self.assertIn("disk full", message)
        self.assertIn("OSError", message)

    def test_non_root_skips_function(self):
        helpers, comm = self._patch_rank(rank=1, size=2)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_not_called()
        comm.bcast.assert_called_once_with(None, root=0)

    def test_non_root_raises_when_root_failed(self):
        helpers, _ = self._patch_rank(
            rank=1, size=2, broadcast_result="OSError('disk full')"
        )
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            helpers.mpi_run_on_root(MagicMock())


class TestMpiLocalGpuIndex(BLonDTestCase):
    """Each rank of a node picks its own GPU."""

    def _patch(self, size: int, local_rank: int):
        helpers = _current_helpers()
        for patcher in (
            patch.object(helpers, "MPI_SIZE", size),
            patch.object(helpers, "MPI_LOCAL_RANK", local_rank),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return helpers

    def test_serial_keeps_current_device(self):
        helpers = self._patch(size=1, local_rank=0)
        self.assertIsNone(helpers.mpi_local_gpu_index(n_devices=4))

    def test_ranks_spread_over_devices(self):
        indices = [
            self._patch(size=8, local_rank=local_rank).mpi_local_gpu_index(
                n_devices=4
            )
            for local_rank in range(8)
        ]
        self.assertEqual(indices, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_one_visible_device_per_rank(self):
        """`CUDA_VISIBLE_DEVICES` per rank leaves one device: index 0."""
        helpers = self._patch(size=8, local_rank=5)
        self.assertEqual(helpers.mpi_local_gpu_index(n_devices=1), 0)

    def test_local_rank_without_mpi_is_zero(self):
        helpers = _current_helpers()
        if helpers.MPI is not None:
            self.skipTest("Only without MPI")
        self.assertEqual(helpers.MPI_LOCAL_RANK, 0)


class TestMpiCudaAware(BLonDTestCase):
    """`BLOND_MPI_CUDA_AWARE` opts in to passing GPU memory to MPI."""

    def _cuda_aware(self, environ: dict[str, str]) -> bool:
        with patch.dict(os.environ, environ, clear=True):
            return _current_helpers().mpi_cuda_aware()

    def test_default_is_not_cuda_aware(self):
        self.assertIs(self._cuda_aware({}), False)

    def test_true_and_false(self):
        self.assertIs(self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "True"}), True)
        self.assertIs(
            self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "False"}), False
        )

    def test_invalid_value_raises(self):
        with self.assertRaisesRegex(ValueError, "BLOND_MPI_CUDA_AWARE"):
            self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "yes"})


@pytest.mark.mpi
class TestMpiLocalRankWithMPI(BLonDTestCase):
    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")

    def test_single_node_local_rank_equals_rank(self):
        """On one node (as in the test run), local and global rank agree."""
        helpers = _current_helpers()
        self.assertEqual(helpers.MPI_LOCAL_RANK, helpers.MPI_RANK)


@pytest.mark.mpi
class TestMpiRunOnRootWithMPI(BLonDTestCase):
    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")

    def test_failure_on_root_raises_on_every_rank(self):
        from blond.core.backends.mpi_distributed.helpers import (
            mpi_run_on_root,
        )

        def fail():
            raise OSError("disk full")

        expected = OSError if mpi_is_root() else RuntimeError
        with self.assertRaisesRegex(expected, "disk full"):
            mpi_run_on_root(fail)


_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


class TestMpiLaunched(BLonDTestCase):
    """`mpi_launched` decides whether MPI is initialised at all."""

    def _mpi_launched(self, environ):
        """Call `mpi_launched` with `environ` as the whole environment.

        `clear=True` matters: under `mpirun` the real environment already
        holds launcher variables.
        """
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        with patch.dict(os.environ, environ, clear=True):
            return mpi_launched()

    def test_plain_python_is_not_launched(self):
        self.assertIs(self._mpi_launched({}), False)

    def test_launcher_variables_are_detected(self):
        for key in _LAUNCHER_ENV_KEYS:
            with self.subTest(key=key):
                self.assertIs(self._mpi_launched({key: "0"}), True)

    def test_override_forces_mpi_on(self):
        self.assertIs(self._mpi_launched({"BLOND_LOAD_MPI": "True"}), True)

    def test_override_forces_mpi_off_under_launcher(self):
        environ = {"BLOND_LOAD_MPI": "False", "OMPI_COMM_WORLD_SIZE": "2"}
        self.assertIs(self._mpi_launched(environ), False)

    def test_invalid_override_raises(self):
        with self.assertRaises(ValueError):
            self._mpi_launched({"BLOND_LOAD_MPI": "yes"})


class TestImportDoesNotInitialiseMpi(BLonDTestCase):
    """Outside an MPI launcher, `import blond` must not call `MPI_Init`.

    `MPI_Init` costs ~0.5 s per import and sets up the MPI runtime in
    processes that never use MPI.
    """

    def test_import_blond_does_not_import_mpi4py_mpi(self):
        env = os.environ.copy()
        for key in (*_LAUNCHER_ENV_KEYS, "BLOND_LOAD_MPI", "PYCHARM_HOSTED"):
            env.pop(key, None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, blond; print('mpi4py.MPI' in sys.modules)",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.split()[-1], "False")


class TestImportMpiUnderLauncher(BLonDTestCase):
    """Module setup of `helpers` when an MPI launcher is detected.

    A fake `mpi4py` stands in for the real one, so that the test process
    itself never calls `MPI_Init`.
    """

    _HELPERS = "blond.core.backends.mpi_distributed.helpers"

    def _import_fresh_helpers(self, mpi4py_module):
        with (
            patch.dict(os.environ, {"BLOND_LOAD_MPI": "True"}),
            patch.dict(sys.modules, {"mpi4py": mpi4py_module}),
        ):
            sys.modules.pop(self._HELPERS, None)
            return importlib.import_module(self._HELPERS)

    def test_rank_and_size_come_from_comm_world(self):
        comm_world = MagicMock()
        comm_world.Get_rank.return_value = 1
        comm_world.Get_size.return_value = 4
        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI.COMM_WORLD = comm_world

        helpers = self._import_fresh_helpers(fake_mpi4py)

        self.assertIs(helpers.MPI, fake_mpi4py.MPI)
        self.assertIs(helpers.MPI_COMM_WORLD, comm_world)
        self.assertEqual(helpers.MPI_RANK, 1)
        self.assertEqual(helpers.MPI_SIZE, 4)

    def test_missing_mpi4py_warns_and_runs_serially(self):
        with self.assertWarns(ImportWarning):
            helpers = self._import_fresh_helpers(None)

        self.assertIsNone(helpers.MPI)
        self.assertIsNone(helpers.MPI_COMM_WORLD)
        self.assertEqual(helpers.MPI_SIZE, 1)
