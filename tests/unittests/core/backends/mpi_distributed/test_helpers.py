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


_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


class TestMpiLaunched(BLonDTestCase):
    """`mpi_launched` decides whether MPI is initialised at all."""

    def test_plain_python_is_not_launched(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        self.assertIs(mpi_launched(environ={}), False)

    def test_launcher_variables_are_detected(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        for key in _LAUNCHER_ENV_KEYS:
            with self.subTest(key=key):
                self.assertIs(mpi_launched(environ={key: "0"}), True)

    def test_override_forces_mpi_on(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        self.assertIs(mpi_launched(environ={"BLOND_USE_MPI": "True"}), True)

    def test_override_forces_mpi_off_under_launcher(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        environ = {"BLOND_USE_MPI": "False", "OMPI_COMM_WORLD_SIZE": "2"}
        self.assertIs(mpi_launched(environ=environ), False)

    def test_invalid_override_raises(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        with self.assertRaises(ValueError):
            mpi_launched(environ={"BLOND_USE_MPI": "yes"})


class TestImportDoesNotInitialiseMpi(BLonDTestCase):
    """Outside an MPI launcher, `import blond` must not call `MPI_Init`.

    `MPI_Init` costs ~0.5 s per import and sets up the MPI runtime in
    processes that never use MPI.
    """

    def test_import_blond_does_not_import_mpi4py_mpi(self):
        env = os.environ.copy()
        for key in (*_LAUNCHER_ENV_KEYS, "BLOND_USE_MPI", "PYCHARM_HOSTED"):
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
            patch.dict(os.environ, {"BLOND_USE_MPI": "True"}),
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
