import sys
import unittest
from unittest.mock import patch

import numpy as np

from blond.generals.distributed.distributed_array import mpi_is_distributed


class TestCallablesWithMPI(unittest.TestCase):
    def setUp(self):
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()
        if not mpi_active:
            self.skipTest("Only with MPI")

    def test_mpi_local_size(self):
        from blond.generals.distributed.helpers import mpi_local_size

        with self.assertWarnsRegex(
            UserWarning, "Because MPI is used, `global_size`"
        ):
            local_n = mpi_local_size(
                global_size=13,
                warning_hint="global_size",
            )  # assume `mpirun -n 2`
            self.assertEqual(local_n, 6)

    def test_distributed_arange(self):
        from blond.generals.distributed.helpers import distributed_arange

        da = distributed_arange(12, dtype=np.int32)
        if da._rank == 0:
            np.testing.assert_allclose(
                da.array_local,
                np.arange(0, 12),
                err_msg=f"{da._rank=} {da._size=}",
            )
        elif da._rank == 1:
            np.testing.assert_allclose(
                da.array_local,
                np.arange(12, 12 + 12),
                err_msg=f"{da._rank=} {da._size=}",
            )


class TestCallablesNoMPI(unittest.TestCase):
    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Only without MPI")

    def test_mpi_local_size(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop("blond.generals.distributed.helpers", None)
            from blond.generals.distributed.helpers import mpi_local_size

            self.assertEqual(
                10, mpi_local_size(global_size=10, warning_hint="")
            )

    def test_distributed_arange(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop("blond.generals.distributed.helpers", None)
            from blond.generals.distributed.helpers import distributed_arange

            da = distributed_arange(12, dtype=np.int32)
            np.testing.assert_allclose(da.array_local, np.arange(0, 12))
