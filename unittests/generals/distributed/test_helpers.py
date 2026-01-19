import sys
import unittest
from unittest.mock import patch


class TestCallablesWithMPI(unittest.TestCase):
    def test_mpi_local_size(self):
        from blond.core.backends.mpi_distributed.callables import (
            mpi_is_active,
        )

        mpi_active = mpi_is_active()
        if not mpi_active:
            self.skipTest("Only with MPI")

        from blond.generals.distributed.helpers import mpi_local_size

        with self.assertWarnsRegex(
            UserWarning, "Because MPI is used, `global_size`"
        ):
            local_n = mpi_local_size(
                global_size=13,
                warning_hint="global_size",
            )  # assume `mpirun -n 2`
            self.assertEqual(local_n, 6)


class TestCallablesNoMPI(unittest.TestCase):
    def test_mpi_local_size(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop("blond.generals.distributed.helpers", None)
            from blond.generals.distributed.helpers import mpi_local_size

            mpi_local_size(global_size=10, warning_hint="")
