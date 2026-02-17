import sys
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from blond.generals.distributed.helpers import (
    MPI_RANK,
    distributed_arange,
    mpi_aware_random_generator_cpu,
    mpi_is_distributed,
    mpi_is_root,
)


@pytest.mark.mpi
class TestCallablesWithMPI(unittest.TestCase):
    def setUp(self):
        is_distributed = mpi_is_distributed()
        if not is_distributed:
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


class TestCallablesNoMPI(unittest.TestCase):
    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Only without MPI")

    def test_mpi_is_root(self):
        self.assertTrue(mpi_is_root())

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
