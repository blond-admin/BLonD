import sys
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from blond import backend, copy_to_cpu
from blond.generals.cupy_.no_cupy_import import is_cupy_array
from blond.generals.distributed.distributed_array import (
    DistributedArray,
    concatenate,
)
from blond.generals.distributed.helpers import mpi_barrier, mpi_is_distributed
from blond.generals.exceptions_ import ArrayPrecisionError
from blond.testing.backend_testing import skip_if_no_cupy


@pytest.mark.mpi
class TestDistributedArray(unittest.TestCase):
    def setUp(self):
        from blond.generals.distributed.distributed_array import (
            DistributedArray,
        )

        rng = np.random.default_rng(0)
        self.array = np.astype(
            rng.normal(loc=0, scale=1.0, size=128), backend.float
        )
        self.distributed_array = DistributedArray(
            backend.array(self.array.copy())
        )

    def test_histogram(self):
        mpi_active = mpi_is_distributed()

        expected, _ = np.histogram(self.array, bins=8)
        if mpi_active:
            self.distributed_array.mpi_scatter()
        actual = self.distributed_array.histogram(bins=8)
        np.testing.assert_allclose(expected, copy_to_cpu(actual))


if __name__ == "__main__":
    unittest.main()
