import unittest

import pytest

from blond import setup_backend
from blond.testing.backend_testing import (
    BLonDTestCase,
    skip_if_cuda_unusable,
)


class TestCallables(BLonDTestCase):
    @pytest.mark.backend_mutation
    def test_setup_backend(self):
        for option in [
            "auto",
            "python",
            "cpp",
            "numba",
        ]:
            setup_backend(option)

    @pytest.mark.backend_mutation
    @pytest.mark.cupy
    def test_setup_backend_gpu(self):
        skip_if_cuda_unusable(self)
        setup_backend("cuda")

    def test_setup_backend_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend "):
            setup_backend("unknows")


if __name__ == "__main__":
    unittest.main()
