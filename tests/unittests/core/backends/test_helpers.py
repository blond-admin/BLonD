import unittest

import pytest

from blond import setup_backend


class TestCallables(unittest.TestCase):
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
        try:
            import cupy as cp
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        setup_backend("cuda")

    def test_setup_backend_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend "):
            setup_backend("unknows")


if __name__ == "__main__":
    unittest.main()
