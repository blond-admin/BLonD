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
            "cuda",
        ]:
            setup_backend(option)


if __name__ == "__main__":
    unittest.main()
