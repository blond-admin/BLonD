"""Executing pytest using Python instead of shell commands.

Authors
-------
Simon Lauber
"""

import os
from pathlib import Path

import pytest  # type: ignore


def run_pytest(folder_path: str) -> None:
    """Run pytest on the specified folder with importlib import mode."""
    pytest.main(["--import-mode=importlib", folder_path])


if __name__ == "__main__":
    cpu = True
    if cpu:
        os.environ["BLOND_BACKEND_MODE"] = "numba"
        os.environ["BLOND_BACKEND_BITS"] = "64"
        unittest_path = Path("./unittests").resolve()
        run_pytest(str(unittest_path))
    else:
        os.environ["BLOND_BACKEND_MODE"] = "cuda"
        os.environ["BLOND_BACKEND_BITS"] = "32"
        unittest_path = Path("./unittests").resolve()
        run_pytest(str(unittest_path))
