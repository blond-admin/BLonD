"""Execute pytest using Python instead of shell commands."""

import os
from pathlib import Path

import pytest  # type: ignore


def run_pytest(folder_path: str) -> None:
    """
    Run pytest on the specified folder with importlib import mode.

    Parameters
    ----------
    folder_path
        Path to the folder containing tests to run.
    """
    pytest.main(["--import-mode=importlib", folder_path, "--randomly-seed=1"])


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
