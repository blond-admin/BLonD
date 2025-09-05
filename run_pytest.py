from pathlib import Path

import pytest


def run_pytest(folder_path: str):
    """Run pytest on the specified folder with importlib import mode"""
    return pytest.main(["--import-mode=importlib", folder_path])


if __name__ == "__main__":
    unittest_path = Path("./unittests").resolve()
    run_pytest(str(unittest_path))
