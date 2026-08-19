from pathlib import Path

import pytest


def run_pytest(folder_path: str):
    """Run pytest on the specified folder"""
    return pytest.main([folder_path])


if __name__ == "__main__":
    cavity_loops_with_sparse_profiles_path = Path(".").resolve()
    run_pytest(str(cavity_loops_with_sparse_profiles_path))
