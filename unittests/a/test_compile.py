import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from blond.compile import main
from unittests.test_utils import is_master_or_dev_branch

this_directory = Path(os.path.dirname(os.path.realpath(__file__)))
blond_project_dir = this_directory.parent
print(f"{blond_project_dir=}")


class TestFunctions(unittest.TestCase):
    @unittest.skipUnless(
        is_master_or_dev_branch(),
        "Runs only on 'develop' or 'master' branch",
    )
    def test_main(self):
        """--parallel - -optimize - -gpu"""
        # mock argparse so that it behaves like it
        # was called by cli
        mock_args = MagicMock()
        options = {
            "parallel": True,
            "boost": None,
            "compiler": "g++",
            "with_fftw": False,
            "with_fftw_threads": False,
            "with_fftw_omp": False,
            "with_fftw_lib": None,
            "with_fftw_header": None,
            "flags": "",
            "libs": "",
            "libname": str(blond_project_dir / "blond/cpp_routines/libblond"),
            "optimize": True,
            "no_cpp": False,
            "gpu": "discover",
            "cuda_libname": str(blond_project_dir / "/blond/gpu/kernels"),
        }
        for key, value in options.items():
            mock_args.__setattr__(key, value)

        with patch(
            "blond.compile.argparse.ArgumentParser.parse_args", return_value=mock_args
        ):
            main()


if __name__ == "__main__":
    unittest.main()
