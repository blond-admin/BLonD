from __future__ import annotations

import os
import subprocess  # Used to run external commands (like f2py)
import sys

_filepath = os.path.realpath(__file__)
_basepath = os.sep.join(_filepath.split(os.sep)[:-1])

from typing import List

# Name of the Python module to generate from Fortran code
_module_name32 = "libblond32"

# List of Fortran source files for 32-bit operations
_fortran_files32 = [
    "beam_phase_32.f90",
    "drift_32.f90",
    "histogram_32.f90",
    "kick_32.f90",
    "kick_induced_32.f90",
]

# Generate corresponding 64-bit file names by replacing "32" with "64"
_fortran_files32 = [os.path.join(_basepath, f) for f in _fortran_files32]


def compile_fortran_module(module_name: str, fortran_files: List[str]) -> bool:
    """
    Compile the Fortran source files into a Python module using f2py.

    Notes
    -----
    Applies high-performance compilation flags.

    """

    print(f"\nTrying to compile Fortran backend.")
    from numpy import f2py  # NOQA must be installed to be compiled / force exception

    # Optimization and parallelization flags for the Fortran compiler
    f90flags = (
        "-O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp"
    )

    # Construct the f2py command:
    # - `-c` for compile
    # - `-m` to specify the module name
    # - include all Fortran files
    # - pass optimization flags and link to OpenMP (`-lgomp`)
    cmd = (
        [
            "f2py",
            "-c",
            "-m",
            module_name,
        ]
        + fortran_files
        + [
            f"--f90flags='{f90flags}'",
            "-lgomp",
        ]
    )

    try:
        # Run the command and capture output
        print(" ".join(cmd))
        result = subprocess.run(
            cmd,
            check=True,  # Raise error on failure
            text=True,  # Decode output as string,
            stdout=sys.stdout,
            cwd=os.path.dirname(__file__),
        )
        print("Compilation successful.\n")
        print(result.stdout)  # Show compilation messages
    except subprocess.CalledProcessError as e:
        # If the compilation fails, show the error message
        print("Compilation failed:")
        print(e.stderr)
        return False

    return True  # Return True if compilation succeeds


def main_cli() -> None:
    """
    Entry point for running from the command line.
    Calls the Fortran compilation function.
    """
    sucess = compile_fortran_module(_module_name32, _fortran_files32)
    if sucess:
        sucess = compile_fortran_module(
            _module_name32.replace("32", "64"),
            [f.replace("32", "64") for f in _fortran_files32],
        )


# If the script is run directly (not imported), execute main_cli
if __name__ == "__main__":  # pragma: no cover
    main_cli()
