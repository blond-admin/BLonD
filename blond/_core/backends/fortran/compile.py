from __future__ import annotations

import os
import shutil
import subprocess  # Used to run external commands (like f2py)
import sys
from pathlib import Path

_filepath = os.path.realpath(__file__)
_basepath = os.path.dirname(_filepath)


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


def compile_fortran_module(module_name: str, fortran_files: list[str]) -> bool:
    """Compile the Fortran source files into a Python module using f2py.

    Notes
    -----
    Applies high-performance compilation flags.

    """
    print("\nTrying to compile Fortran backend.")
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
            f"--f90flags={f90flags}",
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
        # MOVE RESULT TO COMPILED SUBFOLDER
        # This is done because the compile script was crashing with changed
        # compilation target path. Probably can be fixed..
        move_compiled_file_to_subfolder(module_name)
    except subprocess.CalledProcessError as e:
        # If the compilation fails, show the error message
        print("Compilation failed:")
        print(e.stderr)
        return False

    return True  # Return True if compilation succeeds


def move_compiled_file_to_subfolder(module_name: str):
    from blond._generals._hashing import hash_in_folder

    folder = os.path.dirname(os.path.abspath(__file__))
    hash_ = hash_in_folder(
        folder=folder,
        extensions=(".py", ".f90"),
        recursive=False,
    )
    target = os.path.join(folder, "compiled", hash_)
    os.makedirs(target, exist_ok=True)
    matching_files = [
        f.name
        for f in Path(_basepath).iterdir()
        if f.is_file() and f.name.startswith(module_name)
    ]
    from_ = os.path.join(os.path.dirname(__file__), matching_files[0])
    to_ = os.path.join(target, matching_files[0])
    shutil.move(
        from_,
        to_,
    )


def main_cli() -> None:
    """Entry point for running from the command line.
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
