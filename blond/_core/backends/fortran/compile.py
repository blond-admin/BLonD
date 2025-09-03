from __future__ import annotations

import subprocess  # Used to run external commands (like f2py)

# Name of the Python module to generate from Fortran code
module_name = "libblond"

# List of Fortran source files for 32-bit operations
fortran_files = [
    "beam_phase_32.f90",
    "drift_32.f90",
    "histogram_32.f90",
    "kick_32.f90",
    "kick_induced_32.f90",
]

# Generate corresponding 64-bit file names by replacing "32" with "64"
fortran_file = fortran_files + [f.replace("32", "64") for f in fortran_files]


def compile_fortran_module():
    """
    Compile the Fortran source files into a Python module using f2py.

    Notes
    -----
    Applies high-performance compilation flags.

    """
    print("Compiling Fortran module using f2py...")

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
        ["f2py", "-c", "-m", module_name]
        + fortran_files
        + [
            f"--f90flags='{f90flags}'",
            "-lgomp",
        ]
    )

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            check=True,  # Raise error on failure
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode output as string
        )
        print("Compilation successful.\n")
        print(result.stdout)  # Show compilation messages
    except subprocess.CalledProcessError as e:
        # If the compilation fails, show the error message
        print("Compilation failed:")
        print(e.stderr)
        return False

    return True  # Return True if compilation succeeds


def main_cli():
    """
    Entry point for running from the command line.
    Calls the Fortran compilation function.
    """
    sucess = compile_fortran_module()


# If the script is run directly (not imported), execute main_cli
if __name__ == "__main__":
    main_cli()
