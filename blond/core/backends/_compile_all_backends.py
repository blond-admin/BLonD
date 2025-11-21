"""Compiler orchestrator for BLonD backends.

This script provides a unified entry point to compile all BLonD backends —
Fortran, CUDA, and C++ — in sequence. Each backend has its own compilation
CLI exposed through its respective module. Running this script ensures that
all native components are built and ready for use.

Typical usage:
    python compile_all_backends.py
"""


def main():  # pragma: no cover `main_cli_xxx` gets executed anyway by CI/CD  pipeline
    """Compile all BLonD backends sequentially.

    This function invokes the command-line compilation interfaces for
    Fortran, CUDA, and C++ backends in that order. It should be run
    when initializing or rebuilding the BLonD environment to ensure
    all compiled modules are up to date.
    """
    from blond.core.backends.cpp.compile import main_cli as main_cli_cpp
    from blond.core.backends.cuda.compile import main_cli as main_cli_cuda
    from blond.core.backends.fortran.compile import (
        main_cli as main_cli_fortran,
    )

    main_cli_fortran()
    main_cli_cuda()
    main_cli_cpp()


if __name__ == "__main__":  # pragma: no cover
    main()
