from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import warnings
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional

_filepath = os.path.realpath(__file__)
_basepath = os.sep.join(_filepath.split(os.sep)[:-1])

default_libname = "libblond"

cpp_files = [
    "kick.cpp",
    "drift.cpp",
    "linear_interp_kick.cpp",
    "histogram.cpp",
    # "music_track.cpp",
    # "blondmath.cpp",
    # "fast_resonator.cpp",
    "beam_phase.cpp",
    # "fft.cpp",
    # "openmp.cpp",
]
cpp_files = [os.path.join(_basepath, f) for f in cpp_files]


def run_compile(command: List[str], libname: str) -> int:
    if os.path.exists(libname):
        os.remove(libname)
    print(" ".join(command))
    ret = subprocess.run(command, check=False)
    if ret.returncode != 0 or not os.path.isfile(libname):
        return -1
    else:
        return 0


def compile_cpp_library(
    with_fftw: bool = False,
    with_fftw_threads: bool = False,
    with_fftw_omp: bool = False,
    with_fftw_lib: Optional[str] = None,
    with_fftw_header: Optional[str] = None,
    boost: Optional[str] = None,
    compiler: str = "g++",
    libs: str = "",
    parallel: bool = False,
    flags: str = "",
    optimize: bool = False,
    libname: Optional[str] = None,
) -> None:
    """
    Compile the BLonD C++ library with optional FFTW, OpenMP, and Boost support.

    Parameters
    ----------
    with_fftw : bool
        Whether to include FFTW3 support.
    with_fftw_threads : bool
        Enable multi-threaded FFTW3 usage.
    with_fftw_omp : bool
        Enable OpenMP FFTW3 usage.
    with_fftw_lib : str or None
        Path to the FFTW3 library file (.so, .dll). If None, default paths will be used.
    with_fftw_header : str or None
        Path to the FFTW3 header files. If None, default paths will be used.
    boost : str or None
        Path to the Boost library, or an empty string to use system default. If None, Boost will not be used.
    compiler : str
        The C++ compiler to use, e.g., "g++".
    libs : str
        Additional libraries required for compilation, provided as a space-separated string.
    parallel : bool
        If True, compile with OpenMP for multi-threaded execution.
    flags : str
        Additional compiler flags as a space-separated string (e.g., "-O2 -Wall").
    optimize : bool
        If True, enable post-compilation optimizations.
    libname : str
        Path and name of the output library (without file extension).

    Returns
    -------
    None
        The function performs compilation and does not return any value.

    Notes
    -----
    This function assumes the presence of a Makefile or equivalent build system
    capable of processing the supplied options.
    """
    print(f"\nTrying to compile C++ backend.")

    if libname is None:
        libname = os.path.join(_basepath, default_libname)
    # EXAMPLE FLAGS: -Ofast -std=c++11 -fopt-info-vec -march=native
    #                -mfma4 -fopenmp -ftree-vectorizer-verbose=1 '-ffast-math'

    cflags = [
        "-O3",
        "-std=c++11",
        "-shared",
        "-D_USE_MATH_DEFINES",
        # Necessary on windows, as here the M_PI etc.
        # are not defined by mingw without the flag.
    ]
    # Some additional warning reporting related flags
    cflags += [
        "-Wall",
        "-Wno-unknown-pragmas",
    ]

    for file in cpp_files:
        assert os.path.isfile(file), f"{file=}"

    with_fftw = any(
        [
            with_fftw,
            with_fftw_threads,
            with_fftw_omp,
            with_fftw_lib,
            with_fftw_header,
        ]
    )

    # Get boost path
    boost_path = None
    if boost is not None:
        if boost:
            boost_path = os.path.abspath(boost)
        else:
            boost_path = ""
        cflags += ["-I", boost_path, "-DBOOST"]

    if libs:
        libs_ = libs.split()
    else:
        libs_ = []

    if parallel:
        cflags += ["-fopenmp", "-DPARALLEL", "-D_GLIBCXX_PARALLEL"]

    if flags:
        cflags += flags.split()

    fftw_cflags, fftw_libs = prepare_fftw(
        with_fftw=with_fftw,
        with_fftw_header=with_fftw_header,
        with_fftw_lib=with_fftw_lib,
        with_fftw_omp=with_fftw_omp,
        with_fftw_threads=with_fftw_threads,
    )

    cflags, libname_double, libname_single = prepare_cflags(
        cflags=cflags,
        compiler=compiler,
        libname=libname,
        optimize=optimize,
    )

    # Report the compilation options
    print("Enable Multi-threaded code: ", parallel)
    print("Use boost: ", boost is not None)
    if boost is not None:
        print("Boost installation path: ", boost_path)
    print("Link with FFTW3: ", with_fftw)
    if with_fftw:
        print(
            "Parallel FFTW3:",
            with_fftw_threads or with_fftw_omp,
        )
    if with_fftw_lib or with_fftw_header:
        print("FFTW3 Library path: ", with_fftw_lib)
        print("FFTW3 Headers path: ", with_fftw_header)
    print("C++ Compiler: ", compiler)
    compiler_version = (
        subprocess.run(
            [compiler, "--version"], capture_output=True, check=False
        )
        .stdout.decode()
        .split("\n")[0]
    )
    print("Compiler version: ", compiler_version)

    print("Compiler flags: ", " ".join(cflags))
    print("Extra libraries: ", " ".join(libs_))

    command = (
        [compiler]
        + cflags
        + ["-DUSEFLOAT"]
        + cpp_files
        + libs_
        + ["-o", libname_single]
    )
    print("\nCompiling the single-precision (32-bit) C++ library")
    if with_fftw:
        msg = (
            "The FFTW Library is only compiled for  double-precision (64-bit)."
            " For single-precision, the FFTW Library is ignored."
        )
        warnings.warn(msg)
    ret = run_compile(command, libname_single)
    if ret != 0:
        print("There was a compilation error.")
    else:
        # Verify that the libraries have been compiled
        try:
            if ("win" in sys.platform) and hasattr(os, "add_dll_directory"):
                _ = ctypes.CDLL(libname_single, winmode=0)
            else:
                _ = ctypes.CDLL(libname_single)
            print("Compiled successfully.")
        except Exception as exception:
            print("Compilation failed.")
            print(exception)

    command = (
        [compiler]
        + cflags
        + fftw_cflags
        + cpp_files
        + libs_
        + fftw_libs
        + ["-o", libname_double]
    )
    print("\nCompiling the double-precision (64-bit) C++ library")
    ret = run_compile(command, libname_double)
    if ret != 0:
        print("There was a compilation error.")
    else:
        # Verify that the libraries have been compiled
        try:
            if ("win" in sys.platform) and hasattr(os, "add_dll_directory"):
                _ = ctypes.CDLL(libname_double, winmode=0)
            else:
                _ = ctypes.CDLL(libname_double)
            print("Compiled successfully.")
        except Exception as exception:
            print("Compilation failed.")
            print(exception)


from typing import List


def prepare_cflags(
    cflags: List[str],
    compiler: str,
    libname: str,
    optimize: bool,
) -> tuple[List[str], str, str]:
    if "posix" in os.name:
        cflags += ["-fPIC"]
        if optimize:
            if "-ffast-math" not in cflags:
                cflags += ["-ffast-math"]
            cflags = add_avx_flags(
                cflags=cflags,
                compiler=compiler,
            )

        root, ext = os.path.splitext(libname)
        if not ext:
            ext = ".so"
        libname_single = os.path.abspath(root + "_single" + ext)
        libname_double = os.path.abspath(root + "_double" + ext)

    elif "win" in sys.platform:
        root, ext = os.path.splitext(libname)
        if not ext:
            ext = ".dll"

        libname_single = os.path.abspath(root + "_single" + ext)
        libname_double = os.path.abspath(root + "_double" + ext)

        if hasattr(os, "add_dll_directory"):
            directory, _ = os.path.split(libname_double)
            os.add_dll_directory(directory)

    else:
        raise NameError(f"Unknown operating system: {sys.platform=}")
    return cflags, str(libname_double), str(libname_single)


def prepare_fftw(
    with_fftw: bool,
    with_fftw_header: Optional[str] = None,
    with_fftw_lib: Optional[str] = None,
    with_fftw_omp: Optional[bool] = False,
    with_fftw_threads: Optional[bool] = False,
) -> Tuple[List[str], list[str]]:
    fftw_cflags = []
    fftw_libs = []
    if with_fftw:
        fftw_cflags += ["-DUSEFFTW3"]
        if with_fftw_lib is not None:
            fftw_libs += ["-L", with_fftw_lib]
        if with_fftw_header is not None:
            fftw_cflags += ["-I", with_fftw_header]
        if "win" in sys.platform:
            fftw_libs += ["-lfftw3-3"]
        else:
            fftw_libs += ["-lfftw3", "-lfftw3f"]
            if with_fftw_omp:
                fftw_cflags += ["-DFFTW3PARALLEL"]
                fftw_libs += ["-lfftw3_omp", "-lfftw3f_omp"]
            elif with_fftw_threads:
                fftw_cflags += ["-DFFTW3PARALLEL"]
                fftw_libs += ["-lfftw3_threads", "-lfftw3f_threads"]
    return fftw_cflags, fftw_libs


def add_avx_flags(cflags: List[str], compiler: str) -> List[str]:
    # Check compiler defined directives
    # This is compatible with python3.6 - python 3.9
    # The universal_newlines argument transforms output to text (from binary)
    ret = subprocess.run(
        [
            compiler
            + ' -march=native -dM -E - < /dev/null | egrep "SSE|AVX|FMA"'
        ],
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    # If we have an error
    if ret.returncode != 0:
        print(
            "Compiler auto-optimization did not work. Error: ",
            ret.stdout,
        )
    else:
        # Format the output list
        stdout = (
            ret.stdout.replace("#define ", "")
            .replace("__ 1", "")
            .replace("__", "")
            .split("\n")
        )
        # following options exist only on x86 processors
        if "arm" not in platform.machine():
            # Add the appropriate vectorization flag (not use avx512)
            if "AVX2" in stdout:
                cflags += ["-mavx2"]
            elif "AVX" in stdout:
                cflags += ["-mavx"]
            elif "SSE4_2" in stdout or "SSE4_1" in stdout:
                cflags += ["-msse4"]
            elif "SSE3" in stdout:
                cflags += ["-msse3"]
            else:
                cflags += ["-msse"]

            # Add FMA if supported
            if "FMA" in stdout:
                cflags += ["-mfma"]
    return cflags


def main_cli() -> None:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(
        description="Script used to compile the C++ libraries needed by BLonD.",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        type=bool,
        default=True,
        help="Produce Multi-threaded code. Use the environment"
        " variable OMP_NUM_THREADS=xx to control the number of"
        " threads that will be used."
        " Default: Serial code",
    )

    parser.add_argument(
        "-b",
        "--boost",
        type=str,
        nargs="?",
        const="",
        help="Use boost library to speedup synchrotron radiation"
        " routines. If the installation path of boost differs"
        " from the default, you have to pass it as an argument."
        " Default: Boost will not be used",
    )

    parser.add_argument(
        "-c",
        "--compiler",
        type=str,
        default="g++",
        help="C++ compiler that will be used to compile the"
        " source files. Default: g++",
    )

    parser.add_argument(
        "--with-fftw",
        action="store_true",
        help="Use the FFTs from FFTW3.",
    )

    parser.add_argument(
        "--with-fftw-threads",
        action="store_true",
        help="Use the multi-threaded FFTs from FFTW3.",
    )

    parser.add_argument(
        "--with-fftw-omp",
        action="store_true",
        help="Use the OMP FFTs from FFTW3.",
    )

    parser.add_argument(
        "--with-fftw-lib",
        type=str,
        help="Path to the FFTW3 library (.so, .dll).",
    )

    parser.add_argument(
        "--with-fftw-header",
        type=str,
        help="Path to the FFTW3 header files.",
    )

    parser.add_argument(
        "--flags",
        type=str,
        default="",
        help="Additional compile flags.",
    )

    parser.add_argument(
        "--libs",
        type=str,
        default="",
        help="Any extra libraries needed to compile",
    )

    parser.add_argument(
        "-libname",
        "--libname",
        type=str,
        default=os.path.join(_basepath, "libblond"),
        help="The C++ library name, without the file extension.",
    )

    parser.add_argument(
        "-optimize",
        "--optimize",
        type=bool,
        default=True,
        help="Auto optimize the compiled library.",
    )

    # Parse command line options
    args = vars(parser.parse_args())
    compile_cpp_library(
        with_fftw=args["with_fftw"],
        with_fftw_threads=args["with_fftw_threads"],
        with_fftw_omp=args["with_fftw_omp"],
        with_fftw_lib=args["with_fftw_lib"],
        with_fftw_header=args["with_fftw_header"],
        boost=args["boost"],
        compiler=args["compiler"],
        libs=args["libs"],
        parallel=args["parallel"],
        flags=args["flags"],
        optimize=args["optimize"],
        libname=args["libname"],
    )


if __name__ == "__main__":
    main_cli()
