# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""

@author: Danilo Quartullo, Konstantinos Iliakis
"""

# MAKE SURE YOU HAVE GCC 4.8.1 OR LATER VERSIONS ON YOUR SYSTEM LINKED TO YOUR
# SYSTEM PATH.IF YOU ARE ON CERN-LXPLUS YOU CAN TYPE FROM CONSOLE
# source /afs/cern.ch/sw/lcg/contrib/gcc/4.8.1/x86_64-slc6/setup.sh
# TO GET GCC 4.8.1 64 BIT. IN GENERAL IT IS ADVISED TO USE PYTHON 64 BIT PLUS
# GCC 64 BIT.

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import warnings

try:
    # Early import of cupy.
    # This fixes warnings that occurred when
    # importing `cupy` later inside `compile_cuda_library`
    import cupy as cp
except ImportError:
    pass  # ignore missing cupy for users without GPU


def main():
    """Compiles the blond C++ and/or CUDA library."""
    path = os.path.realpath(__file__)
    basepath = os.sep.join(path.split(os.sep)[:-1])

    parser = argparse.ArgumentParser(
        description="Script used to compile the C++ (and CUDA) libraries needed by BLonD.",
        epilog="All arguments can be controlled with the environment variable BLOND_COMPILE_OPTS. E.g.: BLOND_COMPILE_OPTS='-p,--flags=-O0 -g'",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        default=False,
        action="store_true",
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
        "--with-fftw", action="store_true", help="Use the FFTs from FFTW3."
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
        "--with-fftw-header", type=str, help="Path to the FFTW3 header files."
    )

    parser.add_argument(
        "--flags", type=str, default="", help="Additional compile flags."
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
        default=os.path.join(basepath, "cpp_routines/libblond"),
        help="The C++ library name, without the file extension.",
    )

    parser.add_argument(
        "-optimize",
        "--optimize",
        action="store_true",
        help="Auto optimize the compiled library.",
    )

    parser.add_argument(
        "-no-cpp",
        "--no-cpp",
        action="store_true",
        help="Do not compile the C++ library.",
    )

    parser.add_argument(
        "-gpu",
        "--gpu",
        nargs="?",
        const="discover",
        default=None,
        help="Compile the GPU kernels too."
        "Default: Only compile the C++ library.",
    )

    parser.add_argument(
        "-cuda-libname",
        "--cuda-libname",
        type=str,
        default=os.path.join(basepath, "gpu/kernels"),
        help="The CUDA library name, without the file extension.",
    )

    # Additional libs needed to compile the blond library
    libs = []

    # EXAMPLE FLAGS: -Ofast -std=c++11 -fopt-info-vec -march=native
    #                -mfma4 -fopenmp -ftree-vectorizer-verbose=1 '-ffast-math'
    cflags = ["-O3", "-std=c++11", "-shared"]
    # Some additional warning reporting related flags
    cflags += ["-Wall", "-Wno-unknown-pragmas", "-D_USE_MATH_DEFINES"]
    # The last flag is mainly necessary on windows, as here the M_PI etc.
    # are not defined by mingw without the flag.

    float_flags = ["-DUSEFLOAT"]

    nvcc_flags = ["--cubin", "-O3", "--use_fast_math", "-maxrregcount", "32"]

    cpp_files = [
        os.path.join(basepath, "cpp_routines/kick.cpp"),
        os.path.join(basepath, "cpp_routines/drift.cpp"),
        os.path.join(basepath, "cpp_routines/linear_interp_kick.cpp"),
        os.path.join(basepath, "cpp_routines/histogram.cpp"),
        os.path.join(basepath, "cpp_routines/music_track.cpp"),
        os.path.join(basepath, "cpp_routines/blondmath.cpp"),
        os.path.join(basepath, "cpp_routines/fast_resonator.cpp"),
        os.path.join(basepath, "cpp_routines/beam_phase.cpp"),
        os.path.join(basepath, "cpp_routines/fft.cpp"),
        os.path.join(basepath, "cpp_routines/openmp.cpp"),
        os.path.join(basepath, "toolbox/tomoscope.cpp"),
        os.path.join(
            basepath, "synchrotron_radiation/synchrotron_radiation.cpp"
        ),
        os.path.join(basepath, "beam/sparse_histogram.cpp"),
    ]

    cuda_files = [
        os.path.join(basepath, "gpu/kernels.cu"),
    ]

    nvcc = "nvcc"

    # Get nvcc from CUDA_PATH
    cuda_path = os.getenv("CUDA_PATH", default="")
    if cuda_path != "":
        nvcc = cuda_path + "/bin/nvcc"

    # Parse command line options
    args = vars(parser.parse_args())

    # Parse environment variable (BLOND_COMPILE_OPTS) options
    # if 'BLOND_COMPILE_OPTS' in os.environ:
    #     env_args_lst = os.environ['BLOND_COMPILE_OPTS'].split(',')
    #     env_args = vars(parser.parse_args(env_args_lst))
    #     args.update(env_args)

    if not args["no_cpp"]:
        compile_cpp_library(args, cflags, float_flags, libs, cpp_files)

    if args["gpu"]:
        compile_cuda_library(args, nvcc_flags, float_flags, cuda_files, nvcc)


def compile_cpp_library(args, cflags, float_flags, libs, cpp_files):
    # Check if we need to compile with FFTW
    with_fftw = (
        args["with_fftw"]
        or args["with_fftw_threads"]
        or args["with_fftw_omp"]
        or (args["with_fftw_lib"] is not None)
        or (args["with_fftw_header"] is not None)
    )

    # Get boost path
    boost_path = None
    if args["boost"] is not None:
        if args["boost"]:
            boost_path = os.path.abspath(args["boost"])
        else:
            boost_path = ""
        cflags += ["-I", boost_path, "-DBOOST"]
    compiler = args["compiler"]

    if args["libs"]:
        libs = args["libs"].split()

    if args["parallel"]:
        cflags += ["-fopenmp", "-DPARALLEL", "-D_GLIBCXX_PARALLEL"]

    if args["flags"]:
        cflags += args["flags"].split()

    fftw_cflags = []
    fftw_libs = []
    if with_fftw:
        fftw_cflags += ["-DUSEFFTW3"]
        if args["with_fftw_lib"] is not None:
            fftw_libs += ["-L", args["with_fftw_lib"]]
        if args["with_fftw_header"] is not None:
            fftw_cflags += ["-I", args["with_fftw_header"]]
        if "win" in sys.platform:
            fftw_libs += ["-lfftw3-3"]
        else:
            fftw_libs += ["-lfftw3", "-lfftw3f"]
            if args["with_fftw_omp"]:
                fftw_cflags += ["-DFFTW3PARALLEL"]
                fftw_libs += ["-lfftw3_omp", "-lfftw3f_omp"]
            elif args["with_fftw_threads"]:
                fftw_cflags += ["-DFFTW3PARALLEL"]
                fftw_libs += ["-lfftw3_threads", "-lfftw3f_threads"]

    if "posix" in os.name:
        cflags += ["-fPIC"]
        if args["optimize"]:
            if "-ffast-math" not in cflags:
                cflags += ["-ffast-math"]
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

        root, ext = os.path.splitext(args["libname"])
        if not ext:
            ext = ".so"
        libname_single = os.path.abspath(root + "_single" + ext)
        libname_double = os.path.abspath(root + "_double" + ext)

    elif "win" in sys.platform:
        root, ext = os.path.splitext(args["libname"])
        if not ext:
            ext = ".dll"

        libname_single = os.path.abspath(root + "_single" + ext)
        libname_double = os.path.abspath(root + "_double" + ext)

        if hasattr(os, "add_dll_directory"):
            directory, _ = os.path.split(libname_double)
            os.add_dll_directory(directory)

    else:
        print(
            "YOU ARE NOT USING A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING..."
        )
        sys.exit(-1)

    # Report the compilation options
    print("Enable Multi-threaded code: ", args["parallel"])
    print("Use boost: ", args["boost"] is not None)
    if args["boost"] is not None:
        print("Boost installation path: ", boost_path)
    print("Link with FFTW3: ", with_fftw)
    if with_fftw:
        print(
            "Parallel FFTW3:",
            args["with_fftw_threads"] or args["with_fftw_omp"],
        )
    if args["with_fftw_lib"] or args["with_fftw_header"]:
        print("FFTW3 Library path: ", args["with_fftw_lib"])
        print("FFTW3 Headers path: ", args["with_fftw_header"])
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
    print("Extra libraries: ", " ".join(libs))

    command = (
        [compiler]
        + cflags
        + float_flags
        + cpp_files
        + libs
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
        + libs
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


def compile_cuda_library(args, nvccflags, float_flags, cuda_files, nvcc):
    # Compile the GPU library
    # print('\n' + ''.join(['='] * 80))
    import cupy as cp  # force exception, if something is wrong with the installation

    print("\nCompiling the CUDA library")
    if args["gpu"] == "discover":
        print("Discovering the device compute capability..")

        dev = cp.cuda.Device(0)
        dev_name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
        comp_capability = dev.compute_capability
        print(f"Device name {dev_name}")
    elif args["gpu"] is not None:
        comp_capability = args["gpu"]

    print(
        f"Compiling the CUDA library for compute capability {comp_capability}."
    )

    # Add the -arch required argument
    nvccflags += ["-arch", f"sm_{comp_capability}"]

    # Get the CuPy header files location
    path = cp.__file__.split("/")[:-1]  # remove __init__.py from path
    path.extend(["_core", "include"])
    cupyloc = os.path.join("/".join(path))

    print("CUDA Compiler: ", nvcc)
    compiler_version = (
        subprocess.run([nvcc, "--version"], capture_output=True, check=False)
        .stdout.decode()
        .split("\n")[0]
    )
    print("Compiler version: ", compiler_version)
    print("Compiler flags: ", " ".join(nvccflags))
    print("CuPy location: ", cupyloc)

    libname_double = (
        args["cuda_libname"] + f"_sm_{comp_capability}_double.cubin"
    )
    libname_single = (
        args["cuda_libname"] + f"_sm_{comp_capability}_single.cubin"
    )

    command = (
        [nvcc]
        + nvccflags
        + ["-o", libname_single, "-I" + cupyloc]
        + float_flags
        + cuda_files
    )

    print("\nCompiling the single-precision (32-bit) CUDA library")
    ret = run_compile(command, libname_single)
    if ret != 0:
        print("There was a compilation error.")
    else:
        print("Compiled successfully.")

    command = (
        [nvcc]
        + nvccflags
        + ["-o", libname_double, "-I" + cupyloc]
        + cuda_files
    )
    print("\nCompiling the double-precision (64-bit) CUDA library")
    ret = run_compile(command, libname_double)
    if ret != 0:
        print("There was a compilation error.")
    else:
        print("Compiled successfully.")


def run_compile(command, libname):
    if os.path.exists(libname):
        os.remove(libname)
    print(" ".join(command))
    ret = subprocess.run(command, check=False)
    if ret.returncode != 0 or not os.path.isfile(libname):
        return -1
    else:
        return 0


if __name__ == "__main__":
    main()
