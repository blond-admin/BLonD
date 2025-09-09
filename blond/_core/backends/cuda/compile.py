from __future__ import annotations

import argparse
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Literal

_filepath = os.path.realpath(__file__)
_basepath = os.sep.join(_filepath.split(os.sep)[:-1])


def run_compile(command: List[str], libname: str) -> int:
    if os.path.exists(libname):
        os.remove(libname)
    print(" ".join(command))
    ret = subprocess.run(command, check=False)
    if ret.returncode != 0 or not os.path.isfile(libname):
        return -1
    else:
        return 0


def compile_cuda_library(
    compute_capability: int | Literal["discover"] = "discover",
) -> None:
    """
    Compile the GPU library

    Parameters
    ----------
    compute_capability
        The compute capability of your GPU,
        see https://developer.nvidia.com/cuda-gpus.

    """
    print(f"\nTrying to compile CUDA backend.")

    cuda_files = [
        os.path.join(_basepath, "kernels.cu"),
    ]
    nvcc_flags = [
        "--cubin",
        "-O3",
        "--use_fast_math",
        "-maxrregcount",
        "32",
    ]

    # The CUDA library name, without the file extension.
    cuda_libname = os.path.join(_basepath, "kernels")

    nvcc = "nvcc"

    # Get nvcc from CUDA_PATH
    cuda_path = os.getenv("CUDA_PATH", default="")
    if cuda_path != "":
        nvcc = cuda_path + "/bin/nvcc"
    import cupy as cp  # type: ignore # NOQA must be installed to be compiled / force exception

    # if something is wrong with the installation

    print("\nCompiling the CUDA library")
    if compute_capability == "discover":
        print("Discovering the device compute capability..")

        dev = cp.cuda.Device(0)
        dev_name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
        compute_capability = dev.compute_capability
        print(f"Device name {dev_name}")
    elif compute_capability is not None:
        compute_capability = compute_capability
    else:
        raise ValueError(f"{compute_capability=}")
    print(
        f"Compiling the CUDA library for"
        f" compute capability {compute_capability}."
    )

    # Add the -arch required argument
    nvcc_flags += ["-arch", f"sm_{compute_capability}"]

    # Get the CuPy header files location
    path_ = cp.__file__.split("/")[:-1]  # remove __init__.py from path
    path_.extend(["_core", "include"])
    cupyloc = os.path.join("/".join(path_))

    print("CUDA Compiler: ", nvcc)
    compiler_version = (
        subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            check=False,
        )
        .stdout.decode()
        .split("\n")[0]
    )
    print("Compiler version: ", compiler_version)
    print("Compiler flags: ", " ".join(nvcc_flags))
    print("CuPy location: ", cupyloc)

    libname_double = cuda_libname + f"_sm_{compute_capability}_double.cubin"
    libname_single = cuda_libname + f"_sm_{compute_capability}_single.cubin"

    command = (
        [nvcc]
        + nvcc_flags
        + ["-o", libname_single, "-I" + cupyloc]
        + ["-DUSEFLOAT"]
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
        + nvcc_flags
        + ["-o", libname_double, "-I" + cupyloc]
        + cuda_files
    )
    print("\nCompiling the double-precision (64-bit) CUDA library")
    ret = run_compile(command, libname_double)
    if ret != 0:
        print("There was a compilation error.")
    else:
        print("Compiled successfully.")


def main_cli() -> None:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(
        description="Script used to compile the CUDA libraries needed by BLonD.",
    )
    parser.add_argument(
        "-sm",
        "--sm",
        nargs="+",
        default=["discover"],
        help="CUDA Streaming Multiprocessor (SM) compute capabilitie(s),"
        " e.g. -sm 70 80"
        " (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported).",
    )
    args = vars(parser.parse_args())

    for sm in args["sm"]:  # iterate all SM compute capabilities given by user
        compile_cuda_library(compute_capability=sm)


if __name__ == "__main__":  # pragma: no cover
    main_cli()
