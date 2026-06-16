# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions to compile CUDA backend for `CudaSpecials`."""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import TYPE_CHECKING

from blond.core.backends.cuda.compiled_dir_handler import (
    cuda_compiled_dir,
    resolve_nvcc,
)
from blond.generals.compiled_cache import mark_used, prune

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

_filepath = os.path.realpath(__file__)
_basepath = os.sep.join(_filepath.split(os.sep)[:-1])


def run_compile(command: list[str], libname: str) -> int:
    """
    Execute the compile command for the library.

    Parameters
    ----------
    command
        Any bash command.
    libname
        Library that has to be created.

    Returns
    -------
    return_code
        0 if OK, else -1.
    """
    if os.path.exists(libname):
        os.remove(libname)
    print(" ".join(command))
    ret = subprocess.run(command, check=False)
    if ret.returncode != 0 or not os.path.isfile(libname):
        return -1
    else:
        return 0


def compile_cuda_library(  # NOQA: PLR0915
    compute_capability: int | Literal["discover"] = "discover",
) -> None:
    """
    Compile the GPU library.

    Parameters
    ----------
    compute_capability
        The compute capability of your GPU,
        see https://developer.nvidia.com/cuda-gpus.
    """
    print("\nTrying to compile CUDA backend.")

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

    folder = os.path.dirname(os.path.abspath(__file__))

    nvcc = resolve_nvcc()

    # Toolchain-aware directory, computed identically by the loader
    # (callables.py) so it finds exactly what we build here. Call with the
    # default nvcc (same as the loader) to share the memoised entry.
    target = cuda_compiled_dir(folder)
    os.makedirs(target, exist_ok=True)

    # The CUDA library name, without the file extension.
    cuda_libname = os.path.join(target, "kernels")

    import cupy as cp  # type: ignore # NOQA must be installed to be compiled / force exception

    # if something is wrong with the installation

    print("\nCompiling the CUDA library")
    if compute_capability == "discover":
        print("Discovering the device compute capability..")

        dev = cp.cuda.Device(0)
        dev_name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
        compute_capability_ = dev.compute_capability
        print(f"Device name {dev_name}")
    elif compute_capability is not None:
        compute_capability_ = compute_capability
    else:
        raise ValueError(f"{compute_capability=}")
    print(
        f"Compiling the CUDA library for"
        f" compute capability {compute_capability_}."
    )

    # Add the -arch required argument
    nvcc_flags += ["-arch", f"sm_{compute_capability_}"]

    # Get the CuPy header files location
    path_ = cp.__file__.split("/")[:-1]  # remove __init__.py from path
    path_.extend(["core", "include"])
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

    libname_double = cuda_libname + f"_sm_{compute_capability_}_double.cubin"
    # Reuse a previously built cubin when present: it lives in a
    # toolchain-keyed `compiled/<hash>/` dir and its filename encodes the
    # exact target GPU arch, so an existing file is guaranteed compatible.
    if os.path.isfile(libname_double):
        print(f"Reusing cached CUDA library: {libname_double}")
        mark_used(target)
        prune(os.path.dirname(target))
        return
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
        # Stamp this build and evict least-recently-used dirs so the
        # compiled/ tree (and the CI cache) stays bounded.
        mark_used(target)
        prune(os.path.dirname(target))


def main_cli() -> None:
    """Parse arguments from command line."""
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
