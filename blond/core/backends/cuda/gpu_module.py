# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Loads the compiled CUDA kernels, compiling them first if missing."""

from __future__ import annotations

import os

import cupy as cp  # type: ignore
import numpy as np

from blond.core.backends.cuda.compiled_dir_handler import cuda_compiled_dir
from blond.core.backends.mpi_distributed.helpers import mpi_local_gpu_index
from blond.generals.compiled_cache import mark_used

_filepath = os.path.realpath(__file__)

# Under MPI, each rank of a node takes its own GPU. This must happen
# before the device queries below and before any CUDA work.
_device_index = mpi_local_gpu_index(cp.cuda.runtime.getDeviceCount())
if _device_index is not None:
    cp.cuda.Device(_device_index).use()
# `Device()` is the current device, which differs from GPU 0 under MPI.
_compute_capability = cp.cuda.Device().compute_capability

FLOAT = np.float64

folder = os.path.dirname(os.path.abspath(__file__))

# Same toolchain-aware directory the compiler writes to.
_basepath = str(cuda_compiled_dir(folder))


path = os.path.join(
    _basepath,
    f"kernels_sm_{_compute_capability}_double.cubin",
)
if not os.path.isfile(path):
    from blond.core.backends.cuda.compile import compile_cuda_library

    print("CUDA backend was not found.. Trying to compile.")
    compile_cuda_library()

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"The compiled CUDA backend was not found at {path=}.\n"
            f"Has the backend been compiled?"
            f"{os.path.join(folder, 'compile.py')}:1"  # :1 to
            # make PyCharm automatically link the correct file
        )
gpu_module = cp.RawModule(
    path=path,
)
# Refresh the LRU stamp on the cache dir we loaded from.
mark_used(_basepath)
