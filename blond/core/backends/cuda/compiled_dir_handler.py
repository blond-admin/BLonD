# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Shared cache-key for the compiled CUDA backend.

The directory holding a compiled cubin must be located identically by the
compiler (:mod:`blond.core.backends.cuda.compile`) and by the loader
(:mod:`blond.core.backends.cuda.callables`); both therefore call
:func:`cuda_compiled_dir`, the single source of truth for that path.

The key folds the CUDA toolchain version into the hash so a cubin built with
one ``nvcc`` is never reused with an incompatible one. The target GPU
architecture is additionally encoded in the cubin *filename* (``_sm_<cc>_``),
so different GPUs coexist in the same directory; source-defined ``nvcc`` flags
are already covered by hashing the ``.py``/``.cu`` sources.
"""

from __future__ import annotations

import functools
import os

from blond.generals.hashing_ import hash_build_target

_EXTENSIONS = (".py", ".cu")


def resolve_nvcc() -> str:
    """
    Return the ``nvcc`` executable, honouring ``CUDA_PATH`` if set.

    Returns
    -------
    nvcc
        Path to (or bare name of) the CUDA compiler driver.
    """
    cuda_path = os.getenv("CUDA_PATH", default="")
    if cuda_path != "":
        return cuda_path + "/bin/nvcc"
    return "nvcc"


@functools.cache
def cuda_compiled_dir(folder: str, nvcc: str | None = None) -> str:
    """
    Return the ``compiled/<hash>`` directory for the current sources + toolchain.

    Memoised per process: the sources and toolchain do not change within a
    run, so the (subprocess-spawning) ``nvcc --version`` probe runs only once.
    Call with the default ``nvcc`` on both the compile and load side so they
    share the cache entry.

    Parameters
    ----------
    folder
        The backend source folder (the directory of this module).
    nvcc
        The CUDA compiler whose version is folded into the hash. Defaults to
        :func:`resolve_nvcc`. Must match between compile time and load time.

    Returns
    -------
    compiled_dir
        Absolute path to the toolchain-specific ``compiled/<hash>`` directory.
    """
    if nvcc is None:
        nvcc = resolve_nvcc()
    hash_ = hash_build_target(
        folder=folder,
        extensions=_EXTENSIONS,
        recursive=False,
        probe_commands=[[nvcc, "--version"]],
    )
    return os.path.join(folder, "compiled", hash_)
