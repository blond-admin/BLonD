# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Shared cache-key for the compiled C++ backend.

The directory holding a compiled library must be located identically by the
compiler (``compile.py``) and by the loader (``callables.py``); otherwise the
loader cannot find what the compiler produced. Both therefore call
:func:`cpp_compiled_dir`, which is the single source of truth for that path.

The key folds the build environment into the hash so a binary built for one
toolchain/CPU/flag set is never reused on an incompatible one (a
``-march=native`` binary run on an older CPU crashes with SIGILL). Two kinds
of input are captured:

* **Source-defined flags** need not be listed here: ``compile.py`` and this
  module both live in the hashed folder, so any change to the flag logic
  already changes the digest.
* **Caller-supplied build parameters** (compiler, ``optimize``, ``flags``,
  ``libs``, FFTW/Boost options) *are* folded in, with defaults that mirror
  :func:`blond.core.backends.cpp.compile.compile_cpp_library`. The loader has
  no access to these, so it relies on those defaults; a *default* build (what
  CI produces) therefore rendezvouses, while a build with custom parameters
  correctly lands in -- and is loaded from -- a distinct directory only when
  the loader is given the matching parameters.
"""

from __future__ import annotations

import functools
import os

from blond.generals.hashing_ import hash_build_target

# Default C++ compiler. The loader has no access to the compiler that was
# actually used, so both sides assume this default; override consistently if
# you compile with a different one.
DEFAULT_COMPILER = "g++"

_EXTENSIONS = (".py", ".h", ".cpp")


@functools.cache
def cpp_compiled_dir(
    folder: str,
    *,
    compiler: str = DEFAULT_COMPILER,
    optimize: bool = True,
    flags: str = "",
    libs: str = "",
    with_fftw: bool = False,
    with_fftw_threads: bool = False,
    with_fftw_omp: bool = False,
    with_fftw_lib: str | None = None,
    with_fftw_header: str | None = None,
    boost: str | None = None,
) -> str:
    """
    Return the ``compiled/<hash>`` directory for the current sources + build.

    The keyword defaults must mirror
    :func:`blond.core.backends.cpp.compile.compile_cpp_library`; the loader
    calls this with the defaults, so a default build is found there. The
    result is memoised per process: the sources and toolchain do not change
    within a run, so the (subprocess-spawning) toolchain probes run only once.

    Parameters
    ----------
    folder
        The backend source folder (the directory of this module).
    compiler
        The C++ compiler whose identity and target ISA are folded into the
        hash. Must match between compile time and load time.
    optimize
        Whether the optimised (``-march=native``) build is requested. When
        true the host ISA is folded in (so a native binary is never reused on
        a CPU lacking those instructions); when false it is not, so a portable
        build can be shared across CPUs.
    flags, libs
        Caller-supplied extra compiler flags / link libraries.
    with_fftw, with_fftw_threads, with_fftw_omp, with_fftw_lib, with_fftw_header
        FFTW linkage options.
    boost
        Boost path (or ``None``).

    Returns
    -------
    compiled_dir
        Absolute path to the build-specific ``compiled/<hash>`` directory.
    """
    probe_commands = [
        # Toolchain identity: a different compiler version produces an
        # ABI-incompatible binary.
        [compiler, "--version"],
    ]
    if optimize:
        # Target ISA: `-march=native` resolves to different instruction sets
        # per host; this dump records exactly which ones are enabled so the
        # binary is never reused on a CPU lacking them.
        probe_commands.append(
            [compiler, "-march=native", "-dM", "-E", "-xc", "-"]
        )
    hash_ = hash_build_target(
        folder=folder,
        extensions=_EXTENSIONS,
        recursive=False,
        probe_commands=probe_commands,
        extra=(
            f"optimize={optimize}",
            f"flags={flags}",
            f"libs={libs}",
            f"fftw={with_fftw}/{with_fftw_threads}/{with_fftw_omp}"
            f"/{with_fftw_lib}/{with_fftw_header}",
            f"boost={boost}",
        ),
    )
    return os.path.join(folder, "compiled", hash_)
