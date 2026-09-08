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
import json
import os
import shutil
import subprocess
import warnings
from typing import TYPE_CHECKING

from blond.generals.hashing_ import hash_build_target

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

# Default C++ compiler. The loader has no access to the compiler that was
# actually used, so both sides assume this default; override consistently if
# you compile with a different one.
DEFAULT_COMPILER = "g++"

_EXTENSIONS = (".py", ".h", ".cpp")

# Name of the file (in the fixed `compiled/` folder, not inside a
# hash-specific subdirectory -- it must be readable *before* the hash is
# known) recording the parameters of the most recent build.
_BUILD_OPTIONS_NAME = "build_options.json"


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
        The C++ compiler whose identity and target CPU instruction set (the
        instructions ``-march=native`` enables, e.g. AVX2) are folded into the
        hash. Must match between compile time and load time.
    optimize
        Whether the optimised (``-march=native``) build is requested. When
        true the host CPU's instruction set is folded in (so a native binary
        is never reused on a CPU lacking those instructions); when false it is
        not, so a portable build can be shared across CPUs.
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
        # Target CPU instruction set: `-march=native` enables different
        # instructions per host; this dump records exactly which ones are
        # enabled so the binary is never reused on a CPU lacking them.
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


def _build_options_path(folder: str) -> str:
    return os.path.join(folder, "compiled", _BUILD_OPTIONS_NAME)


def save_build_options(folder: str, **options: object) -> None:
    """
    Persist the build parameters used to compile this backend.

    Written to a fixed location, independent of any build-specific hash, so
    a later process can find it *before* it knows which hash to look for.
    :func:`load_build_options` reads it back and :func:`build_options_valid`
    checks whether it is still usable on the current machine. Best-effort: a
    write failure is warned about but never propagated, so it cannot break a
    compile.

    Parameters
    ----------
    folder
        The backend source folder (the directory of this module).
    **options : object
        The keyword arguments this build passed to :func:`cpp_compiled_dir`.
    """
    path = _build_options_path(folder)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(options, file, indent=2)
    except OSError as exc:
        warnings.warn(
            f"Could not save C++ build options to {path!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def load_build_options(folder: str) -> dict | None:
    """
    Load the build parameters saved by :func:`save_build_options`.

    Parameters
    ----------
    folder
        The backend source folder (the directory of this module).

    Returns
    -------
    options
        The saved keyword arguments, or ``None`` if no build options were
        saved, or the saved file could not be used.
    """
    path = _build_options_path(folder)

    options = None
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as file:
                options = json.load(file)
            if not isinstance(options, dict):
                raise ValueError(
                    f"expected a JSON object, got {type(options).__name__}"
                )
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Could not read saved C++ build options from {path!r}: "
                f"{exc}. Falling back to default build options.",
                RuntimeWarning,
                stacklevel=2,
            )
            options = None

    return options


def build_options_valid(options: dict, expected_keys: Sequence[str]) -> bool:
    """
    Check whether saved build options are still usable on this machine.

    Three stages, cheapest first:

    1. Every name in ``expected_keys`` must be present in ``options``.
       If an option is added, the saved ones are considered invalid.
    2. A filesystem/``PATH`` check -- the referenced compiler is
       resolvable, and any explicit FFTW/Boost paths still exist.
       n.b.: The currently applicable options are hard coded, they are
             "with_fftw_lib", "with_fftw_header", "boost".
    3. A syntax-only dry-run compile (``_check_dry_run_compile``) of
       the saved compiler with the saved ``optimize``/``flags`` options,
       which catches a flag the compiler no longer accepts (e.g. after a
       toolchain upgrade/downgrade) without a real build.

    This still cannot catch every way a build might now fail (e.g. a
    system header removed, or an FFTW/Boost include that only breaks
    when actually used) -- it only guards against the common,
    cheaply-detectable cases.

    Parameters
    ----------
    options
        A dict as returned by :func:`load_build_options`.
    expected_keys
        The build-parameter names ``options`` must contain. Missing keys
        make the saved options invalid, so this check stays correct even
        after ``compile_cpp_library`` gains or loses parameters.

    Returns
    -------
    valid
        ``False`` if any of ``expected_keys`` is absent from ``options``,
        the referenced compiler cannot be resolved, an explicit FFTW/Boost
        path no longer exists, or a syntax-only dry-run compile with the
        saved flags fails; ``True`` otherwise.
    """
    compiler = options.get("compiler", DEFAULT_COMPILER)

    # If another path-type parameter is added to `compile_cpp_library`,
    # it must be added here to be validated.
    path_flags = ("with_fftw_lib", "with_fftw_header", "boost")

    dry_run_flags = []
    if options.get("optimize", True):
        dry_run_flags += ["-march=native", "-ffast-math"]
    dry_run_flags += (options.get("flags") or "").split()

    return (
        _check_build_keys(options, expected_keys)
        and _check_compiler(compiler)
        and _check_lib_dirs(options, path_flags)
        and _check_dry_run_compile(compiler, dry_run_flags)
    )


def _check_build_keys(options: dict, expected_keys: Sequence[str]) -> bool:
    """
    Check that every name in ``expected_keys`` is present in ``options``.

    Parameters
    ----------
    options
        A dict as returned by :func:`load_build_options`.
    expected_keys
        The build-parameter names ``options`` must contain.

    Returns
    -------
    ok
        ``True`` if every name in ``expected_keys`` is present in
        ``options``, ``False`` otherwise.
    """
    return not any(key not in options for key in expected_keys)


def _check_compiler(compiler: str) -> bool:
    """
    Check that ``compiler`` is resolvable via ``PATH`` or a real file.

    Parameters
    ----------
    compiler
        A bare command name or a path to an executable.

    Returns
    -------
    ok
        ``True`` if ``compiler`` resolves to something, ``False``
        otherwise.
    """
    return shutil.which(compiler) is not None or os.path.isfile(compiler)


def _check_lib_dirs(options: dict, libs: Sequence[str]) -> bool:
    """
    Check that every explicit path named in ``libs`` still exists.

    Parameters
    ----------
    options
        A dict as returned by :func:`load_build_options`.
    libs
        The names of the ``options`` entries whose value, if set, must
        be an existing filesystem path.

    Returns
    -------
    ok
        ``True`` if every non-empty path named in ``libs`` exists on
        disk, ``False`` otherwise.
    """
    for lib in libs:
        path = options.get(lib)
        if path and not os.path.exists(path):
            return False
    return True


def _check_dry_run_compile(compiler: str, flags: list[str]) -> bool:
    """
    Try a syntax-only compile of a trivial source with the given flags.

    Uses ``-fsyntax-only`` so the compiler parses and validates ``flags``
    without generating code or linking -- much cheaper than a real build,
    but still an actual invocation of ``compiler`` (unlike the filesystem
    checks in :func:`build_options_valid`). Catches a flag that the
    compiler no longer accepts, e.g. after a toolchain upgrade/downgrade.

    Parameters
    ----------
    compiler
        The C++ compiler to invoke.
    flags
        Compiler flags to validate.

    Returns
    -------
    ok
        ``True`` if the compiler accepted the flags, ``False`` if it
        rejected them, could not run, or timed out.
    """
    command = [compiler, "-xc++", "-fsyntax-only", *flags, "-"]
    try:
        result = subprocess.run(
            command,
            input=b"int main() { return 0; }\n",
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception:
        return False
    return result.returncode == 0
