# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Bootstrap of the Julia session used by the Julia backends."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

#: Name of the Julia package holding the BLonD kernels.
JULIA_PACKAGE_NAME = "BLonDKernels"

#: Julia dependency manifest read by `ensure_julia_environment`.
JULIAPKG_JSON_PATH = Path(__file__).resolve().parent / "juliapkg.json"

#: Environment variables that must be set before `juliacall` is imported.
#: `auto` lets Julia use all available cores for the KernelAbstractions
#: `CPU()` device; signal handling keeps Ctrl+C working in Julia code.
JULIACALL_ENVIRONMENT = {
    "PYTHON_JULIACALL_THREADS": "auto",
    "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
}

_MISSING_JULIACALL_MESSAGE = (
    "The Julia backends require `juliacall`, which is not installed. "
    "Install it with `pip install blond[julia]`. The first use "
    "additionally downloads Julia and the Julia packages, which needs "
    "network access."
)

#: Process-wide Julia session state. A dict keeps the lazy
#: initialisation free of `global` statements.
_julia_state: dict[str, Any] = {"kernels": None, "cuda_is_loaded": False}


def is_julia_available() -> bool:
    """
    Report whether `juliacall` can be imported.

    The probe is a pure `importlib` lookup: it neither imports
    `juliacall` nor triggers the (potentially large) Julia download that
    the first real import performs.

    Returns
    -------
    is_available
        True if `juliacall` is installed, False otherwise.
    """
    return importlib.util.find_spec("juliacall") is not None


def _read_julia_dependencies() -> dict[str, dict[str, Any]]:
    """
    Read the Julia dependencies of the `BLonDKernels` package.

    Returns
    -------
    packages
        Mapping of Julia package name to its `juliapkg` specification.

    Raises
    ------
    FileNotFoundError
        If the `juliapkg.json` manifest is missing.
    """
    if not JULIAPKG_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"The Julia dependency manifest {JULIAPKG_JSON_PATH} is "
            f"missing, so the Julia backends cannot be set up."
        )
    with JULIAPKG_JSON_PATH.open() as file:
        manifest = json.load(file)
    return manifest.get("packages", {})


def _register_julia_dependencies(juliapkg: Any) -> None:
    """
    Register every dependency of the manifest with `juliapkg`.

    Relative `path` entries (such as the development checkout of
    `BLonDKernels` shipped next to this file) are resolved against the
    directory of this module, so BLonD works from any working directory.

    Parameters
    ----------
    juliapkg
        The imported `juliapkg` module.
    """
    for name, specification in _read_julia_dependencies().items():
        keyword_arguments: dict[str, Any] = {}
        if "dev" in specification:
            keyword_arguments["dev"] = bool(specification["dev"])
        if "version" in specification:
            keyword_arguments["version"] = specification["version"]
        if "path" in specification:
            keyword_arguments["path"] = str(
                (JULIAPKG_JSON_PATH.parent / specification["path"]).resolve()
            )
        juliapkg.add(name, specification["uuid"], **keyword_arguments)


def _libstdcpp_version(path: str) -> tuple[int, ...]:
    """
    Return the version of a libstdc++ shared object from its file name.

    Parameters
    ----------
    path
        Path to a `libstdc++.so.*` file.

    Returns
    -------
    version
        The trailing version numbers, e.g. ``(6, 0, 33)``. Empty when the
        file name carries no version.
    """
    suffix = os.path.realpath(path).split("libstdc++.so.")[-1]
    numbers = suffix.split(".")
    if not all(number.isdigit() for number in numbers):
        return ()
    return tuple(int(number) for number in numbers)


def _loaded_libstdcpp() -> str | None:
    """
    Return the libstdc++ already mapped into this process, if any.

    NumPy and friends load the *system* libstdc++ long before Julia is
    started. Since the dynamic linker resolves by SONAME, Julia's own,
    newer copy can then no longer be loaded -- which aborts the whole
    process instead of raising.

    Returns
    -------
    path
        Path of the mapped libstdc++, or None if there is none (or if
        the mappings cannot be read, as on non-Linux platforms).
    """
    try:
        with open("/proc/self/maps") as mappings:
            for line in mappings:
                if "libstdc++.so" in line:
                    return line.split()[-1]
    except OSError:
        return None
    return None


def _check_libstdcpp_compatibility(julia_executable: str) -> None:
    """
    Fail early if Julia's libstdc++ is newer than the loaded one.

    Parameters
    ----------
    julia_executable
        Path of the Julia binary chosen by `juliapkg`.

    Raises
    ------
    OSError
        If the libstdc++ already mapped into this process is older than
        the one Julia needs. Loading Julia would abort the process, so
        this raises instead, with the workaround in the message.
    """
    bundled = (
        Path(julia_executable).resolve().parent.parent
        / "lib"
        / "julia"
        / "libstdc++.so.6"
    )
    loaded = _loaded_libstdcpp()
    if loaded is None or not bundled.is_file():
        return
    bundled_version = _libstdcpp_version(str(bundled))
    loaded_version = _libstdcpp_version(loaded)
    if not bundled_version or not loaded_version:
        return
    if loaded_version >= bundled_version:
        return
    raise OSError(
        f"The Julia backends cannot be started in this process: "
        f"`{loaded}` is already loaded, but Julia needs the newer "
        f"`{bundled}`, and the dynamic linker resolves both by the "
        f"same SONAME. Start Python with "
        f"`LD_PRELOAD={bundled}` -- or with any other libstdc++ that "
        f"provides at least the GLIBCXX version Julia was built "
        f"against, such as the one of a newer compiler toolchain "
        f"installed next to the system one -- to use "
        f"`julia_cpu`/`julia_gpu`."
    )


def ensure_julia_environment() -> Any:
    """
    Start the Julia session and load `BLonDKernels`, once per process.

    Returns
    -------
    kernels
        Handle of the loaded `BLonDKernels` Julia module.

    Raises
    ------
    ImportError
        If `juliacall` is not installed.
    OSError
        If a libstdc++ incompatible with Julia is already loaded.
    """
    if not is_julia_available():
        raise ImportError(_MISSING_JULIACALL_MESSAGE)

    if _julia_state["kernels"] is not None:
        return _julia_state["kernels"]

    for variable, value in JULIACALL_ENVIRONMENT.items():
        os.environ.setdefault(variable, value)

    import juliapkg  # type: ignore

    _register_julia_dependencies(juliapkg)
    juliapkg.resolve()
    _check_libstdcpp_compatibility(str(juliapkg.executable()))

    from juliacall import Main as jl  # type: ignore

    jl.seval(f"using {JULIA_PACKAGE_NAME}")
    _julia_state["kernels"] = getattr(jl, JULIA_PACKAGE_NAME)
    return _julia_state["kernels"]


def julia_kernels() -> Any:
    """
    Return the `BLonDKernels` module handle for the host backend.

    Returns
    -------
    kernels
        Handle of the loaded `BLonDKernels` Julia module.
    """
    return ensure_julia_environment()


def julia_cuda_kernels() -> Any:
    """
    Return the `BLonDKernels` handle with CUDA.jl loaded.

    Loading CUDA.jl activates the package extension that teaches
    `BLonDKernels` how to wrap CuPy device pointers.

    Returns
    -------
    kernels
        Handle of the loaded `BLonDKernels` Julia module.

    Raises
    ------
    RuntimeError
        If CUDA.jl is loaded but not functional on this machine.
    """
    kernels = ensure_julia_environment()
    if not _julia_state["cuda_is_loaded"]:
        from juliacall import Main as jl  # type: ignore

        jl.seval("using CUDA")
        if not bool(jl.seval("CUDA.functional()")):
            raise RuntimeError(
                "CUDA.jl is installed but not functional, so the "
                "`julia_gpu` backend cannot be used. Check the CUDA "
                "driver with `CUDA.versioninfo()` in Julia."
            )
        _julia_state["cuda_is_loaded"] = True
    return kernels
