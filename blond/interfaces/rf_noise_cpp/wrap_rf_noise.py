# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Python ctypes binding for CERN's ``VariNoise`` RF-noise generator.

The actual noise algorithm lives in the external CERN C++ library
``rf-noise-cpp`` (see https://gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp),
which is *not* shipped with BLonD. This module only owns the C interop:
locating (and, if necessary, safely compiling) the shared library and exposing
the single :func:`rf_noise` entry point.

Loading and compilation are deferred to the first :func:`rf_noise` call so that
``import blond`` never fails when the external library or source is absent.

Author: Simon Lauber
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

_local_path = pathlib.Path(__file__).parent.resolve()
# The C++ wrapper source and the compiled library live alongside this module.
_src_path = _local_path

#: URL of the external source required to build the noise library.
RF_NOISE_REPO_URL = (
    "https://gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp"
)


def _install_hint() -> str:
    """
    Actionable instructions for obtaining the external source.

    Returns
    -------
    hint
        Multi-line installation instructions.
    """
    return (
        "To install the external RF noise source, either clone it next to the"
        " BLonD repository:\n"
        f"    git clone {RF_NOISE_REPO_URL}\n"
        "or set the 'RF_NOISE_DIR' environment variable to point at an existing"
        " checkout, e.g.:\n"
        "    export RF_NOISE_DIR=/path/to/rf-noise-cpp"
    )


def _compiled_lib_name() -> str:
    """
    Build the platform-specific shared-library file name.

    Returns
    -------
    file_name
        File name (without directory) of the compiled library.
    """
    os_name = platform.system()  # e.g. 'Linux', 'Windows', 'Darwin'
    processor = platform.processor()  # e.g. 'x86_64', 'Intel64'
    # `platform.processor()` can return a verbose string with spaces/commas
    # (notably on Windows), which is unsuitable for a file name. Reduce it to a
    # filesystem-safe token.
    tag = re.sub(
        r"[^A-Za-z0-9]+", "_", f"{os_name.lower()}_{processor}"
    ).strip("_")
    extension = ".dll" if "win" in sys.platform else ".so"
    return f"rf_noise_wrapper_{tag}{extension}"


def _target_library_path() -> pathlib.Path:
    """
    Resolve where the compiled library is expected/stored.

    Returns
    -------
    path
        Absolute path of the compiled library.
    """
    return _src_path / _compiled_lib_name()


def _get_rf_noise_dir() -> pathlib.Path:
    """
    Resolve the location of the external ``rf-noise-cpp`` source tree.

    Returns
    -------
    rf_noise_dir
        Path to the external source. Taken from the ``RF_NOISE_DIR``
        environment variable if set, otherwise assumed to be a sibling of the
        BLonD repository.
    """
    if "RF_NOISE_DIR" in os.environ:
        return pathlib.Path(os.environ["RF_NOISE_DIR"]).resolve()
    # Assume it sits next to the BLonD repository.
    return (_local_path / "../../../../rf-noise-cpp/").resolve()


def _compile_rf_noise_library(
    rf_noise_dir: pathlib.Path, target_library: pathlib.Path
) -> None:
    """
    Compile the RF-noise shared library from the external source.

    The compilation is performed without a shell (``subprocess.run`` with an
    argument list), so user-controlled paths cannot be interpreted as shell
    commands.

    Parameters
    ----------
    rf_noise_dir
        Path of the external ``rf-noise-cpp`` source tree.
    target_library
        Path the compiled library is written to.

    Raises
    ------
    FileNotFoundError
        If the source tree or the C++ compiler cannot be found.
    NameError
        If ``rf_noise_dir`` does not point at an ``rf-noise-cpp`` tree.
    RuntimeError
        If the compilation exits with a non-zero return code.
    """
    if not str(rf_noise_dir).endswith("rf-noise-cpp"):
        raise NameError(f"Path must end on 'rf-noise-cpp', not {rf_noise_dir}")
    if not rf_noise_dir.is_dir():
        raise FileNotFoundError(
            f"Couldn't find the RF Noise repository at {rf_noise_dir}.\n"
            f"{_install_hint()}"
        )

    rf_noise_src = rf_noise_dir / "src" / "rf-noise"
    if not rf_noise_src.is_dir():
        raise FileNotFoundError(
            f"Expected C++ sources at {rf_noise_src}, but the directory does"
            f" not exist. Is {rf_noise_dir} a complete checkout of"
            f" {RF_NOISE_REPO_URL}?\n"
            f"{_install_hint()}"
        )
    cpp_files = [str(p) for p in rf_noise_src.glob("*.cpp")]
    if not cpp_files:
        raise FileNotFoundError(
            f"No C++ source files found in {rf_noise_src}.\n{_install_hint()}"
        )

    # Remove a stale/partial library before rebuilding.
    if target_library.is_file():
        target_library.unlink()

    # Argument list (no shell) closes the legacy OS-command-injection hole.
    # Flags mirror the conventions in blond.core.backends.cpp.compile:
    # the rf-noise sources are pure C++ (no boost), so the legacy
    # ``-lboost_system`` link is dropped; ``-D_USE_MATH_DEFINES`` makes M_PI
    # available under mingw; ``-fPIC`` is only meaningful on posix.
    command = [
        "g++",
        "-O3",
        "-std=c++11",
        "-shared",
        "-D_USE_MATH_DEFINES",
    ]
    if os.name == "posix":
        command.append("-fPIC")
    command += [
        "-o",
        str(target_library),
        str(_src_path / "rf_noise_wrapper.cpp"),
        *cpp_files,
        f"-I{rf_noise_src}",
    ]

    try:
        process = subprocess.run(  # noqa: S603 (shell=False, argv list)
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # g++ not installed
        raise FileNotFoundError(
            "Could not find the 'g++' compiler required to build the RF noise"
            " library. Install a C++ compiler or provide a prebuilt library at"
            f" {target_library}."
        ) from exc

    if process.returncode != 0:
        # Clean up any partial output.
        if target_library.is_file():
            with contextlib.suppress(OSError):  # pragma: no cover
                target_library.unlink()
        raise RuntimeError(
            f"Compilation of the RF noise library failed"
            f" (return code {process.returncode}).\n"
            f"command={command}\n"
            f"stderr:\n{process.stderr}"
        )


@functools.cache
def _load_rf_noise() -> ctypes.CDLL:
    """
    Lazily locate, build if necessary, and load the RF-noise library.

    The result is cached, so the library is located/built and loaded only once.

    Returns
    -------
    library
        The loaded shared library with ``argtypes`` configured.
    """
    target_library = _target_library_path()
    if not target_library.is_file():
        _compile_rf_noise_library(_get_rf_noise_dir(), target_library)

    # On Windows the loader must find the mingw runtime DLLs next to the lib.
    if "win" in sys.platform and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(target_library.parent))
        library = ctypes.CDLL(str(target_library), winmode=0)
    else:
        library = ctypes.CDLL(str(target_library))
    library.rf_noise_wrapper.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    return library


def rf_noise_library_available() -> bool:
    """
    Report whether the RF-noise shared library can be loaded.

    Attempts to locate (and, if necessary, build) and load the library without
    raising. Intended for tests/examples to skip gracefully when the external
    ``rf-noise-cpp`` library is unavailable.

    Returns
    -------
    available
        ``True`` if the library is loadable, ``False`` otherwise.
    """
    try:
        _load_rf_noise()
    except (FileNotFoundError, NameError, RuntimeError, OSError):
        return False
    return True


def rf_noise(
    frequency_high: NumpyArray,
    frequency_low: NumpyArray,
    gain_x: NumpyArray,
    gain_y: NumpyArray,
    n_source: int,
    n_pnt_min: int,
    r_seed: int,
    sampling_rate: float,
    rms: float,
    phase_array: NumpyArray | None = None,
) -> NumpyArray:
    """
    Generate RF noise along time (overwriting ``phase_array``).

    Parameters
    ----------
    frequency_high
        Array of the frequency upper limit along time, in [Hz].
    frequency_low
        Array of the frequency lower limit along time, in [Hz].
    gain_x
        Array from 0 (``frequency_low``) to 1 (``frequency_high``).
    gain_y
        Frequency density distribution between the high and low limits.
        Stays the same along time.
    n_source
        Minimum number of elementary harmonic noise sources. To allow a
        reasonable FFT with small prime factors the finally used number might
        be slightly higher. The noise resolution is the bandwidth
        ``(f_high - f_low)`` divided by ``n_source``.
    n_pnt_min
        Minimum number of steps to express the highest-frequency oscillation,
        automatically set to 6 if lower. The finally used value might be
        slightly higher so that ``n_source * n_pnt`` factorises into small
        primes (the FFT length).
    r_seed
        If ``< 0`` use a clock seed (every call differs); if ``>= 0`` use the
        given starting seed to reproduce the same noise.
    sampling_rate
        Play-back clock frequency, in [Hz].
    rms
        RMS value of the total time-domain output stream. Independent of the
        limit frequencies (wider bands get lower amplitudes).
    phase_array
        Optional pre-allocated output array. If given, the result is written
        into it to avoid an allocation.

    Returns
    -------
    phase_array
        The RF noise along time, in [rad].
    """
    if phase_array is None:
        phase_array = np.empty(len(frequency_high), dtype=np.double)

    # Coerce dtypes, warning on mismatch (mirrors legacy behaviour).
    if frequency_high.dtype != np.double:
        warn(f"{frequency_high.dtype=}, but should be np.double", stacklevel=2)
    frequency_high = frequency_high.astype(np.double)
    if frequency_low.dtype != np.double:
        warn(f"{frequency_low.dtype=}, but should be np.double", stacklevel=2)
    frequency_low = frequency_low.astype(np.double)
    if gain_x.dtype != np.double:
        warn(f"{gain_x.dtype=}, but should be np.double", stacklevel=2)
    gain_x = gain_x.astype(np.double)
    if gain_y.dtype != np.double:
        warn(f"{gain_y.dtype=}, but should be np.double", stacklevel=2)
    gain_y = gain_y.astype(np.double)
    if phase_array.dtype != np.double:
        warn(f"{phase_array.dtype=}, but should be np.double", stacklevel=2)
    phase_array = phase_array.astype(np.double)

    # Validate shapes and ranges.
    assert len(frequency_high) == len(phase_array), (
        f"{len(frequency_high)=}, {len(phase_array)=}"
    )
    assert len(frequency_high) == len(frequency_low), (
        f"{len(frequency_high)=}, {len(frequency_low)=}"
    )
    assert len(gain_x) == len(gain_y), f"{len(gain_x)=}, {len(gain_y)=}"
    assert np.all(frequency_low < frequency_high), (
        "All 'frequency_low' must be smaller than 'frequency_high'"
    )
    assert np.min(gain_x) >= 0.0, (
        f"'gain_x' must be within 0.0 and 1.0, but got {np.min(gain_x)=}"
    )
    assert np.max(gain_x) <= 1.0, (
        f"'gain_x' must be within 0.0 and 1.0, but got {np.max(gain_x)=}"
    )

    library = _load_rf_noise()

    # Prepare pointers to the numpy arrays.
    f_high_ptr = frequency_high.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f_low_ptr = frequency_low.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xs_ptr = gain_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ys_ptr = gain_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result_ptr = phase_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    library.rf_noise_wrapper(
        f_high_ptr,
        ctypes.c_size_t(frequency_high.size),
        f_low_ptr,
        ctypes.c_size_t(frequency_low.size),
        xs_ptr,
        ctypes.c_size_t(gain_x.size),
        ys_ptr,
        ctypes.c_size_t(gain_y.size),
        result_ptr,
        ctypes.c_size_t(phase_array.size),
        ctypes.c_int(n_source),
        ctypes.c_int(n_pnt_min),
        ctypes.c_int(r_seed),
        ctypes.c_double(sampling_rate),
        ctypes.c_double(rms),
    )
    return phase_array
