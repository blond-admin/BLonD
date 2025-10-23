"""
Python ctypes wrapper for CERN RF Noise, provided via Python function rf_noise(...)

See https://gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp

Author: Simon Lauber
"""

from __future__ import annotations

import os
import pathlib
import platform
import subprocess
from os.path import isfile
from warnings import warn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

_local_path = pathlib.Path(__file__).parent.resolve()


def _generate_compiled_file_name() -> str:
    # Get system information
    os_name = (
        platform.system()
    )  # e.g., 'Linux', 'Windows', 'Darwin' (for macOS)
    processor = platform.processor()  # e.g., 'x86_64', 'Intel64'

    # Format the name
    file_name = f"rf_noise_wrapper_{os_name.lower()}_{processor}"

    return file_name


_target_library = str(_local_path / f"{_generate_compiled_file_name()}.so")
# check if lib exists already
if isfile(_target_library):
    _make_library = False
else:
    _make_library = True

if "RF_NOISE_DIR" in os.environ.keys():
    # give user possibility to change the location of the RF Noise source
    _rf_noise_dir = pathlib.Path(os.environ["RF_NOISE_DIR"])
else:
    # assume that its in a neighboring folder of BLonD
    _rf_noise_dir = pathlib.Path("./../../../../rf-noise-cpp/").resolve()


def _compile_rf_noise_library(rf_noise_dir: pathlib.Path):
    """Make the library for python, so that it can be imported later


    Notes
    -----
    Requires the source code from https://gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp


    Parameters
    ----------
    rf_noise_dir: pathlib.Path
        Path of the RF Noise source code
    """

    if not str(rf_noise_dir).endswith("rf-noise-cpp"):
        raise NameError(f"Path must end on 'rf-noise-cpp', not {rf_noise_dir}")
    if not os.path.isdir(rf_noise_dir):
        raise FileNotFoundError(f"""
        Couldn't find the RF Noise repository at {rf_noise_dir} ! 
        Make sure it was downloaded to your computer from
        https://gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp 

        Optional: You can also change the path using the environment variable 'RF_NOISE_DIR'
        """)

    if isfile(_target_library):
        # remove old library
        os.remove(_target_library)

    # get all C++ files in rf-noise directory
    rf_noise_src = rf_noise_dir / "src/rf-noise/"
    cpp_files = tuple(
        filter(lambda s: s.endswith(".cpp"), os.listdir(rf_noise_src))
    )
    rf_noise_cpp_files = " ".join([str(rf_noise_src / s) for s in cpp_files])
    make_command = (
        f"g++ -m64 -fPIC -shared -o {_target_library} {_local_path / 'rf_noise_wrapper.cpp'} "
        f"{rf_noise_cpp_files} "
        f"-I{rf_noise_src} "
        f"-lboost_system"
    )
    process = subprocess.Popen(
        make_command.strip().split(" "),
    )
    process.communicate()
    if process.returncode != 0:
        try:
            os.remove(_target_library)
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Compilation terminated with {process.returncode=}\n for {make_command=}"
        )


if _make_library:
    _compile_rf_noise_library(_rf_noise_dir)

import ctypes
import numpy as np

# Load the shared library
_library_rf_noise = ctypes.CDLL(str(_target_library))

# Define the function's argument and return types
_library_rf_noise.rf_noise_wrapper.argtypes = [
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
    phase_array: NumpyArray = None,
) -> NumpyArray:
    """Generates RF noise along time (overwriting phase_array)

    Parameters
    ----------
    frequency_high: NumpyArray
        Array of frequency upper limit along time
    frequency_low: NumpyArray
        Array of frequency lower limit along time
    gain_x: NumpyArray
        array from 0 (fLow) to 1 (fHigh)
    gain_y: NumpyArray
        Frequency density distribution in between high and low.
        Stays the same along time
    n_source: int
        Minimum number of elementary harmonic noise sources. To allow reasonable FFT with small prime-factors
        the finally used nSource mit be slightly higher. The final noise resolution is the bandwidth
        (fUp-fLow) divided by nSource, e.g. for (fUp-fLow)=100 Hz and nSource=500, elementary oscillators
        are 0.2 Hz apart, so for any application with less resolution the noise is good, else the elementary
        oscillators ca be resolved and the result depends on the precise frequency observed.
    n_pnt_min: int
        Minimum number of steps to express the highest-frequency oscillation, automatically set to 6 if lower.
                To allow reasonable FFT with small prime-factors, the finally used nPnt might be a slightly higher number
                (nData = nSource*nPnt should be able to be split in low prime factors; FFT is done for nData long arrays!!)
    r_seed: int
        If < 0, use clock seed, i.e. each call even with else identical parameters is different
        If � 0, use the given number as starting seed, reproducing same noise for same starting seed
    sampling_rate: float
        Play-back clock frequency
    rms: float
        Rms value of total time-domain output stream, does not change when limit frequencies are changed
        i.e. amplitudes for wider bands are lower
    phase_array: NumpyArray, Optional
        The calculation-result will be written to the phase_array array if given.
        By this, array creation routines (np.empty(...)) can be prevented.


    Returns
    -------
    phase_array: NumpyArray
        The RF noise along time

    Examples
    --------
    >>> ys = np.loadtxt("lhc_spectrum.txt")
    >>> xs = np.linspace(0.0, 1.0, len(ys))
    >>>
    >>> N = 20000000
    >>> f_low = np.linspace(10, 100, N)
    >>> f_high = np.linspace(20, 200, N)
    >>> phase_array = np.empty(N)
    >>>
    >>> phase_array = rf_noise(
    >>>     frequency_high=f_high,
    >>>     frequency_low=f_low,
    >>>     gain_x=xs,
    >>>     gain_y=ys,
    >>>     n_source=2048,
    >>>     n_pnt_min=8,
    >>>     r_seed=0,
    >>>     sampling_rate=11245.49,
    >>>     rms=1.0,
    >>>     phase_array=phase_array,
    >>> )
    >>> from matplotlib import pyplot as plt
    >>>
    >>> plt.title("rf_noise_wrapper Python ctypes interface")
    >>> plt.specgram(phase_array, Fs=11245.49, NFFT=int(20000000 / 1000), label="specgram(phase_array)")
    >>> xxx = np.linspace(0, 20000000 / 11245.49, 20000000)
    >>> plt.plot(xxx, f_low, label="fLo", color="red")
    >>> plt.plot(xxx, f_high, label="fHi", color="red")
    >>> plt.ylim(0, 2 * f_high[-1])
    >>> plt.legend(loc="upper left")
    >>>
    >>> plt.show()
    """
    if phase_array is None:
        phase_array = np.empty(len(frequency_high), dtype=np.double)

    # check dtypes
    if frequency_high.dtype != np.double:
        warn(f"{frequency_high.dtype=}, but should be np.double")
    frequency_high = frequency_high.astype(np.double)
    if frequency_low.dtype != np.double:
        warn(f"{frequency_low.dtype=}, but should be np.double")
    frequency_low = frequency_low.astype(np.double)
    if gain_x.dtype != np.double:
        warn(f"{gain_x.dtype =}, but should be np.double")
    gain_x = gain_x.astype(np.double)
    if gain_y.dtype != np.double:
        warn(f"{gain_y.dtype =}, but should be np.double")
    gain_y = gain_y.astype(np.double)
    if phase_array.dtype != np.double:
        warn(f"{phase_array.dtype=}, but should be np.double")
    phase_array = phase_array.astype(np.double)

    # make sure everything is as expected
    # compare lengths
    assert len(frequency_high) == len(phase_array), (
        f"{len(frequency_high)=}, {len(phase_array)=}"
    )
    assert len(frequency_high) == len(frequency_low), (
        f"{len(frequency_high)=}, {len(frequency_low)=}"
    )
    assert len(gain_x) == len(gain_y), f"{len(gain_x)=}, {len(gain_y)=}"
    # check ranges
    assert np.all(frequency_low < frequency_high), (
        "All 'fLow' must be smaller 'fHigh'"
    )
    assert np.min(gain_x) >= 0.0, (
        f"'xs' must be within 0.0 and 1.0, but got {np.min(gain_x)=}"
    )
    assert np.max(gain_x) <= 1.0, (
        f"'xs' must be within 0.0 and 1.0, but got {np.max(gain_x)=}"
    )

    # Prepare pointers to the numpy arrays
    f_high_ptr = frequency_high.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f_low_ptr = frequency_low.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xs_ptr = gain_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ys_ptr = gain_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result_ptr = phase_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call the function

    _library_rf_noise.rf_noise_wrapper(
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
        ctypes.c_int(n_source),  # nSource
        ctypes.c_int(n_pnt_min),  # nPntMin
        ctypes.c_int(r_seed),  # rSeed
        ctypes.c_double(sampling_rate),  # samplingRate
        ctypes.c_double(rms),  # rms
    )
    return phase_array
