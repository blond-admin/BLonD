# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Global definitions for the capabilities of all backends."""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.exceptions import ComplexWarning

from blond.generals.exceptions_ import ArrayCastingError
from blond.generals.warnings_ import PrecisionWarning

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from types import ModuleType
    from typing import TYPE_CHECKING, Any, Literal

    from cupy import ndarray as CupyArray  # type: ignore
    from numpy.typing import NDArray

    NumpyArray = NDArray[Any]
    from numpy.typing import ArrayLike


DEFAULT_BACKEND = "python"
DEFAULT_BITS = "64"

ALL_BACKENDS: dict[str, type[BackendBaseClass]] = {}
AVAILABLE_BACKENDS: dict[str, type[BackendBaseClass]] = {}


def _register_backend(bd: type[BackendBaseClass]) -> type[BackendBaseClass]:
    ALL_BACKENDS[bd.__name__] = bd
    return bd


class Specials(ABC):
    """Abstract listing of functions that need implementation for a new backend."""

    @staticmethod
    @abstractmethod  # pragma: no cover
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `loss_box` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_single_harmonic(  # NOQA: D102
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `kick_single_harmonic` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_multi_harmonic(  # NOQA: D102
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `kick_multi_harmonic` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def sum_1d_array(array: NumpyArray | CupyArray) -> float:
        """
        Return the sum of an 1d array.

        Parameters
        ----------
        array
            Input array 1.

        Returns
        -------
        sum_1d_array
            Sum of a 1d arrays.
        """
        raise NotImplementedError(
            "Abstract method `sum_1d_array` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def dot_product_1d_array(
        array_1: NumpyArray | CupyArray, array_2: NumpyArray | CupyArray
    ) -> float:
        """
        Return the sum of dot product of two 1d arrays.

        Parameters
        ----------
        array_1
            Input array 1.
        array_2
            Input array 2.

        Returns
        -------
        dot_product_1d_array
            Dot product of two 1d arrays.
        """
        raise NotImplementedError(
            "Abstract method `dot_product_1d_array` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_simple(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_simple` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_exact(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        higher_alpha: NumpyArray,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_exact` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_induced_voltage(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `kick_induced_voltage` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def histogram(  # NOQA: D102
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: float,
        stop: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `histogram` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def beam_phase(  # NOQA: D102
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        raise NotImplementedError(
            "Abstract method `beam_phase` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def move_flagged_elements_to_end(
        flag: int,
        flags: NumpyArray | CupyArray,  # also purged
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        ids: NumpyArray | CupyArray,
    ) -> None:
        """
        Reorder entries where ``flags == flag`` to the array end.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
        flags
            Macro-particle flags.
        dt
            Macro-particle time coordinates [s].
        dE
            Macro-particle energy coordinates [eV].
        ids
            Macro-particle ids.
            This allows to identify single particles,
            even if the array indexing is changed.
        """
        raise NotImplementedError(
            "The backend for `move_flagged_elements_to_end` is missing."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def histogram_sparse(
        x: NumpyArray,
        out: NumpyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: NumpyArray,
        bucket_index_to_memory_index: NumpyArray,
    ) -> None:
        """
        Sparse histogram with strided memory layout (gaps between profiles).

        Parameters
        ----------
        x
            An array, e.g., the particle ``dt`` values.
        out
            Output histogram ``(n_filled_buckets * bins_per_profile)``.
        first_left_cut
            Start of the first histogram.
        left_cut_distance
            Distance between the start of each histogram.
        cut_width
            Distance between left and right edge of the histogram.
        bins_per_profile
            Number of bins per bucket.
        n_active_profiles
            Number of non-empty buckets.
        filling_pattern
            Filling pattern as a boolean array
            where ``True`` means filled bucket.
        bucket_index_to_memory_index
            Maps bucket index to memory index.
            For a ``filling_pattern = [1, 0, 0, 1]``
            ``bucket_index_to_memory_index = [0, 0, 0, 8]`` with
            ``bins_per_profile = 8``.
            Use `_gen_array_bucket_index_to_memory_index` to generate this.
        """


class _ModeSwitchHelper:
    """
    Helper to be used in a `with` statement to set the specials temporarily.

    Parameters
    ----------
    backend
        The active backend class.
    mode
        The mode of the specials to be set.
    """

    def __init__(self, backend: BackendBaseClass, mode: str):
        self.backend = backend
        self.mode_org = None
        self.mode_tmp = mode

    def __enter__(self):
        self.mode_org = self.backend.specials_mode
        self.backend.set_specials(mode=self.mode_tmp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.set_specials(mode=self.mode_org)


class BackendBaseClass(ABC):
    """
    Base class for a backend.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float32, float64.
    complex_
        Precision type for complex, e.g. float32, float64.
    specials_mode
        Default mode to load special libraries.
    is_gpu
        Whether the backend is using the GPU.
    verbose
        Enable verbose output.
    """

    # type annotations for MyPy
    float: type[np.float32 | np.float64]
    complex: type[np.complex128 | np.complex64]

    def __init__(  # noqa: PLR0915
        self,
        float_: type[np.float32 | np.float64],
        complex_: type[np.complex128 | np.complex64],
        specials_mode: Literal[
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
            "cuda",
        ],
        is_gpu: bool,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        self._is_gpu = is_gpu

        self.float = float_
        self.complex = complex_

        self.twopi = self.float(2 * np.pi)
        self.specials_mode = specials_mode
        self.specials: Specials = None  # type: ignore
        self.set_specials(self.specials_mode)

        # Callables that link to e.g. Numpy, Cupy
        self.array: Callable = None  # type: ignore
        self.gradient: Callable = None  # type: ignore
        self.isclose: Callable = None  # type: ignore
        self.empty: Callable = None  # type: ignore
        self.repeat: Callable = None  # type: ignore
        self.linspace: Callable = None  # type: ignore
        self.sinc: Callable = None  # type: ignore
        self.histogram: Callable = None  # type: ignore
        self.zeros: Callable = None  # type: ignore
        self.ones: Callable = None  # type: ignore
        self.zeros_like: Callable = None  # type: ignore
        self.fft: ModuleType = None  # type: ignore
        self.all: Callable = None  # type: ignore
        self.random: ModuleType = None  # type: ignore
        self.sinc: Callable = None  # type: ignore
        self.isnan: Callable = None  # type: ignore
        self.sum: Callable = None  # type: ignore
        self.sqrt: Callable = None  # type: ignore
        self.interp: Callable = None  # type: ignore
        self.meshgrid: Callable = None  # type: ignore
        self.square: Callable = None  # type: ignore
        self.mean: Callable = None  # type: ignore
        self.arange: Callable = None  # type: ignore
        self.average: Callable = None  # type: ignore
        self.fftconvolve: Callable = None  # type: ignore
        self.min: Callable = None  # type: ignore
        self.max: Callable = None  # type: ignore
        self.dot: Callable = None  # type: ignore
        self.percentile: Callable = None  # type: ignore
        self.cumulative_sum: Callable = None  # type: ignore
        self.array_split: Callable = None  # type: ignore
        self.sign: Callable = None  # type: ignore
        self.sin: Callable = None  # type: ignore
        self.cos: Callable = None  # type: ignore
        self.exp: Callable = None  # type: ignore
        self.any: Callable = None  # type: ignore
        self.abs: Callable = None  # type: ignore
        self.convolve: Callable = None  # type: ignore
        self.copy: Callable = None  # type: ignore
        self.ones_like: Callable = None  # type: ignore
        self.add: Callable = None  # type: ignore
        self.default_rng: Callable = None  # type: ignore
        self.concatenate: Callable = None  # type: ignore
        self.unique: Callable = None  # type: ignore
        self.repeat: Callable = None  # type: ignore
        self.ndarray: type = None  # type: ignore

    def _finalize(self) -> None:
        for attribute, val in self.__dict__.items():
            if val is None:
                raise AttributeError(f"{self.__class__}.{attribute} is None.")

    def autoselect_backend(self) -> None:
        """Set automatically the fastest backend that is available on the computer."""
        order = (
            (Cupy64Bit, "cuda"),
            (Numpy64Bit, "cpp"),
            (Numpy64Bit, "numba"),
            (Numpy64Bit, "python"),
        )
        for backend_, mode_ in order:
            try:
                self.change_backend(new_backend=backend_)
                self.set_specials(mode=mode_)
                return
            except Exception:
                pass

    def change_backend(
        self,
        new_backend: (
            type[Numpy32Bit]
            | type[Numpy64Bit]
            | type[Cupy32Bit]
            | type[Cupy64Bit]
            | type[BackendBaseClass]
        ),
    ) -> None:
        """
        Change the backend precision.

        Parameters
        ----------
        new_backend
            One of the available backends.
        """
        if self.__class__ == new_backend.__class__:
            return
        if self.verbose:
            print(f"Changing backend to `{new_backend.__name__}`")
        _new_backend = new_backend()  # ty:ignore[missing-argument]
        # transfer variables that should be kept when changing backend.

        _new_backend.verbose = self.verbose
        self.__dict__ = _new_backend.__dict__
        self.__class__ = _new_backend.__class__
        self.set_specials(self.specials_mode)  # TODO test changing backends

    @abstractmethod  # pragma: no cover
    def set_specials(self, mode: Any) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        raise NotImplementedError(
            "Abstract method `set_specials` is not implemented."
        )

    @property
    def is_gpu(self) -> bool:
        """
        Whether the backend is using the GPU.

        Returns
        -------
        is_gpu
            True if the backend is using the GPU, False otherwise.
        """
        return self._is_gpu

    def apply_environment_variables(self) -> None:  # NOQA PLR0915
        """
        Load the environment variables and set up the backend accordingly.

        Notes
        -----
        Following environment variables can be set:

        - `BLOND_BACKEND_MODE` can be 'python', 'cpp', 'numba', 'cuda'
        - `BLOND_BACKEND_BITS` can be '32' or '64'
        """
        _backend_mode_raw: str = os.environ.get(
            "BLOND_BACKEND_MODE",
            DEFAULT_BACKEND,  # default
        ).lower()
        if _backend_mode_raw != "numba":
            print(
                f"Using environment variable BLOND_BACKEND_MODE = {_backend_mode_raw}"
            )
        _allowed_backend_modes = (
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
            "cuda",
        )
        if _backend_mode_raw in _allowed_backend_modes:
            _backend_mode: Literal[
                "python",
                "cpp",
                "cpp_single_core",
                "numba",
                "cuda",
            ] = _backend_mode_raw  # type: ignore
        else:
            raise ValueError(
                f"The environment variable `BLOND_BACKEND` "
                f"was set to '{_backend_mode_raw}', but can only be one "
                f"of {_allowed_backend_modes}."
            )

        _backend_bits_raw: str = os.environ.get(
            "BLOND_BACKEND_BITS",
            DEFAULT_BITS,  # default
        )
        _allowed_backend_bits_flag = (
            "32",
            "64",
        )
        if _backend_bits_raw in _allowed_backend_bits_flag:
            _backend_bits: Literal[
                "32",
                "64",
            ] = _backend_bits_raw  # type: ignore
        else:
            raise ValueError(
                f"The environment variable `BLOND_BACKEND_BITS` "
                f"was set to '{_backend_bits_raw}', but can only be one "
                f"of {_allowed_backend_bits_flag}."
            )

        if _backend_mode == "cuda":
            if _backend_bits == "32":
                self.change_backend(Cupy32Bit)
            elif _backend_bits == "64":
                self.change_backend(Cupy64Bit)
            else:
                # This statement is not reachable
                # because of `_backend_bits_raw in _allowed_backend_bits_flag`
                # Anyways its beter to write if, elif, else explicitly
                raise ValueError(_backend_bits)  # pragma: no cover
            self.set_specials(mode=_backend_mode)
        else:
            if _backend_bits == "32":
                self.change_backend(Numpy32Bit)
            elif _backend_bits == "64":
                self.change_backend(Numpy64Bit)
            else:
                # This statement is not reachable
                # because of `_backend_bits_raw in _allowed_backend_bits_flag`
                # Anyways its beter to write if, elif, else explicitly
                raise ValueError(_backend_bits)  # pragma: no cover
            self.set_specials(mode=_backend_mode)

    def temporary_specials_mode(self, mode: str):
        """
        Helper to be used in a `with` statement to set the specials temporarily.

        Parameters
        ----------
        mode
            The mode to temporarily switch to.

        Returns
        -------
        mode_switch_helper
            Context manager for temporarily switching modes.

        Examples
        --------
        >>> with backend.temporary_specials_mode("python"):
        ...     print(backend.specials_mode)
        ...     ...
        >>> print(backend.specials_mode)
        """
        return _ModeSwitchHelper(backend=self, mode=mode)

    def _asarray_if_needed(self, arr: ArrayLike) -> NumpyArray | CupyArray:
        # Faster to check than cast, so only cast if needed
        if isinstance(arr, self.ndarray):
            return arr

        # Duck-typed cupy detection: `.device` and `.get()` are cupy-specific
        # attributes not present on the `ArrayLike` union.  The try/except
        # handles the runtime case where the attributes are absent.
        try:
            gpu_arr = arr.device != "cpu"  # ty: ignore[unresolved-attribute]
        except AttributeError:
            gpu_arr = False

        if gpu_arr:
            arr = arr.get()  # ty: ignore[unresolved-attribute]

        return self.array(arr)

    def _cast_dtype_if_needed(
        self, arr: NumpyArray | CupyArray, dtype: type
    ) -> NumpyArray | CupyArray:
        if arr.dtype != dtype:
            warnings.warn(
                f"Automatically casting dtype from {arr.dtype} to {dtype}",
                stacklevel=3,
                category=PrecisionWarning,
            )
            try:
                # Casting numpy complex array -> float is smooth and
                # includes an automatic ComplexWarning.  Trying to cast
                # a cupy array in the same way raises an exception.
                # Maybe a bug in CuPy?
                # Catch the exception then throw the correct warning.
                arr = arr.astype(dtype)
            # Can be removed some years after 2025.
            except AttributeError as e:  # pragma: no cover
                # Cupy bugfix needed for `cupy-cuda12x<14.0.1`
                if (
                    str(e)
                    == "module 'numpy' has no attribute 'ComplexWarning'"
                ):
                    ComplexWarning(
                        "Casting complex values to real discards the imaginary part"
                    )
                    arr = arr.real.astype(dtype)
                else:  # pragma: no cover
                    raise

        return arr

    def _cast_arr_and_dtype(
        self, arr: ArrayLike, dtype: type
    ) -> NumpyArray | CupyArray:
        # Catch likely errors and reraise with slightly friendlier
        # messages.  Raise from the original exception to aid
        # debugging.
        #
        # ValueError is raised by backend.array(arr) if input is ragged.
        # TypeError is raised by arr.astype(backend.[type]) if input
        # cannot be coerced to the new type (e.g. str -> float)

        try:
            new_arr = self._asarray_if_needed(arr)
        except ValueError as exc:
            raise ArrayCastingError(
                f"Unable to convert input data {arr} to array."
            ) from exc

        try:
            new_arr = self._cast_dtype_if_needed(new_arr, dtype)
        except (TypeError, ValueError) as exc:
            raise ArrayCastingError(
                "Unable to automatically cast dtype of input data from "
                f"{new_arr.dtype} to {dtype}."
            ) from exc

        return new_arr

    def cast_arr_float_if_needed(
        self, arr: ArrayLike
    ) -> NumpyArray | CupyArray:
        """
        Convert input to backend.array with ``dtype=backend.float``.

        Uses isinstance and dtype checks to only modify the object if
        needed, which is faster and avoids breaking references.  If the
        reference is required to change, `backend.array` should be
        called directly.

        Parameters
        ----------
        arr
            The object that should be returned as an array.

        Returns
        -------
        NumpyArray | CupyArray
            The modified (if needed) array.
        """
        return self._cast_arr_and_dtype(arr, self.float)

    def cast_arr_complex_if_needed(
        self, arr: ArrayLike
    ) -> NumpyArray | CupyArray:
        """
        Convert input to backend.array with ``dtype=backend.complex``.

        Uses isinstance and dtype checks to only modify the object if
        needed, which is faster and avoids breaking references.  If the
        reference is required to change, `backend.array` should be
        called directly.

        Parameters
        ----------
        arr
            The object that should be returned as an array.

        Returns
        -------
        NumpyArray | CupyArray
            The modified (if needed) array.
        """
        return self._cast_arr_and_dtype(arr, self.complex)


class NumpyBackend(BackendBaseClass):
    """
    Base class for Numpy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float32, float64.
    complex_
        Precision type for complex, e.g. float32, float64.
    """

    def __init__(  # noqa: PLR0915
        self,
        float_: type[np.float32 | np.float64],
        complex_: type[np.complex128 | np.complex64],
    ) -> None:
        super().__init__(
            float_,
            complex_,
            specials_mode="python",
            is_gpu=False,
        )
        from scipy.signal import fftconvolve

        self.array = np.array
        self.gradient = np.gradient
        self.isclose = np.isclose
        self.empty = np.empty
        self.repeat = np.repeat
        self.linspace = np.linspace
        self.sinc = np.sinc
        self.histogram = np.histogram
        self.zeros = np.zeros
        self.ones = np.ones
        self.zeros_like = np.zeros_like
        self.fft = np.fft
        self.all = np.all
        self.random = np.random
        self.sinc = np.sinc
        self.isnan = np.isnan
        self.sum = np.sum
        self.sqrt = np.sqrt
        self.interp = np.interp
        self.meshgrid = np.meshgrid
        self.square = np.square
        self.mean = np.mean
        self.arange = np.arange
        self.average = np.average
        self.fftconvolve = fftconvolve
        self.min = np.min
        self.max = np.max
        self.dot = np.dot
        self.percentile = np.percentile
        try:  # pragma: no cover
            self.cumulative_sum = np.cumulative_sum
        except AttributeError:  # pragma: no cover
            self.cumulative_sum = np.cumsum
        self.array_split = np.array_split
        self.sign = np.sign
        self.sin = np.sin
        self.cos = np.cos
        self.exp = np.exp
        self.any = np.any
        self.abs = np.abs
        self.convolve = np.convolve
        self.copy = np.copy
        self.ones_like = np.ones_like
        self.add = np.add
        self.default_rng = np.random.default_rng
        self.concatenate = np.concatenate
        self.unique = np.unique
        self.repeat = np.repeat
        self.ndarray = np.ndarray

        self._finalize()

    def set_specials(
        self,
        mode: Literal[
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
        ],
    ) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        onchange = self.specials_mode != mode

        if mode == "python":
            from blond.core.backends.python.callables import PythonSpecials

            self.specials = PythonSpecials()
            self.specials_mode = mode
        elif mode == "cpp":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(self.float, parallel=True)
            self.specials_mode = mode
        elif mode == "cpp_single_core":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(self.float, parallel=False)
            self.specials_mode = mode
        elif mode == "numba":
            from blond.core.backends.numba.callables import (
                recompile_numba_backend,
            )

            NumbaSpecials = recompile_numba_backend(self.float)
            self.specials = NumbaSpecials()
            self.specials_mode = mode
        else:
            raise ValueError(mode)
        if self.verbose and onchange:
            print(f"Set special to `{mode}`")


@_register_backend
class Numpy32Bit(NumpyBackend):
    """Numpy backend with 32 bit precision."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            np.float32,
            np.complex64,
        )


@_register_backend
class Numpy64Bit(NumpyBackend):
    """Numpy backend with 64 bit precision."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            np.float64,
            np.complex128,
        )


class CupyBackend(BackendBaseClass):
    """
    Base class for Cupy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float32, float64.
    complex_
        Precision type for complex, e.g. float32, float64.
    """

    def __init__(  # noqa: PLR0915
        self,
        float_: type[np.float32 | np.float64],
        complex_: type[np.complex128 | np.complex64],
    ) -> None:
        super().__init__(
            float_,
            complex_,
            specials_mode="cuda",  # no other backend implemented at the moment
            is_gpu=True,
        )
        import cupy as cp  # type: ignore # import only if needed, which is not always the case

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="cupyx.jit.rawkernel is experimental. The interface can change in the future.",
            )
            from cupyx.scipy.signal import (
                fftconvolve,  # ty: ignore[unresolved-import]
            )

        self.array = cp.array
        self.gradient = cp.gradient
        self.isclose = cp.isclose
        self.empty = cp.empty
        self.repeat = cp.repeat
        self.linspace = cp.linspace
        self.sinc = cp.sinc
        self.histogram = cp.histogram
        self.zeros = cp.zeros
        self.ones = cp.ones
        self.zeros_like = cp.zeros_like
        self.fft = cp.fft
        self.all = cp.all
        self.random = cp.random
        self.sinc = cp.sinc
        self.isnan = cp.isnan
        self.sum = cp.sum
        self.sqrt = cp.sqrt
        self.interp = cp.interp
        self.meshgrid = cp.meshgrid
        self.square = cp.square
        self.mean = cp.mean
        self.arange = cp.arange
        self.average = cp.average
        self.fftconvolve = fftconvolve
        self.min = cp.min
        self.max = cp.max
        self.dot = cp.dot
        self.percentile = cp.percentile
        self.cumulative_sum = cp.cumsum
        self.array_split = cp.array_split
        self.sign = cp.sign
        self.sin = cp.sin
        self.cos = cp.cos
        self.exp = cp.exp
        self.any = cp.any
        self.abs = cp.abs
        self.convolve = cp.convolve
        self.copy = cp.copy
        self.ones_like = cp.ones_like
        self.add = cp.add
        self.default_rng = cp.random.default_rng
        self.concatenate = cp.concatenate
        self.unique = cp.unique
        self.repeat = cp.repeat
        self.ndarray = cp.ndarray

        from blond.core.backends.cuda.callables import CudaSpecials

        self.specials = CudaSpecials()

        self._finalize()

    def set_specials(self, mode: Literal["cuda"]) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        if mode == "cuda":
            from blond.core.backends.cuda.callables import reload_cuda_backend

            CudaSpecials = reload_cuda_backend(self.float)

            self.specials = CudaSpecials()
        else:
            raise ValueError(mode)
        if self.verbose:
            print(f"Set special to `{mode}`")


@_register_backend
class Cupy32Bit(CupyBackend):
    """Cupy backend with 64 bit precision."""

    def __init__(self) -> None:
        super().__init__(
            np.float32,
            np.complex64,
        )


@_register_backend
class Cupy64Bit(CupyBackend):
    """Cupy backend with 32 bit precision."""

    def __init__(self) -> None:
        super().__init__(
            np.float64,
            np.complex128,
        )


default = Numpy64Bit  # use .change_backend(...) to change it anywhere
backend: Numpy32Bit | Numpy64Bit | Cupy32Bit | Cupy64Bit | BackendBaseClass = (
    default()
)
backend.verbose = True
backend.apply_environment_variables()


for k, v in ALL_BACKENDS.items():
    try:
        v()  # ty:ignore[missing-argument]
    # Skip on any exception, we only care that it's not available,
    # we don't care why.
    except Exception:  # pragma: no cover
        pass
    else:
        AVAILABLE_BACKENDS[k] = v
