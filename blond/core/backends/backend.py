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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from types import ModuleType
    from typing import TYPE_CHECKING, Any, Literal

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

DEFAULT_BACKEND = "python"
DEFAULT_BITS = "64"


class Specials(ABC):
    """Abstract listing of functions that need implementation for a new backend."""

    @staticmethod
    @abstractmethod  # pragma: no cover
    def loss_box(  # noqa: D102
        top: float,
        bottom: float,
        left: float,
        right: float,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `loss_box` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_single_harmonic(  # noqa: D102
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
    def kick_multi_harmonic(  # noqa: D102
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
    def drift_simple(  # noqa: D102
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
    def drift_legacy(  # noqa: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_order: int,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_legacy` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_exact(  # noqa: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_exact` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_induced_voltage(  # noqa: D102
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
    def histogram(  # noqa: D102
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
    def beam_phase(  # noqa: D102
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
    ):
        """Reorders entries where ``flags == flag`` to the array end.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
        flags
            Macro-particle flags
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
        ids
            Macro-particle ids.
            This allows to identify single particles,
            even if the array indexing is changed.
        """
        raise NotImplementedError(
            "The backend for `move_flagged_elements_to_end` is missing."
        )


class _ModeSwitchHelper:
    """Helper to be used in a `with` statement to set the specials temporarily.

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
    """Base class for a backend.

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
    """

    # type annotations for MyPy
    float: type[np.float32 | np.float64]
    complex: type[np.complex128 | np.complex64]

    def __init__(
        self,
        float_: type[np.float32 | np.float64],
        complex_: type[np.complex128 | np.complex64],
        specials_mode: Literal[
            "python",
            "cpp",
            "numba",
            "fortran",
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
        self.linspace: Callable = None  # type: ignore
        self.histogram: Callable = None  # type: ignore
        self.zeros: Callable = None  # type: ignore
        self.ones: Callable = None  # type: ignore
        self.zeros_like: Callable = None  # type: ignore
        self.fft: ModuleType = None  # type: ignore
        self.random: ModuleType = None  # type: ignore
        self.isnan: Callable = None  # type: ignore
        self.sum: Callable = None  # type: ignore
        self.sqrt: Callable = None  # type: ignore
        self.meshgrid: Callable = None  # type: ignore
        self.square: Callable = None  # type: ignore
        self.mean: Callable = None  # type: ignore
        self.arange: Callable = None  # type: ignore
        self.average: Callable = None  # type: ignore

    def _finalize(self) -> None:
        for attribute, val in self.__dict__.items():
            if val is None:
                raise AttributeError(f"{self.__class__}.{attribute} is None.")

    def change_backend(
        self,
        new_backend: type[Numpy32Bit | Numpy64Bit | Cupy32Bit | Cupy64Bit],
    ) -> None:
        """Changes the backend precision.

        Parameters
        ----------
        new_backend
            One of the available backends

        """
        if self.__class__ == new_backend.__class__:
            return
        if self.verbose:
            print(f"Changing backend to `{new_backend.__name__}`")
        _new_backend = new_backend()
        # transfer variables that should be kept when changing backend.

        _new_backend.verbose = self.verbose
        self.__dict__ = _new_backend.__dict__
        self.__class__ = _new_backend.__class__
        self.set_specials(self.specials_mode)  # TODO test changing backends

    @abstractmethod  # pragma: no cover
    def set_specials(self, mode: Any) -> None:
        """Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        raise NotImplementedError(
            "Abstract method `set_specials` is not implemented."
        )

    @property
    def is_gpu(self) -> bool:
        """Whether the backend is using the GPU."""
        return self._is_gpu

    def apply_environment_variables(self) -> None:  # NOQA PLR0912
        """Load the environment variables and set up the backend accordingly.

        Notes
        -----
        Following environment variables can be set:

        - `BLOND_BACKEND_MODE` can be 'python', 'cpp', 'numba', 'fortran', 'cuda'
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
            "numba",
            "fortran",
            "cuda",
        )
        if _backend_mode_raw in _allowed_backend_modes:
            _backend_mode: Literal[
                "python",
                "cpp",
                "numba",
                "fortran",
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
            self.set_specials(mode=_backend_mode)  # type: ignore
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
            self.set_specials(mode=_backend_mode)  # type: ignore

    def temporary_specials_mode(self, mode: str):
        """Helper to be used in a `with` statement to set the specials temporarily.

        Examples
        --------
        >>> with backend.temporary_specials_mode("python"):
        >>>     print(backend.specials_mode)
        >>>     ...
        >>> print(backend.specials_mode)

        Returns
        -------
        _mode_switch_helper

        """
        return _ModeSwitchHelper(backend=self, mode=mode)


class NumpyBackend(BackendBaseClass):
    """Base class for Numpy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float32, float64.
    complex_
        Precision type for complex, e.g. float32, float64.
    """

    def __init__(
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
        self.array = np.array
        self.gradient = np.gradient
        self.linspace = np.linspace
        self.histogram = np.histogram
        self.zeros = np.zeros
        self.ones = np.ones
        self.zeros_like = np.zeros_like
        self.fft = np.fft
        self.random = np.random
        self.isnan = np.isnan
        self.sum = np.sum
        self.sqrt = np.sqrt
        self.interp = np.interp
        self.meshgrid = np.meshgrid
        self.square = np.square
        self.mean = np.mean
        self.arange = np.arange
        self.average = np.average

        self._finalize()

    def set_specials(
        self,
        mode: Literal[
            "python",
            "cpp",
            "numba",
            "fortran",
        ],
    ) -> None:
        """Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        onchange = self.specials_mode != mode

        if mode == "python":
            from blond.core.backends.python.callables import PythonSpecials

            self.specials = PythonSpecials()
            self.specials_mode = mode
        elif mode == "cpp":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(self.float)
            self.specials_mode = mode
        elif mode == "numba":
            from blond.core.backends.numba.callables import (
                recompile_numba_backend,
            )

            NumbaSpecials = recompile_numba_backend(self.float)
            self.specials = NumbaSpecials()
            self.specials_mode = mode
        elif mode == "fortran":
            from blond.core.backends.fortran.callables import (
                reload_fortran_backend,
            )

            FortranSpecials = reload_fortran_backend(self.float)

            self.specials = FortranSpecials()
            self.specials_mode = mode
        else:
            raise ValueError(mode)
        if self.verbose and onchange:
            print(f"Set special to `{mode}`")


class Numpy32Bit(NumpyBackend):
    """Numpy backend with 32 bit precision."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            np.float32,
            np.complex64,
        )


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
    """Base class for Cupy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float32, float64.
    complex_
        Precision type for complex, e.g. float32, float64.
    """

    def __init__(
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

        self.array = cp.array
        self.gradient = cp.gradient
        self.linspace = cp.linspace
        self.histogram = cp.histogram
        self.zeros = cp.zeros
        self.ones = cp.ones
        self.zeros_like = cp.zeros_like
        self.fft = cp.fft
        self.random = cp.random
        self.isnan = cp.isnan
        self.sum = cp.sum
        self.sqrt = cp.sqrt
        self.interp = cp.interp
        self.meshgrid = cp.meshgrid
        self.square = cp.square
        self.mean = cp.mean
        self.arange = cp.arange
        self.average = cp.average

        from blond.core.backends.cuda.callables import CudaSpecials

        self.specials = CudaSpecials()

        self._finalize()

    def set_specials(self, mode: Literal["cuda"]) -> None:
        """Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        if mode == "cuda":
            from blond.core.backends.cuda.callables import reload_cuda_backend

            CudaSpecials = reload_cuda_backend(self.float)

            self.specials = CudaSpecials()
        else:
            raise ValueError(mode)
        if self.verbose:
            print(f"Set special to `{mode}`")


class Cupy32Bit(CupyBackend):
    """Cupy backend with 64 bit precision."""

    def __init__(self) -> None:
        super().__init__(
            np.float32,
            np.complex64,
        )


class Cupy64Bit(CupyBackend):
    """Cupy backend with 32 bit precision."""

    def __init__(self) -> None:
        super().__init__(
            np.float64,
            np.complex128,
        )


default = Numpy32Bit()  # use .change_backend(...) to change it anywhere
backend: Numpy32Bit | Numpy64Bit | Cupy32Bit | Cupy64Bit = default
backend.verbose = True
backend.apply_environment_variables()
