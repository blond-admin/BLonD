from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Type, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class Specials(ABC):
    """
    Abstract listing of functions that need implementation for a new backend
    """

    @staticmethod
    @abstractmethod  # pragma: no cover
    def loss_box(
        self, top: float, bottom: float, left: float, right: float
    ) -> None:  # TODO
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_multi_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_legacy(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_order: int,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ):
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def histogram(
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: float,
        stop: float,
    ):
        return

    @staticmethod
    @abstractmethod  # pragma: no cover
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        pass


class BackendBaseClass(ABC):
    def __init__(
        self,
        float_: Type[Union[np.float32, np.float64]],
        int_: Type[np.int32 | np.int64],
        complex_: Type[Union[np.complex128, np.complex64]],
        specials_mode: Literal[
            "python",
            "cpp",
            "numba",
            "fortran",
            "cuda",
        ],
        is_gpu: bool,
    ):
        """
        Base class for a backend.

        Parameters
        ----------
        float_
            Precision type for float, e.g. float32, float64.
        int_
            Precision type for int, e.g. float32, float64.
        complex_
            Precision type for complex, e.g. float32, float64.
        specials_mode
            Default mode to load special libraries.
        is_gpu
            Whether the backend is using the GPU.
        """
        self._is_gpu = is_gpu

        self.float: Type[Union[np.float32, np.float64]] = float_
        self.int: Type[np.int32 | np.int64] = int_
        self.complex: Type[np.complex128 | np.complex64] = complex_

        self.twopi = self.float(2 * np.pi)
        self.specials_mode = specials_mode
        self.specials: Specials = None  # NOQA
        self.set_specials(self.specials_mode)

        # Callables that link to e.g. Numpy, Cupy
        self.array = None
        self.gradient = None
        self.linspace = None
        self.histogram = None
        self.zeros = None

    def change_backend(
        self, new_backend: Type[Numpy32Bit, Numpy64Bit, Cupy32Bit, Cupy64Bit]
    ) -> None:
        """
        Changes the backend precision

        Parameters
        ----------
        new_backend
            One of the available backends

        """
        _new_backend = new_backend()
        self.__dict__ = _new_backend.__dict__
        self.__class__ = _new_backend.__class__
        self.set_specials(self.specials_mode)  # TODO test changing backends

    @abstractmethod  # pragma: no cover
    def set_specials(self, mode) -> None:
        """
        Set the special compiled functions

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        pass

    @property
    def is_gpu(self) -> bool:
        """
        Whether the backend is using the GPU
        """
        return self._is_gpu


def fresh_import(module_location: str, class_name: str) -> type:
    """
    To freshly do `from module_location import ClassName`



    Parameters
    ----------
    module_location
        Import location where the module resides
    class_name
        Class to re-import

    Returns
    -------
    Newly imported class

    """
    # TODO Refactor given files as classes, so that only reinstancing of a
    #  class is needed instead of reloading a module path.
    #  This function is only intended to reload backend specials.
    if module_location in sys.modules:
        del sys.modules[module_location]
    module = importlib.import_module(module_location)
    return getattr(module, class_name)


class NumpyBackend(BackendBaseClass):
    def __init__(
        self,
        float_: Union[np.float32, np.float64],
        int_: np.int32 | np.int64,
        complex_: Union[np.complex128, np.complex64],
    ):
        """
        Base class for Numpy based backends

        Parameters
        ----------
        float_
            Precision type for float, e.g. float32, float64.
        int_
            Precision type for int, e.g. float32, float64.
        complex_
            Precision type for complex, e.g. float32, float64.
        """
        super().__init__(
            float_,
            int_,
            complex_,
            specials_mode="python",
            is_gpu=False,
        )
        self.array = np.array
        self.gradient = np.gradient
        self.linspace = np.linspace
        self.histogram = np.histogram
        self.zeros = np.zeros

    def set_specials(
        self,
        mode: Literal[
            "python",
            "cpp",
            "numba",
            "fortran",
        ],
    ):
        """
        Set the special compiled functions

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        if mode == "python":
            from .python.callables import PythonSpecials

            self.specials = PythonSpecials()
            self.specials_mode = mode
        elif mode == "cpp":
            CppSpecials = fresh_import(
                "blond._core.backends.cpp.callables",
                "CppSpecials",
            )
            self.specials = CppSpecials()
            self.specials_mode = mode
        elif mode == "numba":
            # like
            # from .numba.callables import NumbaSpecials
            # but reimport, so that dtypes are in line with the current backend
            NumbaSpecials = fresh_import(
                "blond._core.backends.numba.callables",
                "NumbaSpecials",
            )
            self.specials = NumbaSpecials()
            self.specials_mode = mode
        elif mode == "fortran":
            FortranSpecials = fresh_import(
                "blond._core.backends.fortran.callables",
                "FortranSpecials",
            )
            self.specials = FortranSpecials
            self.specials_mode = mode
        else:
            raise ValueError(mode)


class Numpy32Bit(NumpyBackend):
    def __init__(
        self,
    ):
        """
        Numpy backend with 32 bit precision.
        """
        super().__init__(
            np.float32,
            np.int32,
            np.complex64,
        )


class Numpy64Bit(NumpyBackend):
    def __init__(
        self,
    ):
        """
        Numpy backend with 64 bit precision.
        """
        super().__init__(
            np.float64,
            np.int64,
            np.complex128,
        )


class CupyBackend(BackendBaseClass):
    def __init__(
        self,
        float_: Union[np.float32, np.float64],
        int_: np.int32 | np.int64,
        complex_: Union[np.complex128, np.complex64],
    ):
        """
        Base class for Cupy based backends

        Parameters
        ----------
        float_
            Precision type for float, e.g. float32, float64.
        int_
            Precision type for int, e.g. float32, float64.
        complex_
            Precision type for complex, e.g. float32, float64.
        """
        super().__init__(
            float_,
            int_,
            complex_,
            specials_mode="cuda",  # no other backend implemented at the moment
            is_gpu=True,
        )
        import cupy as cp  # import only if needed, which is not always the case

        self.array = cp.array
        self.gradient = cp.gradient
        self.linspace = cp.linspace
        self.histogram = cp.histogram
        self.zeros = cp.zeros

        from .cuda.callables import CudaSpecials

        self.specials = CudaSpecials()

    def set_specials(self, mode: Literal["cuda"]):
        """
        Set the special compiled functions

        Parameters
        ----------
        mode
            One of the available backend modes

        """
        if mode == "cuda":
            CudaSpecials = fresh_import(
                "blond._core.backends.cuda.callables",
                "CudaSpecials",
            )

            self.specials = CudaSpecials()
        else:
            raise ValueError(mode)


class Cupy32Bit(CupyBackend):
    def __init__(self):
        """
        Cupy backend with 64 bit precision.
        """
        super().__init__(
            np.float32,
            np.int32,
            np.complex64,
        )


class Cupy64Bit(CupyBackend):
    def __init__(self):
        """
        Cupy backend with 32 bit precision.
        """
        super().__init__(
            np.float64,
            np.int64,
            np.complex128,
        )


default = Numpy32Bit()  # use .change_backend(...) to change it anywhere
backend: Numpy32Bit | Numpy64Bit | Cupy32Bit | Cupy64Bit = default
