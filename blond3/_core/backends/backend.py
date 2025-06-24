from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray


class Specials(ABC):
    @staticmethod
    @abstractmethod
    def loss_box(self, a, b, c, d) -> None:  # TODO
        pass


    @staticmethod
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: float,
        length_ratio: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
    @abstractmethod
    def drift_legacy(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: float,
        length_ratio: float,
        alpha_order,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
    @abstractmethod
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: float,
        length_ratio: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ):
        pass


class BackendBaseClass(ABC):
    def __init__(
        self, float_: Union[np.float32, np.float64], int_: np.int32 | np.int64
    ):
        self.float: Union[np.float32, np.float64] = float_
        self.int: np.int32 | np.int64 = int_

        self.twopi = self.float(2 * np.pi)

        from .python.callables import PythonSpecials

        self.specials = PythonSpecials()

        # Callables
        self.array = None
        self.gradient = None

    def change_backend(
        self, new_backend: Type[Numpy32Bit, Numpy64Bit, Cupy32Bit, Cupy64Bit]
    ):
        _new_backend = new_backend()
        self.__dict__ = _new_backend.__dict__
        self.__class__ = _new_backend.__class__

    @abstractmethod
    def set_specials(self, mode):
        pass


class NumpyBackend(BackendBaseClass):
    def __init__(
        self, float_: Union[np.float32, np.float64], int_: np.int32 | np.int64
    ):
        super().__init__(float_, int_)
        self.array = np.array
        self.gradient = np.gradient

    def set_specials(self, mode):
        if mode == "python":
            from .python.callables import PythonSpecials

            self.specials = PythonSpecials()
        elif mode == "cpp":
            from .cpp.callables import CppSpecials

            self.specials = CppSpecials()
        elif mode == "numba":
            from .numba.callables import NumbaSpecials

            self.specials = NumbaSpecials()
        else:
            raise ValueError(mode)


class Numpy32Bit(NumpyBackend):
    def __init__(self):
        super().__init__(np.float32, np.int32)


class Numpy64Bit(NumpyBackend):
    def __init__(self):
        super().__init__(np.float64, np.int64)


class CupyBackend(BackendBaseClass):
    def __init__(
        self, float_: Union[np.float32, np.float64], int_: np.int32 | np.int64
    ):
        super().__init__(float_, int_)
        self.array = np.array
        self.gradient = np.gradient

        from .cuda.callables import CudaSpecials

        self.specials = CudaSpecials()

    def set_specials(self, mode):
        if mode == "cuda":
            from .cuda.callables import CudaSpecials

            self.specials = CudaSpecials()
        else:
            raise ValueError(mode)


class Cupy32Bit(CupyBackend):
    def __init__(self):
        super().__init__(np.float32, np.int32)


class Cupy64Bit(CupyBackend):
    def __init__(self):
        super().__init__(np.float64, np.int64)


default = Numpy32Bit()  # use .change_backend(...) to change it anywhere
backend: Numpy32Bit = default
