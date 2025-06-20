from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BackendBaseClass(ABC):
    def __init__(self, float_: np.float32 | np.float64, int_: np.int32 | np.int64):
        self.float: np.float32 | np.float64 = float_
        self.int: np.int32 | np.int64 = int_

        self.twopi = self.float(2 * np.pi)
        # Callables
        self.array = None
        self.gradient = None

    def change_backend(self, new_backend: BackendBaseClass):
        self.__dict__ = new_backend.__dict__
        self.__class__ = new_backend.__class__

    @abstractmethod
    def loss_box(self, a, b, c, d) -> None:  # TODO
        pass


class NumpyBackend(BackendBaseClass):
    def __init__(self, float_: np.float32 | np.float64, int_: np.int32 | np.int64):
        super().__init__(float_, int_)
        self.array = np.array
        self.gradient = np.gradient


class Numpy32Bit(NumpyBackend):
    def __init__(self):
        super().__init__(np.float32, np.int32)


class Numpy64Bit(NumpyBackend):
    def __init__(self):
        super().__init__(np.float64, np.int64)


default = Numpy32Bit()  # use .change_backend(...) to change it anywhere
backend = default
