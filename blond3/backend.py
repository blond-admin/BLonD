from __future__ import annotations

import abc
from abc import ABC
from typing import Literal

import numpy as np


class BackendBaseClass(ABC):
    def __init__(
        self, float: Literal[np.float32, np.float64], int: Literal[np.int32, np.int64]
    ):
        self.float = float
        self.int = int

        self.array = None

    def change_backend(self, new_backend: BackendBaseClass):
        self.__dict__ = new_backend.__dict__
        self.__class__ = new_backend.__class__


class NumpyBackend(BackendBaseClass):
    def __init__(
        self, float: Literal[np.float32, np.float64], int: Literal[np.int32, np.int64]
    ):
        super().__init__(float, int)
        self.array = np.array


class Numpy32Bit(BackendBaseClass):
    def __init__(self):
        super().__init__(np.float32, np.int32)


class Numpy64Bit(BackendBaseClass):
    def __init__(self):
        super().__init__(np.float64, np.int64)


default = Numpy32Bit()  # use .change_backend(...) to change it anywhere
backend = default
