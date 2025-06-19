from __future__ import annotations

import os.path
import warnings
from abc import ABC, abstractmethod
from typing import (
    Tuple,
    Optional,
    Literal,
)

import numpy as np
from numpy.typing import NDArray as NumpyArray, DTypeLike


class ArrayRecorder(ABC):
    @abstractmethod
    def write(self, newdata: NumpyArray):
        pass

    @abstractmethod
    def get_valid_entries(self):
        pass

    @abstractmethod
    def to_disk(self):
        pass

    @abstractmethod
    def from_disk(self):
        pass


class DenseArrayRecorder(ArrayRecorder):
    def __init__(
        self,
        filepath: str,
        shape: int | Tuple[int, ...],
        dtype: Optional[DTypeLike] = None,
        order: Literal["C", "F"] = "C",
        overwrite=True,
    ):
        self._memory = np.empty(shape=shape, dtype=dtype, order=order)
        self._write_idx = 0
        self.filepath = filepath
        self.overwrite = overwrite
        if not self.overwrite:
            if os.path.exists(self.filepath):
                warnings.warn(f"{self.filepath} exists already!")

    def to_disk(self):
        if not self.overwrite:
            assert not os.path.exists(self.filepath)
        np.save(self.filepath, self.get_valid_entries())

    def from_disk(self):
        pass

    def write(self, newdata: NumpyArray):
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self):
        return self._memory[: self._write_idx]


class ChunkedArrayRecorder(ArrayRecorder):
    pass
