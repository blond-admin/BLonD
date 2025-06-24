from __future__ import annotations

import os.path
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray, DTypeLike

if TYPE_CHECKING:
    from typing import (
        Tuple,
        Optional,
        Literal,
    )
    from os import PathLike


class ArrayRecorder(ABC):
    @abstractmethod
    def write(self, newdata: NumpyArray) -> None:
        pass

    @abstractmethod
    def get_valid_entries(self) -> NumpyArray:
        pass

    @abstractmethod
    def to_disk(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def from_disk(filepath: str | PathLike) -> ArrayRecorder:
        pass



class DenseArrayRecorder(ArrayRecorder):
    def __init__(
        self,
        filepath: str | PathLike,
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

    @staticmethod
    def from_disk(filepath: str | PathLike):
        _memory: NumpyArray = np.load(filepath)
        dense_recorder = DenseArrayRecorder(
            filepath=filepath,
            shape=(1, 1),
        )
        dense_recorder._memory = _memory
        dense_recorder._write_idx = _memory.shape[0]
        return dense_recorder

    def write(self, newdata: NumpyArray):
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self):
        return self._memory[: self._write_idx]


class ChunkedArrayRecorder(ArrayRecorder):
    def __init__(self):
        raise NotImplementedError()  # TODO
