from __future__ import annotations

import json
import os.path
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray, DTypeLike

from .helpers import callers_relative_path

if TYPE_CHECKING:  # pragma: no cover
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
        filepath_is_relative=True,
        dtype: Optional[DTypeLike] = None,
        order: Literal["C", "F"] = "C",
        overwrite=True,
    ):
        if filepath_is_relative:
            filepath = callers_relative_path(filepath, stacklevel=2)
        self._memory = np.empty(shape=shape, dtype=dtype, order=order)
        self._write_idx = 0

        self.filepath = filepath
        self.overwrite = overwrite
        if not self.overwrite:
            if os.path.exists(self.filepath_array):
                warnings.warn(f"{self.filepath_array} exists already!")

    @property
    def filepath_array(self):
        return f"{self.filepath}.npy"

    @property
    def filepath_attributes(self):
        return f"{self.filepath}.json"

    def purge_from_disk(self, verbose=True):
        if os.path.exists(self.filepath_array):
            os.remove(self.filepath_array)
            if verbose:
                print(f"Removed {self.filepath_array}")
        if os.path.exists(self.filepath_attributes):
            os.remove(self.filepath_attributes)
            if verbose:
                print(f"Removed {self.filepath_attributes}")

    def to_disk(self):
        if not self.overwrite:
            assert not os.path.exists(self.filepath_array)
        np.save(self.filepath_array, self._memory)
        attributes = dict(
            _write_idx=self._write_idx,
            overwrite=self.overwrite,
        )
        with open(self.filepath_attributes, "w") as f:
            json.dump(attributes, f)

    @staticmethod
    def from_disk(filepath: str | Path) -> DenseArrayRecorder:
        dense_recorder = DenseArrayRecorder(
            filepath=filepath,
            shape=(1, 1),
        )
        _memory: NumpyArray = np.load(dense_recorder.filepath_array)
        dense_recorder._memory = _memory
        with open(dense_recorder.filepath_attributes, "r") as f:
            loaded_data = json.load(f)
        dense_recorder._write_idx = loaded_data["_write_idx"]
        dense_recorder.overwrite = loaded_data["overwrite"]
        return dense_recorder

    def write(self, newdata: NumpyArray):
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self):
        return self._memory[: self._write_idx]


class ChunkedArrayRecorder(ArrayRecorder):
    def __init__(self):
        raise NotImplementedError()  # TODO
