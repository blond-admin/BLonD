from __future__ import annotations

import json
import os.path
import warnings
from abc import ABC, abstractmethod
from os.path import isfile
from typing import TYPE_CHECKING

import numpy as np

from .._generals.cupy.no_cupy_import import is_cupy_array
from .helpers import callers_relative_path

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Literal, Optional, Tuple

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray as NumpyArray


class ArrayRecorder(ABC):
    @abstractmethod  # pragma: no cover
    def write(self, newdata: NumpyArray) -> None:
        pass

    @abstractmethod  # pragma: no cover
    def get_valid_entries(self) -> NumpyArray:
        pass

    @abstractmethod  # pragma: no cover
    def to_disk(self) -> None:
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def from_disk(filepath: str | PathLike) -> ArrayRecorder:
        pass


class DenseArrayRecorder(ArrayRecorder):
    def __init__(
        self,
        filepath: str | PathLike,
        shape: int | Tuple[int, ...],
        dtype: Optional[DTypeLike] = None,
        order: Literal["C", "F"] = "C",
        overwrite: bool = True,
    ):
        if filepath_is_relative:
            filepath = callers_relative_path(filepath, stacklevel=2)
        # reserve full memory at init to avoid memory overflow during runtime
        self._memory = np.empty(shape=shape, dtype=dtype, order=order)
        self._write_idx = 0

        self.filepath = filepath
        self.overwrite = overwrite
        if not self.overwrite:
            if os.path.exists(self.filepath_array):
                warnings.warn(f"{self.filepath_array} exists already!")

    @property
    def filepath_array(self) -> str:
        return f"{self.filepath}.npy"

    @property
    def filepath_attributes(self) -> str:
        return f"{self.filepath}.json"

    def purge_from_disk(self, verbose: bool = True):
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
    def from_disk(filepath: str | PathLike) -> DenseArrayRecorder:
        dense_recorder = DenseArrayRecorder(
            filepath=filepath,
            shape=(1, 1),
        )
        assert isfile(dense_recorder.filepath_array)
        _memory: NumpyArray = np.load(dense_recorder.filepath_array)
        dense_recorder._memory = _memory
        with open(dense_recorder.filepath_attributes, "r") as f:
            loaded_data = json.load(f)
        dense_recorder._write_idx = loaded_data["_write_idx"]
        dense_recorder.overwrite = loaded_data["overwrite"]
        return dense_recorder

    def write(self, newdata: NumpyArray | float | CupyArray):
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self) -> NumpyArray:
        if self._write_idx == 0:
            ValueError(
                "Cannot retrieve results:"
                " no data has been written to memory yet."
            )
        return self._memory[: self._write_idx]


class ChunkedArrayRecorder(ArrayRecorder):
    def __init__(self):
        raise NotImplementedError()  # TODO
