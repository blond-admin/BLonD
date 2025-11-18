"""Classes that deal with memory management of simulation results.

Authors
-------
Leonard Thiele
Simon Lauber
"""

from __future__ import annotations

import json
import os.path
import warnings
from abc import ABC, abstractmethod
from os.path import isfile
from typing import TYPE_CHECKING

import numpy as np

from ..generals.cupy.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Literal

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray as NumpyArray


class ArrayRecorder(ABC):
    """Base class to save content to an array."""

    @abstractmethod  # pragma: no cover
    def write(self, newdata: NumpyArray) -> None:
        """Write new data to the internal array.

        Parameters
        ----------
        newdata
            A new array to save into the internal array
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_valid_entries(self) -> NumpyArray:
        """Get a part of the internal array that is written so far."""
        pass

    @abstractmethod  # pragma: no cover
    def to_disk(self) -> None:
        """Save the entire array to the disk."""
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def from_disk(filepath: str | PathLike) -> ArrayRecorder:
        """Load the entire array from the disk."""
        pass


class DenseArrayRecorder(ArrayRecorder):
    """Record all data in a single array that is held entirely in the memory.

    Notes
    -----
    To Record arrays along many turns,
    this class might run into memory
    limitations.
    """

    def __init__(
        self,
        filepath: str | PathLike,
        shape: int | tuple[int, ...],
        dtype: DTypeLike | None = None,
        order: Literal["C", "F"] = "C",
        overwrite: bool = True,
    ):
        # reserve full memory at init to avoid memory overflow during runtime
        self._memory = np.zeros(shape=shape, dtype=dtype, order=order)
        self._write_idx = 0

        self.filepath = filepath
        self.overwrite = overwrite
        if not self.overwrite and os.path.exists(self.filepath_array):
            warnings.warn(
                f"{self.filepath_array} already exists!",
                UserWarning,
                stacklevel=1,
            )

    @property
    def filepath_array(self) -> str:
        """Path of the file that holds the numpy-array."""
        return f"{self.filepath}.npy"

    @property
    def filepath_attributes(self) -> str:
        """Path of the file that holds the properties."""
        return f"{self.filepath}.json"

    def purge_from_disk(self, verbose: bool = True):
        """Delete the saved array from the disk."""
        if os.path.exists(self.filepath_array):
            os.remove(self.filepath_array)
            if verbose:
                print(f"Removed {self.filepath_array}")
        if os.path.exists(self.filepath_attributes):
            os.remove(self.filepath_attributes)
            if verbose:
                print(f"Removed {self.filepath_attributes}")

    def to_disk(self):
        """Save the entire array from the disk."""
        if not self.overwrite:
            assert not os.path.exists(self.filepath_array)
        np.save(self.filepath_array, self._memory)
        attributes = {
            "_write_idx": self._write_idx,
            "overwrite": self.overwrite,
        }
        with open(self.filepath_attributes, "w") as f:
            json.dump(attributes, f)

    @staticmethod
    def from_disk(filepath: str | PathLike) -> DenseArrayRecorder:
        """Load the entire array from the disk."""
        dense_recorder = DenseArrayRecorder(
            filepath=filepath,
            shape=(1, 1),
        )
        assert isfile(dense_recorder.filepath_array)
        _memory: NumpyArray = np.load(dense_recorder.filepath_array)
        dense_recorder._memory = _memory
        with open(dense_recorder.filepath_attributes) as f:
            loaded_data = json.load(f)
        dense_recorder._write_idx = loaded_data["_write_idx"]
        dense_recorder.overwrite = loaded_data["overwrite"]
        return dense_recorder

    def write(self, newdata: NumpyArray | CupyArray | float):
        """Write new data to the internal array.

        Parameters
        ----------
        newdata
            An new array to save into the internal array
        """
        if is_cupy_array(newdata):
            newdata = newdata.get()  # type: ignore
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self) -> NumpyArray:
        """Get a part of the internal array that is written so far."""
        return self._memory[: self._write_idx]
