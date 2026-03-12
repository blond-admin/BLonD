# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Classes that deal with memory management of simulation results."""

from __future__ import annotations

import json
import os.path
import warnings
from abc import ABC, abstractmethod
from os.path import isfile
from typing import TYPE_CHECKING

import numpy as np

from blond.generals.cupy.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Literal

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy import ndarray as NumpyArray
    from numpy.typing import DTypeLike


class ArrayRecorder(ABC):
    """Base class to save content to an array."""

    @abstractmethod  # pragma: no cover
    def write(self, newdata: NumpyArray) -> None:
        """
        Write new data to the internal array.

        Parameters
        ----------
        newdata
            A new array to save into the internal array.
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_valid_entries(self) -> NumpyArray:
        """
        Get a part of the internal array that is written so far.

        Returns
        -------
        valid_entries
            The portion of the array that contains valid data.
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_disk(self) -> None:
        """Save the entire array to the disk."""
        pass

    @staticmethod
    @abstractmethod  # pragma: no cover
    def from_disk(filepath: str | PathLike) -> ArrayRecorder:
        """
        Load the entire array from the disk.

        Parameters
        ----------
        filepath
            Path to the file to load.

        Returns
        -------
        recorder
            Loaded array recorder instance.
        """
        pass


class DenseArrayRecorder(ArrayRecorder):
    """
    Record all data in a single array that is held entirely in the memory.

    Parameters
    ----------
    filepath
        Path for saving the array.
    shape
        Shape of the array to allocate.
    dtype
        Data type of the array.
    order
        Memory layout order ('C' or 'F').
    overwrite
        Whether to overwrite existing files.
    preallocate
        Flag to force memory preallocation to ensure early failure if
        too much data is requested.

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
        preallocate: bool = True,
    ):
        # Declare expected size of data in advance use zeros for safety,
        # less weird results in case of partial data
        self._memory = np.zeros(shape=shape, dtype=dtype, order=order)
        if preallocate:
            # Optionally, force full allocation to detect memory
            # overflow early
            self._memory *= 0
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
        """
        Path of the file that holds the numpy-array.

        Returns
        -------
        path
            Path to the numpy array file.
        """
        return f"{self.filepath}.npy"

    @property
    def filepath_attributes(self) -> str:
        """
        Path of the file that holds the properties.

        Returns
        -------
        path
            Path to the attributes JSON file.
        """
        return f"{self.filepath}.json"

    def purge_from_disk(self, verbose: bool = True):
        """
        Delete the saved array from the disk.

        Parameters
        ----------
        verbose
            Whether to print removal messages.
        """
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
        """
        Load the entire array from the disk.

        Parameters
        ----------
        filepath
            Path to the file to load.

        Returns
        -------
        recorder
            Loaded DenseArrayRecorder instance.
        """
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
        """
        Write new data to the internal array.

        Parameters
        ----------
        newdata
            A new array to save into the internal array.
        """
        if is_cupy_array(newdata):
            newdata = newdata.get()  # type: ignore
        self._memory[self._write_idx] = newdata
        self._write_idx += 1

    def get_valid_entries(self) -> NumpyArray:
        """
        Get a part of the internal array that is written so far.

        Returns
        -------
        valid_entries
            The portion of the array that contains valid data.
        """
        return self._memory[: self._write_idx]
