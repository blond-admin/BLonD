# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Classes that deal with memory management of simulation results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond.generals.cupy_.no_cupy_import import is_cupy_array
from blond.handle_results.hdf5_io import ATTR_WRITE_IDX

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Literal

    import h5py
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray as NumpyArray


class ArrayRecorder(ABC):
    """Base class to save content to an array."""

    @abstractmethod  # pragma: no cover
    def write(self, newdata: NumpyArray, mask: NumpyArray | None) -> None:
        """
        Write new data to the internal array.

        Parameters
        ----------
        newdata
            A new array to save into the internal array.
        mask
            Boolean mask array that handles where to write.
            This is at the moment only needed to handle beams with losses.
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
    def to_group(self, group: h5py.Group, name: str) -> None:
        """
        Write the internal array into an HDF5 group.

        Parameters
        ----------
        group
            Open HDF5 group that receives the dataset.
        name
            Name of the dataset inside the group.
        """
        pass

    @classmethod
    @abstractmethod  # pragma: no cover
    def from_payload(
        cls,
        array: NumpyArray,
        attrs: dict[str, Any],
    ) -> ArrayRecorder:
        """
        Rebuild a recorder from a migrated HDF5 payload.

        Parameters
        ----------
        array
            Array as read from the results file.
        attrs
            Dataset attributes as read from the results file.

        Returns
        -------
        recorder
            Recorder holding the loaded array.
        """
        pass


class DenseArrayRecorder(ArrayRecorder):
    """
    Record all data in a single array that is held entirely in the memory.

    Parameters
    ----------
    shape
        Shape of the array to allocate.
    dtype
        Data type of the array.
    order
        Memory layout order ('C' or 'F').
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
        shape: int | tuple[int, ...],
        dtype: DTypeLike | None = None,
        order: Literal["C", "F"] = "C",
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

    def to_group(self, group: h5py.Group, name: str) -> None:
        """
        Write the internal array into an HDF5 group.

        Parameters
        ----------
        group
            Open HDF5 group that receives the dataset.
        name
            Name of the dataset inside the group.
        """
        dataset = group.create_dataset(name, data=self._memory)
        dataset.attrs[ATTR_WRITE_IDX] = self._write_idx

    @classmethod
    def from_payload(
        cls,
        array: NumpyArray,
        attrs: dict[str, Any],
    ) -> DenseArrayRecorder:
        """
        Rebuild a recorder from a migrated HDF5 payload.

        Parameters
        ----------
        array
            Array as read from the results file.
        attrs
            Dataset attributes as read from the results file.

        Returns
        -------
        recorder
            Recorder holding the loaded array.
        """
        # `preallocate=False` skips touching every element of an array
        # that is discarded on the very next line.
        recorder = cls(shape=0, dtype=array.dtype, preallocate=False)
        recorder._memory = array
        recorder._write_idx = int(attrs[ATTR_WRITE_IDX])
        return recorder

    def write(
        self,
        newdata: NumpyArray | CupyArray | float,
        mask: NumpyArray | CupyArray | None = None,
    ):
        """
        Write new data to the internal array.

        Parameters
        ----------
        newdata
            A new array to save into the internal array.
        mask
            Boolean mask array that handles where to write.
            This is at the moment only needed to handle beams with losses.
            All elements that are not marked by the mask are set to `NaN`.
        """
        if is_cupy_array(newdata):
            newdata = newdata.get()  # type: ignore
        if mask is None:
            self._memory[self._write_idx] = newdata
        else:
            if is_cupy_array(mask):
                mask = mask.get()
            assert mask.dtype == np.bool, f"{mask.dtype=}"
            self._memory[self._write_idx][mask] = newdata
            self._memory[self._write_idx][~mask] = np.nan

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
