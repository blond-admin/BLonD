# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for the implementation of buffers for local rf feedback systems.

Notes
-----
Authors:
Birk Emil Karlsen-Baeck
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class TwoTurnArray:
    """
    Wrapper for a NumPy Array of dimension (2, N) array representing [previous turn, current turn].

    The class is intended to be used with local feedback systems.
    Indexing with a non-negative int/slice reads from CURR, as normal.
    Indexing with a negative int/slice transparently reaches back into
    PREV, as if CURR were conceptually preceded by PREV — no full
    concatenation needed for single-sample access.

    Parameters
    ----------
    n_samples
        Number of samples per turn.
    dtype
        The data-type stored in the array.
    """

    __slots__ = ("_data",)

    def __init__(self, n_samples: int, dtype=np.float64):
        self._data = np.zeros((2, n_samples), dtype=dtype)

    @classmethod
    def from_block_row(cls, block: NumpyArray, row: int) -> TwoTurnArray:
        """
        Wrap one row of a shared block instead of owning an array.

        The row must hold the previous turn followed by the current turn,
        so a block of shape ``(n_signals, 2 * n_samples)`` gives one
        buffer per row. The buffer is a *view*: writes through it and
        writes through the block are the same writes, which is what lets a
        compiled kernel take the whole block as a single argument.

        Parameters
        ----------
        block
            Contiguous array of shape ``(n_signals, 2 * n_samples)``.
        row
            Index of the row to wrap.

        Returns
        -------
        buffer
            Buffer backed by ``block[row]``.
        """
        buffer = cls.__new__(cls)
        buffer._data = block[row].reshape(2, -1)
        return buffer

    @property
    def n_samples(self) -> int:
        """
        Number of samples per turn.

        Returns
        -------
        n_samples
            Number of samples of each turn.
        """
        return self._data.shape[1]

    @property
    def prev(self) -> NumpyArray:
        """
        The array of the previous turn.

        Returns
        -------
        prev
            The array of values from the previous turn.
        """
        return self._data[0]

    @prev.setter
    def prev(self, array: NumpyArray):
        """
        Set the values of the previous-turn array.

        Parameters
        ----------
        array
            Array of values to set the previous-turn array with.
        """
        self._data[0] = array

    @property
    def curr(self) -> NumpyArray:
        """
        The array of the current turn.

        Returns
        -------
        prev
            The array of values from the current turn.
        """
        return self._data[1]

    @curr.setter
    def curr(self, array: NumpyArray):
        """
        Set the values of the current-turn array.

        Parameters
        ----------
        array
            Array of values to set the current-turn array with.
        """
        self._data[1] = array

    def shift(self) -> None:
        """Shift the current turn into the previous."""
        self._data[0] = self._data[1]

    @property
    def two_turn_view(self) -> NumpyArray:
        """
        Flat view spanning the previous and current turn.

        Unlike :attr:`full`, this is a view and not a copy, so writes
        through it reach the buffer. Index ``n_samples + i`` addresses
        sample ``i`` of the current turn, which makes the reach-back
        semantics of :meth:`__getitem__` plain arithmetic: a negative
        offset walks into the previous turn.

        Returns
        -------
        two_turn_view
            View of length ``2 * n_samples`` over previous and current turn.
        """
        return self._data.reshape(-1)

    @property
    def full(self) -> NumpyArray:
        """
        Flat array spanning the previous and current turn.

        Returns
        -------
        full
            Array spanning the previous and current turn.
        """
        return np.concatenate(self._data)

    def __getitem__(self, key: int | np.integer | slice):
        """
        Get elements from the current turn.

        Negative values for the key correspond to values
        indices in the previous turn.

        Parameters
        ----------
        key
            The key for obtaining the values on the array.

        Returns
        -------
        values
            The values corresponding to the key.
        """
        n = self.n_samples
        if isinstance(key, (int, np.integer)):
            if key >= 0:
                return self.curr[key]
            idx = n + key
            if idx < 0:
                raise IndexError(
                    f"index {key} reaches back further than one turn of history"
                )
            return self.prev[idx]

        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = n if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            if start >= 0 and stop >= 0:
                return self.curr[start:stop:step]
            # boundary-crossing or fully negative slice: only concatenate
            # the (small) region actually needed
            concat = np.concatenate(self._data)
            lo = n + start
            hi = n + stop
            return concat[lo:hi:step]

        raise TypeError(f"unsupported index type: {type(key)}")

    def get_window(self, ind: int, n_taps: int) -> NumpyArray:
        """
        Get the samples relevant for a FIR filter.

        Parameters
        ----------
        ind
            The last index of to apply the filter from.
        n_taps
            The number of taps of the FIR filter.

        Returns
        -------
        result
            Array of values from ind - n_taps + 1 up to ind.
        """
        lo = ind - n_taps + 1
        if lo >= 0:
            return self.curr[lo : ind + 1]
        return np.concatenate((self.prev[lo:], self.curr[: ind + 1]))

    def __setitem__(self, key, value) -> None:
        """
        Set elements in the two-turn array.

        Parameters
        ----------
        key
            The kay of the elements you want to change.
        value
            The new values of the elements corresponding to the key.
        """
        if isinstance(key, (int, np.integer)) and key < 0:
            raise IndexError("cannot write into previous-turn history")
        self.curr[key] = value

    def __len__(self) -> int:
        """
        The length of the turns.

        Returns
        -------
        n_samples
            The length of each turn in number of samples.
        """
        return self.n_samples

    def __repr__(self) -> str:
        """
        Printable representation of the TwoTurnArray.

        Returns
        -------
        info_string
            String showing previous and current turn elements.
        """
        return f"TwoTurnArray(prev={self.prev!r}, curr={self.curr!r})"


@dataclass
class BufferBase(ABC):
    """
    Base class for the buffer container used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    # Base parameters
    samples_per_turn: int

    # Base buffers needed for any CavityFeedback class
    v_setpoint: NumpyArray | TwoTurnArray = field(init=False)
    v_ant: NumpyArray | TwoTurnArray = field(init=False)
    i_beam: NumpyArray | TwoTurnArray = field(init=False)
    i_gen: NumpyArray | TwoTurnArray = field(init=False)

    def __post_init__(self):
        """Initialize the buffers."""
        self.v_setpoint = self._make_array(dtype=complex)
        self.v_ant = self._make_array(dtype=complex)
        self.i_beam = self._make_array(dtype=complex)
        self.i_gen = self._make_array(dtype=complex)

    @abstractmethod  # pragma: no cover
    def _make_array(self, dtype) -> NumpyArray | TwoTurnArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.
        """
        raise NotImplementedError

    def shift(self):
        """Roll every two-turn array: curr -> prev, ready for a new curr."""
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, TwoTurnArray):
                val.shift()


@dataclass
class OneTurnBufferBase(BufferBase):
    """
    Base class for buffers spanning a single turn used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    def _make_array(self, dtype) -> NumpyArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.

        Returns
        -------
        array
            An array object initialized with the correct number of samples
            and data type.

        Notes
        -----
        These arrays will span a single turn only.
        """
        return np.zeros(self.samples_per_turn, dtype=dtype)


@dataclass
class TwoTurnBufferBase(BufferBase):
    """
    Base class for buffers spanning two turns used for the LocalFeedbacks.

    This class is intended to be used to store the buffers of a certain
    time-resolution inside the local feedbacks.

    Attributes
    ----------
    v_setpoint
        The setpoint voltage [V] buffer of the feedback system.
    v_ant
        Buffer containing the RF voltage measured by the antenna [V].
    i_beam
        Buffer containing the RF component of the beam current [A].
    i_gen
        Buffer containing the forward current [A] of the generator.
    """

    def _make_array(self, dtype) -> TwoTurnArray:
        """
        Return the array for a buffer with a certain data-type.

        Parameters
        ----------
        dtype
            The data-type the buffer should store.

        Returns
        -------
        array
            An array object initialized with the correct number of samples
            and data type.

        Notes
        -----
        These arrays will span two turns.
        """
        return TwoTurnArray(self.samples_per_turn, dtype=dtype)
