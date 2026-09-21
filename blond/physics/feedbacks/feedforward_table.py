# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Whole-run feedforward tables for a cavity feedback.

A feedforward is a programme, not a loop: it is known before the cell it
acts on is tracked, and nothing in this package computes one. A
:class:`FeedforwardTable` is only the *receiving end* -- whoever owns the
prediction (a model inversion, a shot-to-shot learner) builds the table
and hands it to the feedback, which reads it on its own cell clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


@dataclass(frozen=True, eq=False)
class FeedforwardTable:
    """
    A piecewise-constant complex programme on a feedback's cell clock.

    The cell clock is the feedback's free-running count of tracked coarse
    cells: it runs across spans, backfill replays, passages and turns and
    is never re-phased, so one table covers a whole run and an entry means
    the same cell on every run with the same grid.

    A generator loop forms its command on a controller sample and holds it
    in between, so a table is only ever *read* on controller samples. With
    ``cells_per_entry`` equal to the feedback's
    ``controller_update_interval`` and ``first_cell`` a multiple of it,
    entry ``k`` is exactly the ``k``-th controller update, and a finer
    table would carry entries nothing reads.

    Outside its cells the table is an exact zero: unlike a gain, a
    feedforward that outlives its prediction must stop acting rather than
    coast on its last entry.

    Raises
    ------
    ValueError
        If the table is empty or not one-dimensional, or if
        ``cells_per_entry`` is not positive.
    """

    values: NumpyArray
    """The programme, in cell order (complex128, read-only copy). Its unit
    and IQ frame are those of the quantity it is added to."""
    cells_per_entry: int
    """Coarse cells each entry spans."""
    first_cell: int = 0
    """Cell clock value the first entry starts at."""

    def __post_init__(self) -> None:
        """Freeze the values into a private array and validate the span."""
        values = np.array(self.values, dtype=np.complex128)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "a feedforward table needs a non-empty one-dimensional "
                f"array of values, got shape {values.shape}"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cells_per_entry", int(self.cells_per_entry))
        object.__setattr__(self, "first_cell", int(self.first_cell))
        if self.cells_per_entry < 1:
            raise ValueError(
                f"cells_per_entry={self.cells_per_entry} must be >= 1"
            )

    @property
    def n_entries(self) -> int:
        """
        Count the entries of the table.

        Returns
        -------
        n_entries
            Length of :attr:`values`.
        """
        return int(self.values.size)

    @property
    def n_cells(self) -> int:
        """
        Count the cells the table covers.

        Returns
        -------
        n_cells
            ``n_entries * cells_per_entry``.
        """
        return self.n_entries * self.cells_per_entry

    def value_at(self, cell: int) -> complex:
        """
        The programme at one cell.

        Parameters
        ----------
        cell
            Cell clock value.

        Returns
        -------
        value
            The entry covering ``cell``, or an exact zero outside the
            table.
        """
        index = (int(cell) - self.first_cell) // self.cells_per_entry
        if 0 <= index < self.n_entries:
            return complex(self.values[index])
        return 0.0 + 0.0j

    def over_cells(self, first_cell: int, n_cells: int) -> NumpyArray:
        """
        The programme expanded over a run of consecutive cells.

        Parameters
        ----------
        first_cell
            Cell clock value of the first cell of the run.
        n_cells
            Number of cells in the run.

        Returns
        -------
        values
            One complex128 value per cell, :meth:`value_at` of each.
        """
        index = (
            int(first_cell) - self.first_cell + np.arange(int(n_cells))
        ) // self.cells_per_entry
        covered = (index >= 0) & (index < self.n_entries)
        expanded = np.zeros(int(n_cells), dtype=np.complex128)
        expanded[covered] = self.values[index[covered]]
        return expanded
