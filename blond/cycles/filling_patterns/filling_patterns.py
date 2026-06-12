# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
RF filling pattern construction for accelerator physics simulations.

**Glossary**

bucket
    One RF period; the smallest unit of position. Every integer position
    and gap in this module counts RF buckets.
slot
    Machine-specific grouping of buckets (e.g. one LHC slot = 10 buckets of
    the 400 MHz RF). Not a core concept here: derive it per bunch as
    ``bucket_indices // buckets_per_slot``, or store it as a label.
bunch
    One filled bucket.
batch
    Bunches injected together from the upstream machine, equally spaced.
train
    A repeated group of batches.
gap
    Empty buckets between units. Gaps always count the number of *empty
    buckets in between*, never start-to-start.
abort gap
    The empty buckets between the last bunch and the end of the ring.
filling pattern
    The complete, ring-wide arrangement: exactly ``harmonic_number``
    buckets (:class:`FillingPattern`).

Laboratories disagree on the words for batch/train ("PS batch",
"SPS train", "injection", "pulse", ...). The label names ``"batch"`` and
``"train"`` are only convenient defaults — any grouping can be stored under
any name with :meth:`PatternSegment.with_label`, at any nesting depth.

**Conventions**

* "gap" always counts empty RF buckets between units, as an integer
  (``bunch_gap``, ``copy_gap``, :class:`Gap`,
  :meth:`PatternSegment.with_trailing_gap`)
  — never start-to-start. Beware: the LHC "25 ns bunch spacing" is
  ``bunch_gap=9`` on the 400 MHz RF, not 10.
* "spacing" always means a physical start-to-start distance in seconds
  (the ``from_spacing`` constructors).

**Composition**

Segments compose with ``+`` (concatenate), ``*`` (repeat) and ``.with_trailing_gap(n)``.
Label indices are re-numbered automatically on concatenation::

    batch     = Batch(n_bunches=72, bunch_gap=9)
    train     = Train(unit=batch, n_copies=4, copy_gap=8)
    injection = train.with_label("injection")
    pattern   = FillingPattern(injection.with_trailing_gap(38) * 11 + injection,
                               harmonic_number=35640)

    pattern.intensity = np.full(pattern.n_bunches, 1.1e11)
    pattern.intensity[pattern.label("injection") == 0] = 1.0e11

**Per-bunch quantities (consumer interface)**

Consumers (beam preparation, profiles, injection) read the
:class:`BunchTable` interface of a finished :class:`FillingPattern`:
``bucket_indices`` (sorted RF bucket index per bunch), ``harmonic_number``,
``has_bunch``, the label columns, and the per-bunch quantity arrays.
Conventional quantity names and units, shared by all consumers:

* ``intensity`` — particles per bunch (bunch population)
* ``bunch_length`` — seconds (bunch length, e.g. 4 sigma)
* ``emittance`` — eVs (longitudinal emittance)

Quantity arrays are stored as float64; NaN entries mean "unspecified"
(quantity arrays are NaN-filled when patterns defining different quantity
names are concatenated). Values that do not cast to float are rejected.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

_UNASSIGNED: int = -1


def n_buckets_from_time(
    time_distance: float,
    f_rf: float,
    tolerance: float = 0.005,
    stacklevel: int = 2,
) -> int:
    """
    Return the number of RF buckets matching a physical time distance.

    Rounds to the nearest integer number of buckets and warns when
    ``time_distance`` deviates from that integer by more than ``tolerance``
    as a fraction of the distance (default 0.5 %). The tolerance is
    relative because nominal spacings quoted in round nanoseconds miss the
    exact bucket multiple by a fixed *fraction* (e.g. 25 ns on the LHC
    400 MHz RF is 10.02 buckets, ~0.2 % per bucket), so the absolute
    deviation grows linearly with distance — 25 ns and its multiples
    (75 ns, 225 ns, ...) all pass silently.

    Parameters
    ----------
    time_distance
        Physical start-to-start distance in seconds.
    f_rf
        RF frequency in Hz.
    tolerance
        Maximum accepted deviation from an integer number of buckets,
        as a fraction of the distance (relative tolerance).
    stacklevel
        Passed to :func:`warnings.warn`; raise it when calling through
        wrappers so the warning points at the user's code.

    Returns
    -------
    n_buckets
        Number of RF buckets, rounded to the nearest integer.
    """
    n_buckets_exact = time_distance * f_rf
    n_buckets = round(n_buckets_exact)
    # max(..., 1.0) keeps sub-bucket distances from being judged against
    # a near-zero scale.
    if abs(n_buckets_exact - n_buckets) > tolerance * max(
        abs(n_buckets_exact), 1.0
    ):
        warnings.warn(
            f"time_distance = {time_distance} s corresponds to "
            f"{n_buckets_exact:.4f} RF buckets, which is not an integer "
            f"number of buckets (rounded to {n_buckets}).",
            stacklevel=stacklevel,
        )
    return n_buckets


# --------------------------------------------------------------- helpers


def _as_int(value: Any, name: str) -> int:
    # Bucket counts are integers; accept integral floats (e.g. 5.0) but
    # reject fractional values instead of silently truncating.
    as_int = int(value)
    if as_int != value:
        raise ValueError(f"{name} must be an integer, got {value}.")
    return as_int


def _next_index(label_indices: np.ndarray) -> int:
    # First unused group index of a label column (-1 = unassigned).
    assigned = label_indices[label_indices >= 0]
    return int(assigned.max()) + 1 if len(assigned) else 0


def _renumber(label_indices: np.ndarray, index_offset: int) -> np.ndarray:
    # Shift assigned label indices; unassigned entries stay -1.
    return np.where(
        label_indices >= 0, label_indices + index_offset, _UNASSIGNED
    ).astype(np.int32)


def _unassigned_label(n_bunches: int) -> np.ndarray:
    return np.full(n_bunches, _UNASSIGNED, dtype=np.int32)


def _nan_column(n_bunches: int) -> np.ndarray:
    return np.full(n_bunches, np.nan)


def _as_quantity_column(value: Any, name: str, n_bunches: int) -> np.ndarray:
    # Quantity arrays are stored as float64 (owned copy) so that NaN can
    # mark unspecified entries when segments with different quantity names
    # are concatenated.
    try:
        column = np.array(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"Quantity '{name}' must be castable to float (NaN marks "
            f"unspecified entries): {error}"
        ) from None
    if column.ndim != 1 or len(column) != n_bunches:
        raise ValueError(
            f"Quantity '{name}' must be 1-D with length {n_bunches}; "
            f"got shape {column.shape}."
        )
    return column


def _is_structural_name(cls: type, name: str) -> bool:
    # Quantity arrays travel from segments into the final FillingPattern,
    # so FillingPattern's structural names (harmonic_number, has_bunch)
    # are reserved on every BunchTable, not only where they are defined.
    return hasattr(cls, name) or hasattr(FillingPattern, name)


def _merge_columns(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    n_left: int,
    n_right: int,
    missing_column: Callable[[int], np.ndarray],
    renumber: bool = False,
) -> dict[str, np.ndarray]:
    # Concatenate per-bunch columns of two segments; names absent on one
    # side are filled via missing_column(n). With renumber=True (labels)
    # the right side is shifted so group indices stay unique.
    merged = {}
    for name in set(left) | set(right):
        left_column = left.get(name, missing_column(n_left))
        right_column = right.get(name, missing_column(n_right))
        if renumber:
            right_column = _renumber(right_column, _next_index(left_column))
        merged[name] = np.concatenate([left_column, right_column])
    return merged


def _gap_from_spacing(
    unit_n_buckets: int, start_to_start_distance: float, f_rf: float
) -> int:
    # Physical start-to-start distance (s) -> empty buckets between units.
    # stacklevel=4: user -> from_spacing -> here -> n_buckets_from_time.
    n_buckets = n_buckets_from_time(
        start_to_start_distance, f_rf, stacklevel=4
    )
    gap = n_buckets - unit_n_buckets
    if gap < 0:
        raise ValueError(
            f"start_to_start_distance = {start_to_start_distance} s spans "
            f"only {n_buckets} RF buckets, shorter than the repeated unit "
            f"itself ({unit_n_buckets} buckets); units would overlap."
        )
    return gap


def _repeat_with_gap(
    unit: PatternSegment, n_copies: int, copy_gap: int
) -> PatternSegment:
    # Repeat a unit with a gap between copies (no trailing gap).
    if n_copies == 1:
        return unit
    return unit.with_trailing_gap(copy_gap) * (n_copies - 1) + unit


# --------------------------------------------------------------- BunchTable


class BunchTable:
    """
    Read interface shared by all patterns: per-bunch arrays and quantities.

    Per-bunch arrays (all length n_bunches)::

        bucket_indices  RF bucket index of each bunch (strictly increasing)
        label(name)      membership index per bunch in the named label
                        (-1 = unassigned); 'batch' and 'train' are the
                        conventional names, any name can be added via
                        PatternSegment.with_label()

    Public attributes not starting with '_' are stored as per-bunch
    quantity arrays, enabling numpy-masked assignment::

        pattern.intensity = np.ones(pattern.n_bunches) * 1e11
        pattern.intensity[pattern.label("batch") == 2] = 0.5e11

    Names that collide with structural attributes (bucket_indices,
    n_buckets, label names, ...) are rejected.

    The structure is fixed at construction: ``bucket_indices`` and the label
    columns are read-only arrays. Quantity arrays are copied to float64 on
    assignment (NaN = unspecified) and stay mutable in place (that is the
    masked-assignment interface).

    Parameters
    ----------
    bucket_indices
        RF bucket index per bunch, strictly increasing, within
        [0, n_buckets).
    n_buckets
        Total number of RF buckets, including any trailing empty gap.
    labels
        Label columns keyed by label name (-1 = unassigned).
    quantities
        Per-bunch quantity arrays keyed by attribute name; must be
        castable to float64.
    """

    _bucket_indices: np.ndarray
    _labels: dict[str, np.ndarray]
    _n_buckets: int
    _quantities: dict[str, np.ndarray]

    def __init__(
        self,
        bucket_indices: np.ndarray,
        n_buckets: int,
        labels: dict[str, np.ndarray] | None = None,
        quantities: dict[str, np.ndarray] | None = None,
    ):
        bucket_indices = np.array(bucket_indices, dtype=np.int64)  # owned copy
        n_buckets = _as_int(n_buckets, "n_buckets")
        if bucket_indices.ndim != 1:
            raise ValueError(
                f"bucket_indices must be 1-D, got shape {bucket_indices.shape}."
            )
        if len(bucket_indices) and np.any(np.diff(bucket_indices) <= 0):
            raise ValueError(
                "bucket_indices must be strictly increasing (sorted, no duplicates)."
            )
        if len(bucket_indices) and (
            bucket_indices[0] < 0 or bucket_indices[-1] >= n_buckets
        ):
            raise ValueError(
                f"bucket_indices must lie in [0, n_buckets); got range "
                f"[{bucket_indices[0]}, {bucket_indices[-1]}] with "
                f"n_buckets {n_buckets}."
            )
        n_bunches = len(bucket_indices)
        labels = (
            {}
            if labels is None
            else {
                name: np.array(column, dtype=np.int32)
                for name, column in labels.items()
            }
        )
        quantities = (
            {}
            if quantities is None
            else {
                name: _as_quantity_column(arr, name, n_bunches)
                for name, arr in quantities.items()
            }
        )
        for name, column in labels.items():
            if len(column) != n_bunches:
                raise ValueError(
                    f"Label '{name}' has length {len(column)}, expected {n_bunches}."
                )
        for name in quantities:
            if name in labels:
                raise ValueError(
                    f"'{name}' is both a label and a quantity name; label and "
                    f"quantity names must not collide."
                )
            if _is_structural_name(type(self), name):
                raise ValueError(
                    f"Quantity '{name}' collides with a structural pattern "
                    f"attribute."
                )
        # Structure is fixed after construction; only quantity values may
        # change in place.
        bucket_indices.setflags(write=False)
        for column in labels.values():
            column.setflags(write=False)
        object.__setattr__(self, "_bucket_indices", bucket_indices)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_n_buckets", n_buckets)
        object.__setattr__(self, "_quantities", quantities)

    @property
    def n_bunches(self) -> int:
        """
        Return the number of bunches.

        Returns
        -------
        n_bunches
            Number of bunches.
        """
        return len(self._bucket_indices)

    @property
    def n_buckets(self) -> int:
        """
        Return the total number of RF buckets, including trailing gaps.

        Returns
        -------
        n_buckets
            Total number of RF buckets.
        """
        return self._n_buckets

    @property
    def bucket_indices(self) -> np.ndarray:
        """
        Return the RF bucket index of each bunch.

        Returns
        -------
        bucket_indices
            Strictly increasing array of shape (n_bunches,).
        """
        return self._bucket_indices

    @property
    def labels(self) -> dict[str, np.ndarray]:
        """
        Return all label columns.

        Returns
        -------
        labels
            Label columns, keyed by label name. The dict is a snapshot
            (adding keys does not affect the table); the columns are
            read-only.
        """
        return dict(self._labels)

    def label(self, label_name: str) -> np.ndarray:
        """
        Return the membership index per bunch in the named label.

        Parameters
        ----------
        label_name
            Name of the label (raises KeyError if unknown; use
            :meth:`PatternSegment.with_label` to add one).

        Returns
        -------
        label_column
            Membership index per bunch (-1 = unassigned).
        """
        try:
            return self._labels[label_name]
        except KeyError:
            raise KeyError(
                f"No label '{label_name}'; available labels: {sorted(self._labels)}."
            ) from None

    def n_groups(self, label_name: str) -> int:
        """
        Return the number of distinct assigned indices in the named label.

        Parameters
        ----------
        label_name
            Name of the label.

        Returns
        -------
        n_groups
            Number of groups in the label (0 if the label is absent).
        """
        column = self._labels.get(label_name)
        if column is None:
            return 0
        return int(len(np.unique(column[column >= 0])))

    @property
    def quantities(self) -> dict[str, np.ndarray]:
        """
        Return the per-bunch quantity arrays.

        Returns
        -------
        quantities
            Quantity arrays, keyed by attribute name. The dict is a
            snapshot (adding keys does not affect the table); the arrays
            are the live per-bunch arrays.
        """
        return dict(self._quantities)

    # Public attributes (no leading '_') are routed to _quantities, enabling
    # the pattern.intensity = ...; pattern.intensity[mask] = ... interface.
    # Structural names and label names are rejected to prevent silent
    # write/read mismatches (e.g. pattern.bucket_indices = ...).

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return object.__getattribute__(self, "_quantities")[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if _is_structural_name(type(self), name):
            raise AttributeError(
                f"'{name}' is a structural pattern attribute and cannot be "
                f"used as a quantity name."
            )
        if name in self._labels:
            raise AttributeError(
                f"'{name}' is a label name (read it with .label('{name}')); "
                f"quantity names must not shadow labels."
            )
        self._quantities[name] = _as_quantity_column(
            value, name, self.n_bunches
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_bunches={self.n_bunches}, n_buckets={self.n_buckets})"


# --------------------------------------------------------- PatternSegment


class PatternSegment(BunchTable):
    """
    Composable building block of a filling pattern.

    Concatenation (+) shifts bucket_indices and re-numbers every label::

        combined = a.with_trailing_gap(5) + b

    See :class:`BunchTable` for the per-bunch arrays, the quantity
    interface, and the constructor parameters.
    """

    def with_trailing_gap(self, n_empty_buckets: int) -> PatternSegment:
        """
        Return a copy of this segment with empty buckets appended.

        Parameters
        ----------
        n_empty_buckets
            Number of empty RF buckets to append.

        Returns
        -------
        segment
            New segment with the empty buckets appended.
        """
        return self + Gap(n_empty_buckets)

    def with_label(self, label_name: str) -> PatternSegment:
        """
        Return a copy with a new label in which every bunch has index 0.

        Concatenating labeled segments re-numbers the indices, so label the
        repeating unit, then repeat::

            injection = sps_train.with_label("injection")
            full = injection.with_trailing_gap(38) * 12
            full.label("injection")   # 0, ..., 0, 1, ..., 1, ..., 11

        Raises if the label already exists — e.g. nesting ``Train`` in
        ``Train`` — so inner structure is never silently overwritten; pick
        a new label name instead.

        Parameters
        ----------
        label_name
            Name of the new label (must not exist yet).

        Returns
        -------
        labeled
            Copy of this segment with the additional label.
        """
        if label_name in self._labels:
            raise ValueError(
                f"Label '{label_name}' already exists in this segment; "
                f"label it with a different name to keep the inner "
                f"'{label_name}' structure."
            )
        if label_name in self._quantities:
            raise ValueError(
                f"'{label_name}' is already a quantity name; label names must "
                f"not shadow quantities."
            )
        new_labels = dict(self._labels)
        new_labels[label_name] = np.zeros(self.n_bunches, dtype=np.int32)
        return PatternSegment(
            bucket_indices=self._bucket_indices,
            n_buckets=self._n_buckets,
            labels=new_labels,
            quantities=self._quantities,
        )

    def __add__(self, other: PatternSegment) -> PatternSegment:
        # Concatenate, re-numbering labels of the right side.
        if not isinstance(other, PatternSegment):
            raise TypeError(
                f"Can only concatenate PatternSegment, not "
                f"{type(other).__name__}; a FillingPattern is complete — "
                f"compose segments first, then wrap once."
            )
        return PatternSegment(
            bucket_indices=np.concatenate(
                [self.bucket_indices, other.bucket_indices + self.n_buckets]
            ),
            n_buckets=self.n_buckets + other.n_buckets,
            labels=_merge_columns(
                self._labels,
                other._labels,
                self.n_bunches,
                other.n_bunches,
                _unassigned_label,
                renumber=True,
            ),
            quantities=_merge_columns(
                self._quantities,
                other._quantities,
                self.n_bunches,
                other.n_bunches,
                _nan_column,
            ),
        )

    def __mul__(self, n_repetitions: int) -> PatternSegment:
        # Repeat back-to-back, re-numbering labels per copy.
        n_repetitions = _as_int(n_repetitions, "n_repetitions")
        if n_repetitions < 1:
            raise ValueError(f"Multiplier must be >= 1, got {n_repetitions}.")
        result: PatternSegment = self
        for _ in range(n_repetitions - 1):
            result = result + self
        return result

    def __rmul__(self, n_repetitions: int) -> PatternSegment:
        return self.__mul__(n_repetitions)


# ------------------------------------------------------- named segments


class Gap(PatternSegment):
    """
    Segment of empty RF buckets, used to space other segments.

    Parameters
    ----------
    n_empty_buckets
        Number of empty RF buckets (>= 0).
    """

    def __init__(self, n_empty_buckets: int):
        n_empty_buckets = _as_int(n_empty_buckets, "n_empty_buckets")
        if n_empty_buckets < 0:
            raise ValueError(
                f"n_empty_buckets must be >= 0, got {n_empty_buckets}."
            )
        super().__init__(
            bucket_indices=np.empty(0, dtype=np.int64),
            n_buckets=n_empty_buckets,
        )


class Batch(PatternSegment):
    """
    Equally spaced bunches, all labeled batch index 0 (label 'batch').

    Concatenation re-numbers batch indices automatically::

        two = Batch(n_bunches=4, bunch_gap=1).with_trailing_gap(5) + Batch(n_bunches=4, bunch_gap=1)
        two.label("batch")  # [0, 0, 0, 0,  1, 1, 1, 1]

    Parameters
    ----------
    n_bunches
        Number of bunches per batch.
    bunch_gap
        Empty RF buckets between consecutive bunches.
    """

    def __init__(self, n_bunches: int, bunch_gap: int):
        n_bunches = _as_int(n_bunches, "n_bunches")
        bunch_gap = _as_int(bunch_gap, "bunch_gap")
        if n_bunches < 1:
            raise ValueError(f"n_bunches must be >= 1, got {n_bunches}.")
        if bunch_gap < 0:
            raise ValueError(f"bunch_gap must be >= 0, got {bunch_gap}.")
        bunch_stride = 1 + bunch_gap
        super().__init__(
            bucket_indices=np.arange(n_bunches, dtype=np.int64) * bunch_stride,
            n_buckets=n_bunches + (n_bunches - 1) * bunch_gap,
            labels={"batch": np.zeros(n_bunches, dtype=np.int32)},
        )

    @classmethod
    def from_spacing(
        cls, n_bunches: int, start_to_start_distance: float, f_rf: float
    ) -> Batch:
        """
        Construct from a physical bunch start-to-start distance.

        Parameters
        ----------
        n_bunches
            Number of bunches per batch.
        start_to_start_distance
            Physical bunch start-to-start distance in seconds.
        f_rf
            RF frequency in Hz.

        Returns
        -------
        batch
            Batch with the equivalent integer bunch gap.
        """
        bunch_n_buckets = 1  # each bunch occupies exactly one RF bucket
        return cls(
            n_bunches=n_bunches,
            bunch_gap=_gap_from_spacing(
                bunch_n_buckets, start_to_start_distance, f_rf
            ),
        )


class Train(PatternSegment):
    """
    Repeated unit, all labeled train index 0 (label 'train').

    Label indices from the unit (e.g. 'batch') are preserved and re-numbered
    across copies. Concatenation re-numbers train indices::

        two = Train(batch, n_copies=3, copy_gap=5).with_trailing_gap(100) + Train(batch, n_copies=3, copy_gap=5)
        two.label("train")  # [0, 0, ...,  1, 1, ...]

    A unit that already contains a 'train' label is rejected — label deeper
    nesting levels with :meth:`PatternSegment.with_label` instead::

        super_train = (train.with_trailing_gap(20) * 3).with_label("super_train")

    Parameters
    ----------
    unit
        Segment to repeat (typically a Batch or concatenated batches).
    n_copies
        Number of repetitions.
    copy_gap
        Empty RF buckets between consecutive copies of unit.
    """

    def __init__(self, unit: PatternSegment, n_copies: int, copy_gap: int):
        n_copies = _as_int(n_copies, "n_copies")
        copy_gap = _as_int(copy_gap, "copy_gap")
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}.")
        if copy_gap < 0:
            raise ValueError(f"copy_gap must be >= 0, got {copy_gap}.")
        combined = _repeat_with_gap(unit, n_copies, copy_gap).with_label(
            "train"
        )
        super().__init__(
            bucket_indices=combined.bucket_indices,
            n_buckets=combined.n_buckets,
            labels=combined.labels,
            quantities=combined.quantities,
        )

    @classmethod
    def from_spacing(
        cls,
        unit: PatternSegment,
        n_copies: int,
        start_to_start_distance: float,
        f_rf: float,
    ) -> Train:
        """
        Construct from a physical unit start-to-start distance.

        Parameters
        ----------
        unit
            Segment to repeat.
        n_copies
            Number of repetitions.
        start_to_start_distance
            Physical unit start-to-start distance in seconds.
        f_rf
            RF frequency in Hz.

        Returns
        -------
        train
            Train with the equivalent integer copy gap.
        """
        return cls(
            unit=unit,
            n_copies=n_copies,
            copy_gap=_gap_from_spacing(
                unit.n_buckets, start_to_start_distance, f_rf
            ),
        )


# --------------------------------------------------------- FillingPattern


class FillingPattern(BunchTable):
    """
    Complete ring filling pattern of exactly harmonic_number RF buckets.

    Wraps any PatternSegment; remaining buckets form the abort gap. A
    FillingPattern is finished — unlike :class:`PatternSegment` it cannot
    be concatenated, repeated, or extended.

    Usage::

        pattern = FillingPattern(injection.with_trailing_gap(38) * 12, harmonic_number=35640)
        pattern.intensity = np.ones(pattern.n_bunches) * 1.1e11
        pattern.intensity[pattern.label("batch") == 3] = 0.5e11
        pattern.intensity[pattern.label("injection") == 1] = 0.8e11

    Parameters
    ----------
    segment
        Pattern to wrap; must not be longer than harmonic_number.
    harmonic_number
        Total number of RF buckets in the ring.
    """

    _harmonic_number: int

    def __init__(self, segment: PatternSegment, harmonic_number: int):
        harmonic_number = _as_int(harmonic_number, "harmonic_number")
        if harmonic_number < 1:
            raise ValueError(
                f"harmonic_number must be >= 1, got {harmonic_number}."
            )
        if segment.n_buckets > harmonic_number:
            raise ValueError(
                f"Segment spans {segment.n_buckets} buckets, more than "
                f"harmonic_number ({harmonic_number})."
            )
        object.__setattr__(self, "_harmonic_number", harmonic_number)
        super().__init__(
            bucket_indices=segment.bucket_indices,
            n_buckets=harmonic_number,
            labels=segment.labels,
            quantities=segment.quantities,
        )

    @property
    def harmonic_number(self) -> int:
        """
        Return the total number of RF buckets in the ring.

        Returns
        -------
        harmonic_number
            Total number of RF buckets.
        """
        return self._harmonic_number

    @property
    def has_bunch(self) -> np.ndarray:
        """
        Return the occupancy of every RF bucket in the ring.

        Returns
        -------
        has_bunch
            Bool array of length harmonic_number; True where filled.
        """
        occupied_buckets = np.zeros(self.harmonic_number, dtype=bool)
        occupied_buckets[self.bucket_indices] = True
        return occupied_buckets

    def __repr__(self) -> str:
        return (
            f"FillingPattern(harmonic_number={self.harmonic_number}, "
            f"n_bunches={self.n_bunches})"
        )

    @classmethod
    def from_placements(
        cls,
        harmonic_number: int,
        placements: list[tuple[PatternSegment, int]],
    ) -> FillingPattern:
        """
        Construct from explicitly positioned segments.

        Label and quantity arrays of the placed segments are preserved and
        merged (labels re-numbered in position order).

        Parameters
        ----------
        harmonic_number
            Total RF buckets; unoccupied buckets become the abort gap.
        placements
            (segment, start_bucket) pairs with start_bucket >= 0. Segments
            must occupy disjoint bucket ranges (a segment's range includes
            its trailing gap; no interleaving).

        Returns
        -------
        pattern
            Complete filling pattern.
        """
        if not placements:
            return cls(Gap(0), harmonic_number)

        ordered = sorted(placements, key=lambda placement: placement[1])
        combined: PatternSegment = Gap(0)
        previous_end = 0
        for segment, raw_start in ordered:
            start_bucket = _as_int(raw_start, "start_bucket")
            if start_bucket < 0:
                raise ValueError(
                    f"start_bucket must be >= 0, got {start_bucket}."
                )
            if start_bucket < previous_end:
                raise ValueError(
                    f"Segment placed at bucket {start_bucket} overlaps the "
                    f"bucket range of the previous segment, which extends "
                    f"to bucket {previous_end - 1} (trailing gaps included)."
                )
            combined = combined + Gap(start_bucket - previous_end) + segment
            previous_end = start_bucket + segment.n_buckets
        return cls(combined, harmonic_number)
