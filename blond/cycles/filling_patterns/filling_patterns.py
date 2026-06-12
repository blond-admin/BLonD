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
    and spacing in this module counts RF buckets.
slot
    Machine-specific grouping of buckets (e.g. one LHC slot = 10 buckets of
    the 400 MHz RF). Not a core concept here: derive it per bunch as
    ``positions // buckets_per_slot``, or store it as a tier.
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
"SPS train", "injection", "pulse", ...). The tier names ``"batch"`` and
``"train"`` are only convenient defaults — any grouping can be stored under
any name with :meth:`PatternSegment.label`, at any nesting depth.

**Conventions**

* Integer spacings (``bunch_spacing``, ``copy_spacing``, :class:`Gap`)
  count empty buckets between units.
* Physical times (the ``from_spacing`` constructors) are start-to-start
  distances in seconds.

**Composition**

Segments compose with ``+`` (concatenate), ``*`` (repeat) and ``.gap(n)``.
Tier indices are re-numbered automatically on concatenation::

    batch     = Batch(n_bunches=72, bunch_spacing=9)
    train     = Train(unit=batch, n_copies=4, copy_spacing=8)
    injection = train.label("injection")
    pattern   = FillingPattern(injection.gap(38) * 11 + injection,
                               harmonic_number=35640)

    pattern.intensity = np.full(pattern.n_bunches, 1.1e11)
    pattern.intensity[pattern.tier("injection") == 0] = 1.0e11

**Payload contract (consumer interface)**

A finished :class:`FillingPattern` guarantees: ``positions`` (sorted RF
bucket index per bunch), ``harmonic_number``, ``has_bunch``, the tier
columns, and the per-bunch payload arrays. Conventional payload names and
units, shared by all consumers (beam preparation, profiles, injection):

* ``intensity`` — particles per bunch (bunch population)
* ``bunch_length`` — seconds (bunch length, e.g. 4 sigma)
* ``emittance`` — eVs (longitudinal emittance)

NaN entries mean "unspecified" (payload arrays are NaN-filled when
patterns defining different payload names are concatenated).
"""

from __future__ import annotations

from typing import Any, NoReturn

import numpy as np

from blond.cycles.filling_patterns.helpers import as_n_buckets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UNASSIGNED: int = -1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _next_index(tier_indices: np.ndarray) -> int:
    """
    Return the first unused index of a tier column.

    Parameters
    ----------
    tier_indices
        Membership index per bunch (-1 = unassigned).

    Returns
    -------
    next_index
        Maximum assigned index + 1, or 0 if none is assigned.
    """
    assigned = tier_indices[tier_indices >= 0]
    return int(assigned.max()) + 1 if len(assigned) else 0


def _renumber(tier_indices: np.ndarray, index_offset: int) -> np.ndarray:
    """
    Shift assigned tier indices, leaving unassigned entries unchanged.

    Parameters
    ----------
    tier_indices
        Membership index per bunch (-1 = unassigned).
    index_offset
        Offset added to every non-negative index.

    Returns
    -------
    renumbered
        Shifted tier column (unassigned entries stay -1).
    """
    return np.where(
        tier_indices >= 0, tier_indices + index_offset, _UNASSIGNED
    ).astype(np.int32)


def _unassigned_tier(n_bunches: int) -> np.ndarray:
    """
    Return a tier column with all bunches unassigned.

    Parameters
    ----------
    n_bunches
        Number of bunches, i.e. length of the column.

    Returns
    -------
    tier_column
        Array of -1 with length n_bunches.
    """
    return np.full(n_bunches, _UNASSIGNED, dtype=np.int32)


def _merge_tiers(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    n_left_bunches: int,
    n_right_bunches: int,
) -> dict[str, np.ndarray]:
    """
    Concatenate tier columns of two segments, re-numbering the right side.

    Tier names absent from one side are filled with -1 (unassigned) for
    that side's bunches.

    Parameters
    ----------
    left
        Tier columns of the left segment.
    right
        Tier columns of the right segment.
    n_left_bunches
        Number of bunches in the left segment.
    n_right_bunches
        Number of bunches in the right segment.

    Returns
    -------
    merged
        Concatenated tier columns, keyed by tier name.
    """
    merged = {}
    for tier_name in set(left) | set(right):
        left_column = left.get(tier_name, _unassigned_tier(n_left_bunches))
        right_column = right.get(tier_name, _unassigned_tier(n_right_bunches))
        merged[tier_name] = np.concatenate(
            [
                left_column,
                _renumber(right_column, _next_index(left_column)),
            ]
        )
    return merged


def _merge_payload(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    n_left_bunches: int,
    n_right_bunches: int,
) -> dict[str, np.ndarray]:
    """
    Concatenate payload arrays, NaN-filling names absent from one side.

    Parameters
    ----------
    left
        Payload arrays of the left segment.
    right
        Payload arrays of the right segment.
    n_left_bunches
        Number of bunches in the left segment.
    n_right_bunches
        Number of bunches in the right segment.

    Returns
    -------
    merged
        Concatenated payload arrays, keyed by attribute name.
    """
    return {
        attribute_name: np.concatenate(
            [
                left.get(attribute_name, np.full(n_left_bunches, np.nan)),
                right.get(attribute_name, np.full(n_right_bunches, np.nan)),
            ]
        )
        for attribute_name in set(left) | set(right)
    }


def _spacing_from_distance(
    unit_length: int, start_to_start_distance: float, f_rf: float
) -> int:
    """
    Convert a physical start-to-start distance to a gap in buckets.

    Parameters
    ----------
    unit_length
        Length of the repeated unit in buckets.
    start_to_start_distance
        Physical start-to-start distance in seconds.
    f_rf
        RF frequency in Hz.

    Returns
    -------
    spacing
        Empty RF buckets between consecutive units.
    """
    return as_n_buckets(start_to_start_distance, f_rf) - unit_length


def _repeat_with_gap(
    unit: PatternSegment, n_copies: int, copy_spacing: int
) -> PatternSegment:
    """
    Repeat a unit with a gap between copies (no trailing gap).

    Parameters
    ----------
    unit
        Segment to repeat.
    n_copies
        Number of repetitions.
    copy_spacing
        Empty RF buckets between consecutive copies.

    Returns
    -------
    repeated
        Concatenation of n_copies of unit.
    """
    if n_copies == 1:
        return unit
    return unit.gap(copy_spacing) * (n_copies - 1) + unit


# ---------------------------------------------------------------------------
# Base: composable segment
# ---------------------------------------------------------------------------


class PatternSegment:
    """
    Composable building block of a filling pattern (per-bunch array table).

    Per-bunch arrays (all length n_bunches)::

        positions     RF bucket index of each bunch (strictly increasing)
        bunch         0-based ordinal index  (0, 1, 2, ...)
        tier(name)    membership index per bunch in the named tier
                      (-1 = unassigned); 'batch' and 'train' are the
                      conventional names, any name can be added via label()

    Public attributes not starting with '_' are stored as per-bunch payload
    arrays, enabling numpy-masked assignment::

        segment.intensity = np.ones(segment.n_bunches) * 1e11
        segment.intensity[segment.batch == 2] = 0.5e11

    Names that collide with structural attributes (positions, length,
    tier names, ...) are rejected.

    Concatenation (+) shifts positions and re-numbers every tier::

        combined = a.gap(5) + b

    Parameters
    ----------
    positions
        RF bucket index per bunch, strictly increasing, within [0, length).
    length
        Total number of RF buckets, including any trailing empty gap.
    tiers
        Tier columns keyed by tier name (-1 = unassigned).
    payload
        Per-bunch payload arrays keyed by attribute name.
    """

    _positions: np.ndarray
    _tiers: dict[str, np.ndarray]
    _length: int
    _payload: dict[str, np.ndarray]

    def __init__(
        self,
        positions: np.ndarray,
        length: int,
        tiers: dict[str, np.ndarray] | None = None,
        payload: dict[str, np.ndarray] | None = None,
    ):
        positions = np.asarray(positions, dtype=np.int64)
        length = int(length)
        if positions.ndim != 1:
            raise ValueError(
                f"positions must be 1-D, got shape {positions.shape}."
            )
        if len(positions) and np.any(np.diff(positions) <= 0):
            raise ValueError(
                "positions must be strictly increasing (sorted, no duplicates)."
            )
        if len(positions) and (positions[0] < 0 or positions[-1] >= length):
            raise ValueError(
                f"positions must lie in [0, length); got range "
                f"[{positions[0]}, {positions[-1]}] with length {length}."
            )
        n_bunches = len(positions)
        tiers = (
            {}
            if tiers is None
            else {
                name: np.asarray(column, dtype=np.int32)
                for name, column in tiers.items()
            }
        )
        payload = {} if payload is None else dict(payload)
        for name, column in tiers.items():
            if len(column) != n_bunches:
                raise ValueError(
                    f"Tier '{name}' has length {len(column)}, expected {n_bunches}."
                )
        for name, arr in payload.items():
            if np.ndim(arr) != 1 or len(arr) != n_bunches:
                raise ValueError(
                    f"Payload '{name}' must be 1-D with length {n_bunches}."
                )
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_tiers", tiers)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_payload", payload)

    # ------------------------------------------------------------------ counts

    @property
    def n_bunches(self) -> int:
        """
        Return the number of bunches in this segment.

        Returns
        -------
        n_bunches
            Number of bunches.
        """
        return len(self._positions)

    @property
    def n_batches(self) -> int:
        """
        Return the number of distinct assigned indices in the 'batch' tier.

        Returns
        -------
        n_batches
            Number of batches (0 if the tier is absent).
        """
        return self.n_in_tier("batch")

    @property
    def n_trains(self) -> int:
        """
        Return the number of distinct assigned indices in the 'train' tier.

        Returns
        -------
        n_trains
            Number of trains (0 if the tier is absent).
        """
        return self.n_in_tier("train")

    def n_in_tier(self, tier_name: str) -> int:
        """
        Return the number of distinct assigned indices in the named tier.

        Parameters
        ----------
        tier_name
            Name of the tier.

        Returns
        -------
        n_in_tier
            Number of groups in the tier (0 if the tier is absent).
        """
        return _next_index(self._tiers.get(tier_name, _unassigned_tier(0)))

    # ------------------------------------------------------------------ labels

    @property
    def length(self) -> int:
        """
        Return the total number of RF buckets, including trailing gaps.

        Returns
        -------
        length
            Total number of RF buckets.
        """
        return self._length

    @property
    def positions(self) -> np.ndarray:
        """
        Return the RF bucket index of each bunch.

        Returns
        -------
        positions
            Strictly increasing array of shape (n_bunches,).
        """
        return self._positions

    @property
    def bunch(self) -> np.ndarray:
        """
        Return the 0-based ordinal index per bunch.

        Enables masks such as ``bunch % 2 == 0``.

        Returns
        -------
        bunch
            Array [0, 1, 2, ...] of shape (n_bunches,).
        """
        return np.arange(self.n_bunches, dtype=np.int32)

    @property
    def batch(self) -> np.ndarray:
        """
        Return the 'batch' tier column.

        Returns
        -------
        batch
            Batch index per bunch (-1 where unassigned or tier absent).
        """
        return self._tiers.get("batch", _unassigned_tier(self.n_bunches))

    @property
    def train(self) -> np.ndarray:
        """
        Return the 'train' tier column.

        Returns
        -------
        train
            Train index per bunch (-1 where unassigned or tier absent).
        """
        return self._tiers.get("train", _unassigned_tier(self.n_bunches))

    @property
    def tiers(self) -> dict[str, np.ndarray]:
        """
        Return all tier columns.

        Returns
        -------
        tiers
            Tier columns, keyed by tier name.
        """
        return self._tiers

    def tier(self, tier_name: str) -> np.ndarray:
        """
        Return the membership index per bunch in the named tier.

        Parameters
        ----------
        tier_name
            Name of the tier (raises KeyError if unknown; use
            :meth:`label` to add one, or the ``batch``/``train``
            properties which default to -1).

        Returns
        -------
        tier_column
            Membership index per bunch (-1 = unassigned).
        """
        try:
            return self._tiers[tier_name]
        except KeyError:
            raise KeyError(
                f"No tier '{tier_name}'; available tiers: {sorted(self._tiers)}."
            ) from None

    @property
    def payload(self) -> dict[str, np.ndarray]:
        """
        Return the per-bunch payload arrays.

        Returns
        -------
        payload
            Payload arrays, keyed by attribute name.
        """
        return self._payload

    # ----------------------------------------------------------- payload i/o
    # Public attributes (no leading '_') are routed to _payload, enabling
    # the pattern.intensity = ...; pattern.intensity[mask] = ... interface.
    # Structural names and tier names are rejected to prevent silent
    # write/read mismatches (e.g. pattern.positions = ...).

    def __getattr__(self, name: str) -> np.ndarray:
        """
        Return the payload array stored under the given name.

        Parameters
        ----------
        name
            Payload attribute name.

        Returns
        -------
        payload_array
            Per-bunch payload array.
        """
        try:
            return object.__getattribute__(self, "_payload")[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Store a per-bunch payload array under the given name.

        Parameters
        ----------
        name
            Payload attribute name (must not collide with structural
            attributes or tier names).
        value
            1-D array of length n_bunches.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if hasattr(type(self), name):
            raise AttributeError(
                f"'{name}' is a structural attribute of {type(self).__name__} "
                f"and cannot be used as a payload name."
            )
        if name in self._tiers:
            raise AttributeError(
                f"'{name}' is a tier name (read it with .tier('{name}')); "
                f"payload names must not shadow tiers."
            )
        arr = np.asarray(value)
        if arr.ndim != 1 or len(arr) != self.n_bunches:
            raise ValueError(
                f"Payload '{name}' must be 1-D with length {self.n_bunches}; "
                f"got shape {arr.shape}."
            )
        self._payload[name] = arr

    # ------------------------------------------------------------ gap & label

    def gap(self, n_empty_buckets: int) -> PatternSegment:
        """
        Return self with empty buckets appended.

        Parameters
        ----------
        n_empty_buckets
            Number of empty RF buckets to append.

        Returns
        -------
        segment
            New segment with increased length.
        """
        return self + Gap(n_empty_buckets)

    def label(self, tier_name: str) -> PatternSegment:
        """
        Return a copy with a new tier in which every bunch has index 0.

        Concatenating labeled segments re-numbers the indices, so label the
        repeating unit, then repeat::

            injection = sps_train.label("injection")
            full = injection.gap(38) * 12
            full.tier("injection")   # 0, ..., 0, 1, ..., 1, ..., 11

        Raises if the tier already exists — e.g. nesting ``Train`` in
        ``Train`` — so inner structure is never silently overwritten; pick
        a new tier name instead.

        Parameters
        ----------
        tier_name
            Name of the new tier (must not exist yet).

        Returns
        -------
        labeled
            Copy of this segment with the additional tier.
        """
        if tier_name in self._tiers:
            raise ValueError(
                f"Tier '{tier_name}' already exists in this segment; "
                f"label it with a different name to keep the inner "
                f"'{tier_name}' structure."
            )
        if tier_name in self._payload:
            raise ValueError(
                f"'{tier_name}' is already a payload name; tier names must "
                f"not shadow payloads."
            )
        new_tiers = {
            name: column.copy() for name, column in self._tiers.items()
        }
        new_tiers[tier_name] = np.zeros(self.n_bunches, dtype=np.int32)
        return PatternSegment(
            positions=self._positions.copy(),
            length=self._length,
            tiers=new_tiers,
            payload={name: arr.copy() for name, arr in self._payload.items()},
        )

    # ----------------------------------------------------------- concatenation

    def __add__(self, other: PatternSegment) -> PatternSegment:
        """
        Concatenate two segments, re-numbering tiers of the right side.

        Parameters
        ----------
        other
            Segment appended after this one.

        Returns
        -------
        combined
            Concatenated segment.
        """
        if isinstance(other, FillingPattern):
            raise other._complete_error()
        return PatternSegment(
            positions=np.concatenate(
                [self.positions, other.positions + self.length]
            ),
            length=self.length + other.length,
            tiers=_merge_tiers(
                self._tiers, other._tiers, self.n_bunches, other.n_bunches
            ),
            payload=_merge_payload(
                self._payload, other._payload, self.n_bunches, other.n_bunches
            ),
        )

    def __mul__(self, n_repetitions: int) -> PatternSegment:
        """
        Repeat this segment back-to-back.

        Parameters
        ----------
        n_repetitions
            Number of copies (>= 1).

        Returns
        -------
        repeated
            Concatenation of n_repetitions copies.
        """
        if n_repetitions < 1:
            raise ValueError(f"Multiplier must be >= 1, got {n_repetitions}.")
        result: PatternSegment = self
        for _ in range(n_repetitions - 1):
            result = result + self
        return result

    def __rmul__(self, n_repetitions: int) -> PatternSegment:
        """
        Repeat this segment back-to-back (reflected operand).

        Parameters
        ----------
        n_repetitions
            Number of copies (>= 1).

        Returns
        -------
        repeated
            Concatenation of n_repetitions copies.
        """
        return self.__mul__(n_repetitions)

    def __len__(self) -> int:
        """
        Return the segment length in RF buckets.

        Returns
        -------
        length
            Total number of RF buckets.
        """
        return self._length

    def __repr__(self) -> str:
        """
        Return a short description of the segment.

        Returns
        -------
        representation
            Class name with bunch count and length.
        """
        return f"{type(self).__name__}(n_bunches={self.n_bunches}, length={self.length})"


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------


class Gap(PatternSegment):
    """
    Segment of empty RF buckets, used to space other segments.

    Parameters
    ----------
    n_empty_buckets
        Number of empty RF buckets (>= 0).
    """

    def __init__(self, n_empty_buckets: int):
        if n_empty_buckets < 0:
            raise ValueError(
                f"n_empty_buckets must be >= 0, got {n_empty_buckets}."
            )
        super().__init__(
            positions=np.empty(0, dtype=np.int64),
            length=n_empty_buckets,
        )


# ---------------------------------------------------------------------------
# Bunch
# ---------------------------------------------------------------------------


class Bunch(PatternSegment):
    """Single filled RF bucket, not assigned to any tier."""

    def __init__(self):
        super().__init__(
            positions=np.array([0], dtype=np.int64),
            length=1,
        )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class Batch(PatternSegment):
    """
    Equally spaced bunches, all labeled batch index 0 (tier 'batch').

    Concatenation re-numbers batch indices automatically::

        two = Batch(n_bunches=4, bunch_spacing=1).gap(5) + Batch(n_bunches=4, bunch_spacing=1)
        two.batch  # [0, 0, 0, 0,  1, 1, 1, 1]

    Parameters
    ----------
    n_bunches
        Number of bunches per batch.
    bunch_spacing
        Empty RF buckets between consecutive bunches.
    """

    def __init__(self, n_bunches: int, bunch_spacing: int):
        if n_bunches < 1:
            raise ValueError(f"n_bunches must be >= 1, got {n_bunches}.")
        if bunch_spacing < 0:
            raise ValueError(
                f"bunch_spacing must be >= 0, got {bunch_spacing}."
            )
        bunch_stride = 1 + bunch_spacing
        super().__init__(
            positions=np.arange(n_bunches, dtype=np.int64) * bunch_stride,
            length=n_bunches + (n_bunches - 1) * bunch_spacing,
            tiers={"batch": np.zeros(n_bunches, dtype=np.int32)},
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
            Batch with the equivalent integer bunch spacing.
        """
        bunch_length = 1  # each bunch occupies exactly one RF bucket
        return cls(
            n_bunches=n_bunches,
            bunch_spacing=_spacing_from_distance(
                bunch_length, start_to_start_distance, f_rf
            ),
        )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


class Train(PatternSegment):
    """
    Repeated unit, all labeled train index 0 (tier 'train').

    Tier indices from the unit (e.g. 'batch') are preserved and re-numbered
    across copies. Concatenation re-numbers train indices::

        two = Train(batch, n_copies=3, copy_spacing=5).gap(100) + Train(batch, n_copies=3, copy_spacing=5)
        two.train  # [0, 0, ...,  1, 1, ...]

    A unit that already contains a 'train' tier is rejected — label deeper
    nesting levels with :meth:`PatternSegment.label` instead::

        super_train = (train.gap(20) * 3).label("super_train")

    Parameters
    ----------
    unit
        Segment to repeat (typically a Batch or concatenated batches).
    n_copies
        Number of repetitions.
    copy_spacing
        Empty RF buckets between consecutive copies of unit.
    """

    def __init__(self, unit: PatternSegment, n_copies: int, copy_spacing: int):
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}.")
        combined = _repeat_with_gap(unit, n_copies, copy_spacing).label(
            "train"
        )
        super().__init__(
            positions=combined.positions,
            length=combined.length,
            tiers=combined.tiers,
            payload=combined.payload,
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
            Train with the equivalent integer copy spacing.
        """
        return cls(
            unit=unit,
            n_copies=n_copies,
            copy_spacing=_spacing_from_distance(
                unit.length, start_to_start_distance, f_rf
            ),
        )


# ---------------------------------------------------------------------------
# FillingPattern: the complete ring
# ---------------------------------------------------------------------------


class FillingPattern(PatternSegment):
    """
    Complete ring filling pattern of exactly harmonic_number RF buckets.

    Wraps any PatternSegment; remaining buckets form the abort gap. A
    FillingPattern is finished — it cannot be concatenated or repeated.

    Usage::

        pattern = FillingPattern(injection.gap(38) * 12, harmonic_number=35640)
        pattern.intensity = np.ones(pattern.n_bunches) * 1.1e11
        pattern.intensity[pattern.batch == 3] = 0.5e11
        pattern.intensity[pattern.tier("injection") == 1] = 0.8e11

    Parameters
    ----------
    segment
        Pattern to wrap; must not be longer than harmonic_number.
    harmonic_number
        Total number of RF buckets in the ring.
    """

    _harmonic_number: int

    def __init__(self, segment: PatternSegment, harmonic_number: int):
        if segment.length > harmonic_number:
            raise ValueError(
                f"Segment length ({segment.length} buckets) exceeds "
                f"harmonic_number ({harmonic_number})."
            )
        object.__setattr__(self, "_harmonic_number", int(harmonic_number))
        super().__init__(
            positions=segment.positions.copy(),
            length=harmonic_number,
            tiers={
                name: column.copy() for name, column in segment.tiers.items()
            },
            payload={
                name: arr.copy() for name, arr in segment.payload.items()
            },
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
        occupied_buckets[self.positions] = True
        return occupied_buckets

    def __repr__(self) -> str:
        """
        Return a short description of the pattern.

        Returns
        -------
        representation
            Class name with harmonic number and bunch count.
        """
        return (
            f"FillingPattern(harmonic_number={self.harmonic_number}, "
            f"n_bunches={self.n_bunches})"
        )

    # A FillingPattern is complete: composition is a usage error.

    def _complete_error(self) -> TypeError:
        """
        Return the error raised by every composition attempt.

        Returns
        -------
        error
            TypeError explaining that the pattern is complete.
        """
        return TypeError(
            "FillingPattern is complete (covers all harmonic_number buckets); "
            "compose PatternSegments first, then wrap once."
        )

    def __add__(self, other: PatternSegment) -> NoReturn:
        """
        Raise TypeError: a complete pattern cannot be concatenated.

        Parameters
        ----------
        other
            Unused.
        """
        raise self._complete_error()

    def __radd__(self, other: PatternSegment) -> NoReturn:
        """
        Raise TypeError: a complete pattern cannot be concatenated.

        Parameters
        ----------
        other
            Unused.
        """
        raise self._complete_error()

    def __mul__(self, n_repetitions: int) -> NoReturn:
        """
        Raise TypeError: a complete pattern cannot be repeated.

        Parameters
        ----------
        n_repetitions
            Unused.
        """
        raise self._complete_error()

    def __rmul__(self, n_repetitions: int) -> NoReturn:
        """
        Raise TypeError: a complete pattern cannot be repeated.

        Parameters
        ----------
        n_repetitions
            Unused.
        """
        raise self._complete_error()

    def gap(self, n_empty_buckets: int) -> NoReturn:
        """
        Raise TypeError: a complete pattern cannot be extended.

        Parameters
        ----------
        n_empty_buckets
            Unused.
        """
        raise self._complete_error()

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_trains(
        cls,
        unit: PatternSegment,
        n_copies: int,
        copy_spacing: int,
        harmonic_number: int,
    ) -> FillingPattern:
        """
        Construct from uniformly spaced copies of a unit.

        The abort gap fills the remainder of the ring.

        Parameters
        ----------
        unit
            Segment to repeat.
        n_copies
            Number of repetitions.
        copy_spacing
            Empty RF buckets between consecutive copies.
        harmonic_number
            Total number of RF buckets in the ring.

        Returns
        -------
        pattern
            Complete filling pattern.
        """
        return cls(
            _repeat_with_gap(unit, n_copies, copy_spacing), harmonic_number
        )

    @classmethod
    def from_spacing(
        cls,
        unit: PatternSegment,
        n_copies: int,
        start_to_start_distance: float,
        f_rf: float,
        harmonic_number: int,
    ) -> FillingPattern:
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
        harmonic_number
            Total number of RF buckets in the ring.

        Returns
        -------
        pattern
            Complete filling pattern.
        """
        copy_spacing = _spacing_from_distance(
            unit.length, start_to_start_distance, f_rf
        )
        return cls.from_trains(
            unit=unit,
            n_copies=n_copies,
            copy_spacing=copy_spacing,
            harmonic_number=harmonic_number,
        )

    @classmethod
    def from_batch_list(
        cls,
        harmonic_number: int,
        placements: list[tuple[PatternSegment, int]],
    ) -> FillingPattern:
        """
        Construct from explicitly positioned segments.

        Tier and payload arrays of the placed segments are preserved and
        merged (tiers re-numbered in position order).

        Parameters
        ----------
        harmonic_number
            Total RF buckets; unoccupied buckets become the abort gap.
        placements
            (segment, start_bucket) pairs. Segments must occupy disjoint
            bucket ranges (no interleaving).

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
        for segment, start_bucket in ordered:
            if start_bucket < previous_end:
                raise ValueError(
                    f"Segment at bucket {start_bucket} overlaps the previous "
                    f"segment (which ends at bucket {previous_end - 1})."
                )
            combined = combined + Gap(start_bucket - previous_end) + segment
            previous_end = start_bucket + segment.length
        return cls(combined, harmonic_number)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from blond.cycles.filling_patterns.plot import plot

    # LHC-like nesting: PS batch -> SPS train -> LHC injection -> full ring
    ps_batch = Batch(n_bunches=72, bunch_spacing=9)  # 25 ns spacing
    sps_train = Train(unit=ps_batch, n_copies=4, copy_spacing=8)
    injection = sps_train.label("injection")
    pattern = FillingPattern(
        injection.gap(38) * 11 + injection,
        harmonic_number=35640,
    )

    print(pattern)
    print(
        f"  n_batches={pattern.n_batches}, n_trains={pattern.n_trains}, "
        f"n_injections={pattern.n_in_tier('injection')}"
    )
    print(f"  positions[:8]:  {pattern.positions[:8]}")
    print(f"  batch[:8]:      {pattern.batch[:8]}")
    print(f"  train[:8]:      {pattern.train[:8]}")
    print(f"  injection[:8]:  {pattern.tier('injection')[:8]}")

    pattern.intensity = np.ones(pattern.n_bunches) * 1.1e11
    pattern.intensity[pattern.batch == 0] = 0.5e11
    pattern.intensity[pattern.tier("injection") == 1] = 0.8e11
    print(f"  intensity[:8]:  {pattern.intensity[:8]}")

    plot(pattern)
    plt.show()
