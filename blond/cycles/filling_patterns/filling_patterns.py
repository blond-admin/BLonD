"""RF filling pattern construction for accelerator physics simulations.

Hierarchy: Bunch < Batch < Train < Ring

Each tier stores per-bunch numpy arrays. Concatenation (+) shifts bucket
positions and re-numbers tier indices. Payload attributes can be set per
bunch or per tier using numpy indexing:

    ring.intensity[ring.batch == 2] = 1e11
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib
import numpy as np

from blond.cycles.filling_patterns.helpers import as_n_buckets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UNASSIGNED: int = -1

_BATCH_PALETTE = [matplotlib.colormaps["tab10"](i / 10) for i in range(10)]
_TRAIN_PALETTE = [matplotlib.colormaps["Dark2"](i / 8) for i in range(8)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _next_index(tier_indices: np.ndarray) -> int:
    """First unused index in a tier (max assigned + 1, or 0 if none assigned)."""
    assigned = tier_indices[tier_indices >= 0]
    return int(assigned.max()) + 1 if len(assigned) else 0


def _renumber(tier_indices: np.ndarray, index_offset: int) -> np.ndarray:
    """Shift non-negative tier indices by index_offset; leave _UNASSIGNED unchanged."""
    return np.where(
        tier_indices >= 0, tier_indices + index_offset, _UNASSIGNED
    ).astype(np.int32)


def _merge_payload(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    n_left_bunches: int,
    n_right_bunches: int,
) -> dict[str, np.ndarray]:
    """Concatenate payload arrays; NaN-fill attribute names absent from one side."""
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
    """Empty RF buckets between units given a physical start-to-start distance."""
    return as_n_buckets(start_to_start_distance, f_rf) - unit_length


def _repeat_with_gap(
    unit: FillingPattern, n_copies: int, copy_spacing: int
) -> FillingPattern:
    """n_copies of unit separated by copy_spacing empty buckets (no trailing gap)."""
    if n_copies == 1:
        return unit
    return unit.gap(copy_spacing) * (n_copies - 1) + unit


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class FillingPattern:
    """
    Per-bunch array table for a filling pattern at any tier level.

    Per-bunch arrays (all length n_bunches):

        positions   RF bucket index of each bunch
        batch       batch membership index per bunch  (-1 = unassigned)
        train       train membership index per bunch  (-1 = unassigned)
        bunch       0-based ordinal index  (0, 1, 2, …)

    length
        Total number of RF buckets, including any trailing empty gap.

    payload
        Dict of additional per-bunch arrays. Public attributes not starting
        with '_' are stored here, enabling numpy-masked assignment::

            pattern.intensity = np.ones(pattern.n_bunches) * 1e11
            pattern.intensity[pattern.batch == 2] = 0.5e11

    Concatenation (+) shifts positions and re-numbers tier indices::

        combined = a.gap(5) + b
    """

    _positions: np.ndarray
    _batch_indices: np.ndarray
    _train_indices: np.ndarray
    _length: int
    _payload: dict[str, np.ndarray]

    def __init__(
        self,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        train_indices: np.ndarray,
        length: int,
        payload: dict[str, np.ndarray] | None = None,
    ):
        object.__setattr__(
            self, "_positions", np.asarray(positions, dtype=np.int64)
        )
        object.__setattr__(
            self, "_batch_indices", np.asarray(batch_indices, dtype=np.int32)
        )
        object.__setattr__(
            self, "_train_indices", np.asarray(train_indices, dtype=np.int32)
        )
        object.__setattr__(self, "_length", int(length))
        object.__setattr__(
            self, "_payload", {} if payload is None else dict(payload)
        )

    # ------------------------------------------------------------------ counts

    @property
    def n_bunches(self) -> int:
        """Number of bunches in this pattern."""
        return len(self._positions)

    @property
    def n_batches(self) -> int:
        """Number of distinct assigned batch indices."""
        return _next_index(self.batch)

    @property
    def n_trains(self) -> int:
        """Number of distinct assigned train indices."""
        return _next_index(self.train)

    # ------------------------------------------------------------------ labels

    @property
    def length(self) -> int:
        """Total number of RF buckets, including any trailing empty gap."""
        return self._length

    @property
    def positions(self) -> np.ndarray:
        """RF bucket index of each bunch. Shape (n_bunches,)."""
        return self._positions

    @property
    def bunch(self) -> np.ndarray:
        """0-based ordinal index per bunch. Enables ``bunch % 2 == 0``."""
        return np.arange(self.n_bunches, dtype=np.int32)

    @property
    def batch(self) -> np.ndarray:
        """Batch membership index per bunch (-1 = unassigned). Enables ``batch == 2``."""
        return self._batch_indices

    @property
    def train(self) -> np.ndarray:
        """Train membership index per bunch (-1 = unassigned). Enables ``train == 0``."""
        return self._train_indices

    @property
    def payload(self) -> dict[str, np.ndarray]:
        """Per-bunch payload arrays, keyed by attribute name."""
        return self._payload

    # ----------------------------------------------------------- payload i/o
    # Public attributes (no leading '_') are routed to _payload, enabling
    # the ring.intensity = ...; ring.intensity[mask] = ... interface.

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return object.__getattribute__(self, "_payload")[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        arr = np.asarray(value)
        if arr.ndim != 1 or len(arr) != self.n_bunches:
            raise ValueError(
                f"Payload '{name}' must be 1-D with length {self.n_bunches}; "
                f"got shape {arr.shape}."
            )
        self._payload[name] = arr

    # -------------------------------------------------------------------- gap

    def gap(self, n_empty_buckets: int) -> FillingPattern:
        """Return self with n_empty_buckets appended (length increases accordingly)."""
        return self + Gap(n_empty_buckets)

    # ----------------------------------------------------------- concatenation

    def __add__(self, other: FillingPattern) -> FillingPattern:
        return FillingPattern(
            positions=np.concatenate(
                [self.positions, other.positions + self.length]
            ),
            batch_indices=np.concatenate(
                [self.batch, _renumber(other.batch, _next_index(self.batch))]
            ),
            train_indices=np.concatenate(
                [self.train, _renumber(other.train, _next_index(self.train))]
            ),
            length=self.length + other.length,
            payload=_merge_payload(
                self.payload, other.payload, self.n_bunches, other.n_bunches
            ),
        )

    def __mul__(self, n_repetitions: int) -> FillingPattern:
        if n_repetitions < 1:
            raise ValueError(f"Multiplier must be >= 1, got {n_repetitions}.")
        result: FillingPattern = self
        for _ in range(n_repetitions - 1):
            result = result + self
        return result

    def __rmul__(self, n_repetitions: int) -> FillingPattern:
        return self.__mul__(n_repetitions)

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_bunches={self.n_bunches}, length={self.length})"


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------


class Gap(FillingPattern):
    """n_empty_buckets RF buckets with no bunches — used to space patterns."""

    def __init__(self, n_empty_buckets: int):
        super().__init__(
            positions=np.empty(0, dtype=np.int64),
            batch_indices=np.empty(0, dtype=np.int32),
            train_indices=np.empty(0, dtype=np.int32),
            length=n_empty_buckets,
        )


# ---------------------------------------------------------------------------
# Bunch
# ---------------------------------------------------------------------------


class Bunch(FillingPattern):
    """Single filled RF bucket, not assigned to any batch or train."""

    def __init__(self):
        super().__init__(
            positions=np.array([0], dtype=np.int64),
            batch_indices=np.full(1, _UNASSIGNED, dtype=np.int32),
            train_indices=np.full(1, _UNASSIGNED, dtype=np.int32),
            length=1,
        )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class Batch(FillingPattern):
    """
    n_bunches RF buckets filled, separated by bunch_spacing empty buckets,
    all labeled batch index 0.

    Concatenation re-numbers batch indices automatically::

        two = Batch(n_bunches=4, bunch_spacing=1).gap(5) + Batch(n_bunches=4, bunch_spacing=1)
        two.batch  # [0, 0, 0, 0,  1, 1, 1, 1]

    Parameters
    ----------
    n_bunches : int
        Number of bunches per batch.
    bunch_spacing : int
        Empty RF buckets between consecutive bunches.
    """

    def __init__(self, n_bunches: int, bunch_spacing: int):
        if n_bunches < 1:
            raise ValueError(f"n_bunches must be >= 1, got {n_bunches}.")
        bunch_stride = 1 + bunch_spacing
        super().__init__(
            positions=np.arange(n_bunches, dtype=np.int64) * bunch_stride,
            batch_indices=np.zeros(n_bunches, dtype=np.int32),
            train_indices=np.full(n_bunches, _UNASSIGNED, dtype=np.int32),
            length=n_bunches + (n_bunches - 1) * bunch_spacing,
        )

    @classmethod
    def from_spacing(
        cls, n_bunches: int, start_to_start_distance: float, f_rf: float
    ) -> Batch:
        """Construct from a physical bunch start-to-start distance (seconds)."""
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


class Train(FillingPattern):
    """
    n_copies of unit separated by copy_spacing empty buckets, all labeled
    train index 0.  Batch indices from unit are preserved and re-numbered
    across copies.

    Concatenation re-numbers train indices::

        two = Train(batch, n_copies=3, copy_spacing=5).gap(100) + Train(batch, n_copies=3, copy_spacing=5)
        two.train  # [0, 0, ...,  1, 1, ...]

    Parameters
    ----------
    unit : FillingPattern
        Pattern to repeat (typically a Batch or concatenated batches).
    n_copies : int
        Number of repetitions.
    copy_spacing : int
        Empty RF buckets between consecutive copies of unit.
    """

    def __init__(self, unit: FillingPattern, n_copies: int, copy_spacing: int):
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}.")
        combined_pattern = _repeat_with_gap(unit, n_copies, copy_spacing)
        super().__init__(
            positions=combined_pattern.positions.copy(),
            batch_indices=combined_pattern.batch.copy(),
            train_indices=np.zeros(combined_pattern.n_bunches, dtype=np.int32),
            length=combined_pattern.length,
            payload={
                name: arr.copy()
                for name, arr in combined_pattern.payload.items()
            },
        )

    @classmethod
    def from_spacing(
        cls,
        unit: FillingPattern,
        n_copies: int,
        start_to_start_distance: float,
        f_rf: float,
    ) -> Train:
        """Construct from a physical unit start-to-start distance (seconds)."""
        return cls(
            unit=unit,
            n_copies=n_copies,
            copy_spacing=_spacing_from_distance(
                unit.length, start_to_start_distance, f_rf
            ),
        )


# ---------------------------------------------------------------------------
# Ring
# ---------------------------------------------------------------------------


class Ring(FillingPattern):
    """
    Complete ring filling pattern: exactly harmonic_number RF buckets.

    Wraps any FillingPattern; remaining buckets form the abort gap.

    Usage::

        ring = Ring.from_trains(train, n_copies=2, copy_spacing=200, harmonic_number=2545)
        ring.intensity = np.ones(ring.n_bunches) * 1e11
        ring.intensity[ring.batch == 3] = 0.5e11
        ring.intensity[ring.train == 1] = 0.8e11
    """

    _harmonic_number: int

    def __init__(self, pattern: FillingPattern, harmonic_number: int):
        if pattern.length > harmonic_number:
            raise ValueError(
                f"Pattern length ({pattern.length} buckets) exceeds "
                f"harmonic_number ({harmonic_number})."
            )
        object.__setattr__(self, "_harmonic_number", harmonic_number)
        super().__init__(
            positions=pattern.positions.copy(),
            batch_indices=pattern.batch.copy(),
            train_indices=pattern.train.copy(),
            length=harmonic_number,
            payload={
                name: arr.copy() for name, arr in pattern.payload.items()
            },
        )

    @property
    def harmonic_number(self) -> int:
        """Total number of RF buckets in the ring."""
        return self._harmonic_number

    @property
    def has_bunch(self) -> np.ndarray:
        """Bool array of length harmonic_number; True at each filled RF bucket."""
        occupied_buckets = np.zeros(self.harmonic_number, dtype=bool)
        occupied_buckets[self.positions] = True
        return occupied_buckets

    def __repr__(self) -> str:
        return f"Ring(harmonic_number={self.harmonic_number}, n_bunches={self.n_bunches})"

    @classmethod
    def from_trains(
        cls,
        unit: FillingPattern,
        n_copies: int,
        copy_spacing: int,
        harmonic_number: int,
    ) -> Ring:
        """n_copies of unit uniformly spaced; abort gap fills the remainder."""
        return cls(
            _repeat_with_gap(unit, n_copies, copy_spacing), harmonic_number
        )

    @classmethod
    def from_spacing(
        cls,
        unit: FillingPattern,
        n_copies: int,
        start_to_start_distance: float,
        f_rf: float,
        harmonic_number: int,
    ) -> Ring:
        """Construct from a physical unit start-to-start distance (seconds)."""
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
        placements: list[tuple[Batch, int]],
    ) -> Ring:
        """
        Construct from explicitly positioned batches.

        Parameters
        ----------
        harmonic_number
            Total RF buckets; unoccupied buckets become the abort gap.
        placements
            (batch, start_bucket) pairs.  Batches must not overlap.
        """
        bunch_bucket_list: list[int] = []
        batch_index_list: list[int] = []
        occupied_buckets: set[int] = set()

        for batch_index, (batch_unit, start_bucket) in enumerate(placements):
            if start_bucket + batch_unit.length > harmonic_number:
                raise ValueError(
                    f"Batch at bucket {start_bucket} (length {batch_unit.length}) "
                    f"exceeds harmonic_number {harmonic_number}."
                )
            for relative_bucket in batch_unit.positions:
                bucket_position = start_bucket + int(relative_bucket)
                if bucket_position in occupied_buckets:
                    raise ValueError(
                        f"Overlapping bunches at bucket {bucket_position}."
                    )
                occupied_buckets.add(bucket_position)
                bunch_bucket_list.append(bucket_position)
                batch_index_list.append(batch_index)

        sorted_order = np.argsort(bunch_bucket_list)
        bunch_positions = np.array(bunch_bucket_list, dtype=np.int64)[
            sorted_order
        ]
        batch_indices = np.array(batch_index_list, dtype=np.int32)[
            sorted_order
        ]
        train_indices = np.full(
            len(bunch_positions), _UNASSIGNED, dtype=np.int32
        )

        return cls(
            FillingPattern(
                bunch_positions, batch_indices, train_indices, harmonic_number
            ),
            harmonic_number,
        )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(
    pattern: FillingPattern,
    f_rf: float | None = None,
    ax: Any = None,
    face: Any = None,
    edge: Any = None,
) -> Any:
    """
    Plot a filling pattern as one bar per bunch.

    Bar fill color encodes batch index; bar edge color encodes train index.
    Override with face/edge: array of length n_bunches, callable, or None.

    Parameters
    ----------
    pattern
        Any filling pattern (Batch, Train, Ring, …).
    f_rf : float, optional
        RF frequency in Hz.  When given, x-axis shows time in nanoseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if None.
    face : array-like or callable, optional
        Face color per bunch (None = color by batch index).
    edge : array-like or callable, optional
        Edge color per bunch (None = color by train index).

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    Highlight batch 2 in red::

        mask = ring.batch == 2
        plot(ring, face=np.where(mask, 'red', 'lightgray'))
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2))

    bucket_x, bucket_width = _x_axis(pattern.positions, f_rf)
    faces = _resolve_colors(
        face, pattern.positions, pattern.batch, _BATCH_PALETTE, default="gray"
    )
    edges = _resolve_colors(
        edge, pattern.positions, pattern.train, _TRAIN_PALETTE, default="black"
    )

    ax.bar(
        bucket_x,
        1.0,
        width=bucket_width,
        color=faces,
        edgecolor=edges,
        linewidth=1.0,
        align="edge",
        antialiased=False,
    )
    ax.set_xlabel("Time (ns)" if f_rf is not None else "Bucket")
    _draw_train_boundaries(ax, pattern.positions, pattern.train, f_rf)
    ax.set_xlim(
        0.0, pattern.length * (1e9 / f_rf if f_rf is not None else 1.0)
    )
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    return ax


def _x_axis(
    bunch_positions: np.ndarray, f_rf: float | None
) -> tuple[np.ndarray, float]:
    if f_rf is None:
        return bunch_positions.astype(float), 1.0
    return bunch_positions * 1e9 / f_rf, 1e9 / f_rf


def _draw_train_boundaries(
    ax: Any,
    bunch_positions: np.ndarray,
    train_indices: np.ndarray,
    f_rf: float | None,
) -> None:
    for i, (bucket, train_index) in enumerate(
        zip(bunch_positions, train_indices)
    ):
        is_train_start = train_index >= 0 and (
            i == 0 or train_indices[i - 1] != train_index
        )
        if is_train_start:
            x = float(bucket) * (1e9 / f_rf if f_rf is not None else 1.0)
            ax.axvline(
                x,
                color=_TRAIN_PALETTE[train_index % len(_TRAIN_PALETTE)],
                lw=1.0,
                ls="--",
                alpha=0.8,
            )


def _resolve_colors(
    spec: Any,
    bunch_positions: np.ndarray,
    tier_labels: np.ndarray,
    palette: list,
    default: str,
) -> list:
    if spec is None:
        return _structural_colors(tier_labels, palette, default)
    if callable(spec):
        return [spec(int(bucket)) for bucket in bunch_positions]
    colors = list(spec)
    if len(colors) != len(bunch_positions):
        raise ValueError(
            f"Color array length {len(colors)} != n_bunches {len(bunch_positions)}."
        )
    return colors


def _structural_colors(
    tier_labels: np.ndarray, palette: list, default: str
) -> list:
    return [
        palette[int(tier_labels[i]) % len(palette)]
        if tier_labels[i] >= 0
        else default
        for i in range(len(tier_labels))
    ]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch = Batch(n_bunches=4, bunch_spacing=1)
    train = Train(unit=batch, n_copies=10, copy_spacing=10)
    n_trains = int(math.floor(2545 / (train.length + 100)))
    ring = Ring.from_trains(
        unit=train, n_copies=n_trains, copy_spacing=100, harmonic_number=2545
    )

    print(ring)
    print(f"  n_batches={ring.n_batches}, n_trains={ring.n_trains}")
    print(f"  positions[:8]: {ring.positions[:8]}")
    print(f"  batch[:8]:     {ring.batch[:8]}")
    print(f"  train[:8]:     {ring.train[:8]}")

    ring.intensity = np.ones(ring.n_bunches) * 1e11
    ring.intensity[ring.batch == 0] = 0.5e11
    ring.intensity[ring.train == 1] = 0.8e11
    print(f"  intensity[:8]: {ring.intensity[:8]}")

    plot(ring)
    plt.show()
