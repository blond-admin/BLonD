# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Plotting for filling patterns.

matplotlib is imported lazily so the rest of the package works headless
without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blond.cycles.filling_patterns.filling_patterns import BunchTable


def plot(
    pattern: BunchTable,
    f_rf: float | None = None,
    ax: Any = None,
    face: Any = None,
    edge: Any = None,
    face_label: str = "batch",
    edge_label: str = "train",
) -> Any:
    """
    Plot a filling pattern as one bar per bunch.

    Bar fill color encodes the face_label group index; bar edge color
    encodes the edge_label group index. Override with face/edge: array of
    length n_bunches, callable, or None.

    Parameters
    ----------
    pattern
        Any pattern (Batch, Train, FillingPattern, ...).
    f_rf
        RF frequency in Hz.  When given, x-axis shows time in nanoseconds.
    ax
        Matplotlib axes to draw on; created if None.
    face
        Face color per bunch: array of length n_bunches, callable, or
        None (= color by face_label group index).
    edge
        Edge color per bunch: array of length n_bunches, callable, or
        None (= color by edge_label group index).
    face_label
        Label whose group indices select the face color (default 'batch').
    edge_label
        Label whose group indices select the edge color and the dashed
        boundary lines (default 'train').

    Returns
    -------
    ax
        Matplotlib axes containing the plot.

    Examples
    --------
    Highlight batch 2 in red::

        mask = pattern.label("batch") == 2
        plot(pattern, face=np.where(mask, 'red', 'lightgray'))

    Color by a custom label::

        plot(pattern, face_label='injection')
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2))

    face_palette = _palette("tab10", 10)
    edge_palette = _palette("Dark2", 8)
    face_groups = _label_column_or_unassigned(pattern, face_label)
    edge_groups = _label_column_or_unassigned(pattern, edge_label)

    bucket_x, bucket_width = _x_axis(pattern.bucket_indices, f_rf)
    faces = _resolve_colors(
        face, pattern.bucket_indices, face_groups, face_palette, default="gray"
    )
    edges = _resolve_colors(
        edge,
        pattern.bucket_indices,
        edge_groups,
        edge_palette,
        default="black",
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
    _draw_group_boundaries(
        ax, pattern.bucket_indices, edge_groups, f_rf, edge_palette
    )
    ax.set_xlim(
        0.0, pattern.n_buckets * (1e9 / f_rf if f_rf is not None else 1.0)
    )
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    return ax


def _palette(colormap_name: str, n_colors: int) -> list:
    # Evenly sampled colors from a named matplotlib colormap.
    import matplotlib

    return [
        matplotlib.colormaps[colormap_name](i / n_colors)
        for i in range(n_colors)
    ]


def _label_column_or_unassigned(
    pattern: BunchTable, label_name: str
) -> np.ndarray:
    # Label column, defaulting to all-unassigned (-1) if absent.
    return pattern.labels.get(
        label_name, np.full(pattern.n_bunches, -1, dtype=np.int32)
    )


def _x_axis(
    bunch_positions: np.ndarray, f_rf: float | None
) -> tuple[np.ndarray, float]:
    # Bar left edges and width (nanoseconds if f_rf given, else buckets).
    if f_rf is None:
        return bunch_positions.astype(float), 1.0
    return bunch_positions * 1e9 / f_rf, 1e9 / f_rf


def _draw_group_boundaries(
    ax: Any,
    bunch_positions: np.ndarray,
    group_indices: np.ndarray,
    f_rf: float | None,
    palette: list,
) -> None:
    # Dashed vertical line at the first bunch of each group.
    for i, (bucket, group_index) in enumerate(
        zip(bunch_positions, group_indices, strict=True)
    ):
        is_group_start = group_index >= 0 and (
            i == 0 or group_indices[i - 1] != group_index
        )
        if is_group_start:
            x = float(bucket) * (1e9 / f_rf if f_rf is not None else 1.0)
            ax.axvline(
                x,
                color=palette[group_index % len(palette)],
                lw=1.0,
                ls="--",
                alpha=0.8,
            )


def _resolve_colors(
    spec: Any,
    bunch_positions: np.ndarray,
    group_indices: np.ndarray,
    palette: list,
    default: str,
) -> list:
    # One color per bunch from a user spec (array | callable | None);
    # None falls back to coloring by group index, `default` for unassigned.
    if spec is None:
        return [
            palette[int(group) % len(palette)] if group >= 0 else default
            for group in group_indices
        ]
    if callable(spec):
        return [spec(int(bucket)) for bucket in bunch_positions]
    colors = list(spec)
    if len(colors) != len(bunch_positions):
        raise ValueError(
            f"Color array length {len(colors)} != n_bunches {len(bunch_positions)}."
        )
    return colors
