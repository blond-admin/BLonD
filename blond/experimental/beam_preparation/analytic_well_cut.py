# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Separatrix cut of an analytic potential well.

Restricts an uncut potential well (margined frame, multi-bucket span,
tilted/accelerating well) to a single RF bucket, cut at the separatrix —
the equivalent of the BLonD 2 ``potential_well_cut`` step, built on
:class:`~blond.acc_math.empiric.potential_well.PotentialWellHelper` for
the bucket detection.

The returned well satisfies the single-bucket contract of
``hamiltonian_grid`` and ``action_from_potential_well`` (validated with
:func:`~blond.experimental.beam_preparation.analytic_potential_well.\
check_single_bucket_well` before returning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.empiric.potential_well import PotentialWellHelper
from blond.experimental.beam_preparation.analytic_potential_well import (
    check_single_bucket_well,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray


def cut_potential_well(
    time_array: NumpyArray,
    potential_well: NumpyArray,
    *,
    bucket_index: int | Literal["deepest"] = "deepest",
    subtract_min: bool = True,
    single_bucket_tolerance: float = 1e-2,
    allow_inner_buckets: bool = False,
    verbose: bool = False,
    plot: bool = False,
) -> tuple[NumpyArray, NumpyArray]:
    """
    Cut a potential well at the separatrix of one RF bucket.

    Detects the buckets present in the input frame (via
    :class:`~blond.acc_math.empiric.potential_well.PotentialWellHelper`),
    selects one, and returns the sample-aligned cut: from the bounding
    maximum on one side to the equal-potential crossing on the other.
    This handles the frames that the raw
    :func:`~blond.experimental.beam_preparation.analytic_potential_well.\
rf_potential_well` output may present:

    - margined frames (``dt_margin_fraction > 0``): the neighbouring
      partial buckets are cut away;
    - multi-bucket spans: one bucket is selected;
    - tilted (accelerating) wells: the cut runs from the unstable fixed
      point to the equal-potential turning point, i.e. the true
      separatrix, discarding the unbound branch.

    Parameters
    ----------
    time_array
        Time coordinates of the potential well, in [s].
    potential_well
        Potential well at ``time_array``, in [eV] (uncut).
    bucket_index
        Which bucket to return: ``"deepest"`` (default) selects the
        bucket containing the global minimum of the well; an integer
        selects the ``bucket_index``-th detected bucket counting from
        the left (0-based) — useful for multi-bucket frames.
    subtract_min
        If True (default), shift the cut well so its minimum is at zero
        (the contract expected by the Hamiltonian/action functions).
    single_bucket_tolerance
        Relative tolerance used to validate the cut with
        :func:`check_single_bucket_well` before returning.
    allow_inner_buckets
        If True, a cut containing prominent inner maxima (e.g. the
        enclosing bucket of a well split by an induced potential during
        intensity iterations) is accepted with a warning instead of
        raising; downstream F(H)/F(J) then integrates across the
        sub-wells, neglecting the fine structure (BLonD 2 behavior).
    verbose
        If True, print diagnostic quantities.
    plot
        If True, draw the uncut well with the detected buckets and the
        selected cut.

    Returns
    -------
    time_array_cut
        Time coordinates of the cut well, in [s].
    potential_well_cut
        Cut potential well, in [eV], minimum at zero when
        ``subtract_min`` is True.

    Raises
    ------
    ValueError
        If no bucket is detected in the frame (e.g. below transition
        with ``phi_rf = 0``, where the bucket is split across the frame
        edges — see the ``phi_rf`` convention in
        :func:`bucket_time_array`), if the requested bucket does not
        exist, or if the cut fails the single-bucket validation.
    """
    time_array = np.asarray(time_array, dtype=float)
    potential_well = np.asarray(potential_well, dtype=float)
    assert time_array.shape == potential_well.shape, (
        f"{time_array.shape=} must match {potential_well.shape=}"
    )

    helper = PotentialWellHelper(time_array, potential_well)
    intervals = np.asarray(helper.bucket_list, dtype=float).reshape(-1, 2)
    if intervals.shape[0] == 0:
        raise ValueError(
            "No bucket detected in the potential well frame. If below "
            "transition, remember the BLonD 2 convention phi_rf=pi "
            "(for positive charge) so the bucket is not split across "
            "the frame edges."
        )

    # Work on index pairs and merge near-duplicates:
    # PotentialWellHelper can report the same bucket twice (detected
    # from both of its bounding maxima, borders one sample apart; its
    # off-by-one purge uses index//2 cells and misses pairs straddling
    # a cell boundary). Deduplicate so `bucket_index` counts physical
    # buckets.
    left_indices = np.searchsorted(time_array, intervals[:, 0], "left")
    right_indices = np.searchsorted(time_array, intervals[:, 1], "right") - 1
    index_pairs = np.column_stack((left_indices, right_indices))
    index_pairs = index_pairs[np.argsort(index_pairs[:, 0])]
    kept_pairs = [index_pairs[0]]
    for pair in index_pairs[1:]:
        is_duplicate = (
            abs(int(pair[0]) - int(kept_pairs[-1][0])) <= 2
            and abs(int(pair[1]) - int(kept_pairs[-1][1])) <= 2
        )
        if not is_duplicate:
            kept_pairs.append(pair)
    index_pairs = np.asarray(kept_pairs)
    intervals = np.column_stack(
        (time_array[index_pairs[:, 0]], time_array[index_pairs[:, 1]])
    )

    if bucket_index == "deepest":
        index_of_minimum = int(potential_well.argmin())
        containing = np.flatnonzero(
            (index_pairs[:, 0] <= index_of_minimum)
            & (index_of_minimum <= index_pairs[:, 1])
        )
        if len(containing) == 0:
            raise ValueError(
                "The deepest minimum of the potential well lies outside "
                "every detected bucket — the frame is likely "
                "pathological (minimum on a frame edge?). See the "
                "phi_rf convention in bucket_time_array."
            )
        selected = int(containing[0])
    else:
        selected = int(bucket_index)
        if not 0 <= selected < index_pairs.shape[0]:
            raise ValueError(
                f"bucket_index={bucket_index} out of range: only "
                f"{index_pairs.shape[0]} bucket(s) detected."
            )

    start_time, stop_time = intervals[selected]
    left_index = int(index_pairs[selected, 0])
    right_index = int(index_pairs[selected, 1])
    cut_slice = slice(left_index, right_index + 1)

    time_array_cut = time_array[cut_slice].copy()
    potential_well_cut = potential_well[cut_slice].copy()
    if subtract_min:
        potential_well_cut = potential_well_cut - potential_well_cut.min()

    # Guarantee the downstream single-bucket contract before returning.
    check_single_bucket_well(
        potential_well_cut,
        relative_tolerance=single_bucket_tolerance,
        allow_inner_buckets=allow_inner_buckets,
    )

    if verbose:
        potential_well_amplitude = float(
            potential_well_cut.max() - potential_well_cut.min()
        )
        print(
            "[cut_potential_well] "
            f"buckets detected={intervals.shape[0]}, "
            f"selected={selected} "
            f"(span=[{start_time:.3e}, {stop_time:.3e}] s), "
            f"n_samples={len(time_array_cut)}, "
            f"potential_well_amplitude={potential_well_amplitude:.3e} eV"
        )

    if plot:
        _plot_cut(time_array, potential_well, intervals, selected, cut_slice)

    return time_array_cut, potential_well_cut


def _plot_cut(
    time_array: NumpyArray,
    potential_well: NumpyArray,
    intervals: NumpyArray,
    selected: int,
    cut_slice: slice,
) -> None:
    """Diagnostic plot: uncut well, detected buckets, selected cut."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(num="cut_potential_well")
    time_ns = time_array * 1e9
    ax.plot(
        time_ns, potential_well, color="grey", alpha=0.6, label="uncut well"
    )
    for index, (start_time, stop_time) in enumerate(intervals):
        ax.axvspan(
            start_time * 1e9,
            stop_time * 1e9,
            alpha=0.25 if index == selected else 0.08,
            color="C1" if index == selected else "C0",
        )
    ax.plot(
        time_ns[cut_slice],
        potential_well[cut_slice],
        color="C1",
        lw=2.0,
        label="selected cut",
    )
    separatrix_level = float(potential_well[cut_slice].max())
    ax.axhline(separatrix_level, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Potential well [eV]")
    ax.set_title("Separatrix cut of the potential well")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
