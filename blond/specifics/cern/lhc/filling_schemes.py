# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Importers for the official LHC filling-scheme files.

The LHC Programme Coordination (LPC, https://lpc.web.cern.ch) publishes
the filling schemes as JSON files. These files are imported directly —
scheme *names* are never parsed, they are lossy labels.

The JSON files carry redundant information: the per-injection composition
(``beam1``/``beam2``: PS batches with per-slot ``bunchArray`` and slot
offsets) and a flat per-slot occupancy mask (``schemebeam1``/
``schemebeam2``). The loader builds the pattern from the composition (so
the ``injection`` and ``batch`` labels survive) and then cross-validates
the result against the mask, refusing inconsistent files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blond.cycles.filling_patterns import FillingPattern, PatternSegment

# Machine constants of the LHC: the harmonic number is fixed by the ring
# circumference and the 400 MHz RF system, the 10-bucket slot by the
# 25 ns timing grid. The scheme files do not state them — they assume
# this grid implicitly, so the loader supplies (and validates) it.
LHC_HARMONIC_NUMBER = 35640
BUCKETS_PER_SLOT = 10  # 400 MHz RF buckets per 25 ns slot
LHC_N_SLOTS = LHC_HARMONIC_NUMBER // BUCKETS_PER_SLOT  # 3564
_NOMINAL_BUNCH_SPACING_NS = 25  # slot length; the only supported scheme type


def filling_pattern_from_scheme_file(
    path: str | Path, beam: int = 1
) -> FillingPattern:
    """
    Load one beam of an official LHC filling-scheme JSON file.

    Builds the :class:`~blond.cycles.filling_patterns.FillingPattern`
    from the file's injection composition, so every bunch carries the
    ``injection`` and ``batch`` labels (group indices in ring order).
    The result is cross-validated against the file's redundant per-slot
    occupancy mask (``schemebeam1``/``schemebeam2``); inconsistent files
    are rejected.

    Parameters
    ----------
    path
        Path to the filling-scheme JSON file (as published by the LHC
        Programme Coordination, see module docstring).
    beam
        LHC beam number, 1 or 2.

    Returns
    -------
    pattern
        Complete ring pattern (harmonic number 35640) with
        ``injection`` and ``batch`` labels.

    Raises
    ------
    ValueError
        If ``beam`` is not 1 or 2, the file uses a bunch spacing other
        than 25 ns, the built pattern contradicts the file's own
        occupancy mask, or a bunch sits at or beyond the abort gap
        keeper boundary (``AGK``).
    """
    if beam not in (1, 2):
        raise ValueError(f"beam must be 1 or 2, got {beam}.")
    data = json.loads(Path(path).read_text())

    placements = []
    for injection in data[f"beam{beam}"]:
        segment, first_slot_in_injection = _injection_segment(
            injection=injection
        )
        start_slot = injection["lhcbunch"] + first_slot_in_injection
        placements.append((segment, start_slot * BUCKETS_PER_SLOT))

    pattern = FillingPattern.from_placements(
        harmonic_number=LHC_HARMONIC_NUMBER, placements=placements
    )
    _validate_against_slot_mask(pattern=pattern, data=data, beam=beam)
    _validate_abort_gap_keeper(pattern=pattern, data=data)
    return pattern


def _injection_segment(
    injection: dict[str, Any],
) -> tuple[PatternSegment, int]:
    # One SPS injection = PS batches at the slot offsets the file states
    # (``injBunch``). Offsets are used directly instead of recomputing a
    # stride from counts and ``batchSpacing`` — the stride shortcut
    # breaks for unequal batch lengths (the legacy loader's bug).
    batches = injection["batches"]
    first_slot = batches[0]["injBunch"]
    segment = _batch_segment(batch=batches[0])
    for batch in batches[1:]:
        offset_buckets = (batch["injBunch"] - first_slot) * BUCKETS_PER_SLOT
        gap = offset_buckets - segment.n_buckets
        if gap < 0:
            raise ValueError(
                f"Batches overlap within one injection: batch at slot "
                f"offset {batch['injBunch']} starts before the previous "
                f"batch ends."
            )
        segment = segment.with_trailing_gap(gap) + _batch_segment(batch=batch)
    return segment.with_label("injection"), first_slot


def _batch_segment(batch: dict[str, Any]) -> PatternSegment:
    # ``bunchArray`` is the per-slot truth (1 = filled slot) and is only
    # known to mean "one 25 ns slot per entry" for 25 ns schemes.
    if batch["bunch_spacing"] != _NOMINAL_BUNCH_SPACING_NS:
        raise ValueError(
            f"Only 25 ns schemes are supported; got bunch_spacing = "
            f"{batch['bunch_spacing']} ns."
        )
    slot_mask = np.asarray(batch["bunchArray"], dtype=int)
    filled_slots = np.nonzero(slot_mask)[0]
    if len(filled_slots) != batch["bunches"]:
        raise ValueError(
            f"Batch declares {batch['bunches']} bunches but its "
            f"bunchArray contains {len(filled_slots)}."
        )
    return PatternSegment(
        bucket_indices=filled_slots * BUCKETS_PER_SLOT,
        # the segment ends right after its last bunch; spacing to the
        # next batch is added as an explicit gap by the caller
        n_buckets=int(filled_slots[-1]) * BUCKETS_PER_SLOT + 1,
        labels={"batch": np.zeros(len(filled_slots), dtype=np.int32)},
    )


def _validate_abort_gap_keeper(
    pattern: FillingPattern, data: dict[str, Any]
) -> None:
    # AGK = abort gap keeper boundary in RF buckets: the abort gap must
    # start there, so no bunch may sit at or beyond it (machine
    # protection). Sanity check only — skipped for files without the key.
    agk = data.get("AGK")
    if agk is None or pattern.n_bunches == 0:
        return
    abort_gap_start = int(agk)
    last_bunch = int(pattern.bucket_indices[-1])
    if last_bunch >= abort_gap_start:
        raise ValueError(
            f"Scheme places a bunch at bucket {last_bunch}, at or beyond "
            f"the abort gap keeper boundary (AGK = {agk}); refusing the "
            f"scheme file."
        )


def _validate_against_slot_mask(
    pattern: FillingPattern, data: dict[str, Any], beam: int
) -> None:
    # The scheme files are redundant; use that to refuse files where
    # the injection composition and the flat mask disagree.
    key = f"schemebeam{beam}"
    expected = np.asarray(data[key], dtype=bool)
    if len(expected) != LHC_N_SLOTS:
        raise ValueError(
            f"'{key}' has {len(expected)} slots, expected {LHC_N_SLOTS}."
        )
    built = np.zeros(LHC_N_SLOTS, dtype=bool)
    built[pattern.bucket_indices // BUCKETS_PER_SLOT] = True
    if not np.array_equal(built, expected):
        n_differing = int(np.sum(built != expected))
        raise ValueError(
            f"Pattern built from the injection composition contradicts "
            f"the file's '{key}' occupancy mask in {n_differing} slots; "
            f"refusing the inconsistent scheme file."
        )
