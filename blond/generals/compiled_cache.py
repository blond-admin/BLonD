# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Bounded LRU housekeeping for compiled-backend directories.

Each ``compiled/<hash>/`` directory holds a backend built for one specific
toolchain/CPU/flag combination (see the backends' ``compiled_dir_handler``
modules). Across many CI runs on heterogeneous runners these accumulate
without bound, bloating the on-disk -- and CI-cached -- footprint. To keep it
bounded we:

* stamp each directory with a small ``meta.json`` recording when it was first
  built (``built_at``) and when it was last used (``last_used``), and
* keep only the ``keep`` most-recently-used directories, evicting the rest.

``built_at`` is informational (debugging "when was this compiled"); eviction is
driven solely by ``last_used``.

Every operation is best-effort: cache bookkeeping must never break a compile or
a backend load, so errors are swallowed.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import shutil
import warnings
from pathlib import Path

_META_NAME = "meta.json"

# How many ``compiled/<hash>/`` directories to retain. Overridable via the
# ``BLOND_COMPILED_CACHE_KEEP_N`` environment variable (e.g. in CI).
DEFAULT_KEEP_N = int(os.environ.get("BLOND_COMPILED_CACHE_KEEP_N", "100"))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def mark_used(directory: str) -> None:
    """
    Record that ``directory`` was just built or loaded.

    Writes ``built_at`` once (preserved across calls) and refreshes
    ``last_used`` to now. Best-effort: an I/O error is warned about but never
    propagated, so cache bookkeeping cannot break the build/load.

    Parameters
    ----------
    directory
        The ``compiled/<hash>`` directory to stamp.
    """
    try:
        meta_path = os.path.join(directory, _META_NAME)
        meta: dict[str, str] = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as file:
                    meta = json.load(file)
            except (OSError, ValueError):
                meta = {}
        meta.setdefault("built_at", _now())
        meta["last_used"] = _now()
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(meta, file)
    except OSError as exc:
        # Never let cache bookkeeping break the build/load, but surface it:
        # a persistently failing stamp defeats the LRU eviction silently.
        warnings.warn(
            f"Could not update compiled-cache metadata in {directory!r}: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _last_used(directory: str) -> float:
    """
    Return the eviction sort key for a directory (newer == larger).

    Uses ``last_used`` from ``meta.json`` when available; otherwise falls back
    to the directory mtime, and finally to ``0`` so that un-stamped (e.g.
    legacy) directories sort oldest and are evicted first.

    Parameters
    ----------
    directory
        The ``compiled/<hash>`` directory to score.

    Returns
    -------
    key
        POSIX timestamp used to order directories from newest to oldest.
    """
    meta_path = os.path.join(directory, _META_NAME)
    try:
        with open(meta_path, encoding="utf-8") as file:
            stamp = json.load(file)["last_used"]
        return datetime.datetime.fromisoformat(stamp).timestamp()
    except (OSError, ValueError, KeyError, TypeError):
        try:
            return os.path.getmtime(directory)
        except OSError:
            return 0.0


def prune_siblings(active_dir: str, keep_n: int = DEFAULT_KEEP_N) -> None:
    """
    Evict least-recently-used sibling directories of ``active_dir``.

    ``active_dir`` is the ``compiled/<hash>`` directory just built or loaded;
    it is **never** removed. Among its siblings (the other ``<hash>``
    directories in the same ``compiled/`` parent) the most-recently-used are
    retained and the older ones removed, so that at most ``keep_n`` directories
    remain in total (the active one plus the ``keep_n - 1`` freshest siblings).

    Best-effort: a directory that cannot be removed (for instance a library
    still loaded by another process, which Windows locks) is skipped and
    retried on a later run.

    Parameters
    ----------
    active_dir
        The ``compiled/<hash>`` directory currently in use; protected from
        eviction. Its parent is the ``compiled/`` directory being pruned.
    keep_n
        Maximum number of directories to retain in total (``>= 1``).
    """
    active = Path(active_dir)
    try:
        siblings = [
            entry.path
            for entry in os.scandir(active.parent)
            if entry.is_dir() and entry.name != active.name
        ]
    except OSError:
        return
    keep_siblings = max(keep_n - 1, 0)  # reserve one slot for the active dir
    if len(siblings) <= keep_siblings:
        return
    siblings.sort(key=_last_used, reverse=True)  # newest first
    for stale in siblings[keep_siblings:]:
        # e.g. still in use (Windows locks loaded libs); retried next run.
        with contextlib.suppress(OSError):
            shutil.rmtree(stale)
