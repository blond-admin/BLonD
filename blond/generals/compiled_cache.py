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

_META_NAME = "meta.json"

#: How many ``compiled/<hash>/`` directories to retain. Overridable via the
#: ``BLOND_COMPILED_CACHE_KEEP`` environment variable (e.g. in CI).
DEFAULT_KEEP = int(os.environ.get("BLOND_COMPILED_CACHE_KEEP", "20"))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def mark_used(directory: str) -> None:
    """
    Record that ``directory`` was just built or loaded.

    Writes ``built_at`` once (preserved across calls) and refreshes
    ``last_used`` to now. Best-effort: any I/O error is ignored.

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
    except OSError:
        pass  # never let cache bookkeeping break the build/load


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


def prune(compiled_root: str, keep: int = DEFAULT_KEEP) -> None:
    """
    Keep only the ``keep`` most-recently-used subdirectories of a root.

    Best-effort: a directory that cannot be removed (for instance a library
    still loaded by another process, which Windows locks) is skipped and
    retried on a later run.

    Parameters
    ----------
    compiled_root
        The ``compiled/`` directory whose ``<hash>`` subdirectories are pruned.
    keep
        Number of most-recently-used subdirectories to retain.
    """
    try:
        subdirs = [
            entry.path for entry in os.scandir(compiled_root) if entry.is_dir()
        ]
    except OSError:
        return
    if len(subdirs) <= keep:
        return
    subdirs.sort(key=_last_used, reverse=True)  # newest first
    for stale in subdirs[keep:]:
        # e.g. still in use (Windows locks loaded libs); retried next run.
        with contextlib.suppress(OSError):
            shutil.rmtree(stale)
