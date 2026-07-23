"""Bound the on-disk numba JIT cache to a maximum size.

numba persists compiled kernels as ``.nbc``/``.nbi`` files under
``NUMBA_CACHE_DIR``. It never deletes stale entries, so as the kernels change
across commits the cache grows without bound -- which bloats the GitLab cache
that persists ``.numba_cache/`` between pipelines.

This script keeps the cache under a size budget by deleting the
least-recently-written files first (numba simply recompiles and rewrites an
evicted entry on the next miss, so eviction is always safe). It is pure
standard library on purpose: it runs with a bare ``python3`` in CI
``after_script`` (before the cache is uploaded) without activating the project
venv, and without importing ``blond`` -- importing the package would trigger
backend setup and *add* to the very cache we are about to prune.

Configuration (environment variables):

``NUMBA_CACHE_DIR``
    Directory holding the cache (set globally in the CI config).
``BLOND_NUMBA_CACHE_MAX_MB``
    Size budget in MiB. Defaults to :data:`DEFAULT_MAX_MB`.
"""

from __future__ import annotations

import os

#: Default cache size budget, in MiB. Generous: it only caps runaway growth
#: from orphaned entries, not the working set of a single run.
DEFAULT_MAX_MB = 50


def prune(cache_dir: str | None, max_mb: float = DEFAULT_MAX_MB) -> None:
    """
    Delete least-recently-written cache files until under ``max_mb``.

    Best-effort: a missing directory or an unremovable file is ignored, so
    this never raises.

    Parameters
    ----------
    cache_dir
        The numba cache directory. If ``None`` or missing, nothing happens.
    max_mb
        Maximum total size to retain, in MiB.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    max_bytes = int(max_mb * 1024 * 1024)

    files = []
    total = 0
    for root, _dirs, names in os.walk(cache_dir):
        for name in names:
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size

    if total <= max_bytes:
        return

    files.sort()  # oldest mtime first -> evicted first
    for _mtime, size, path in files:
        if total <= max_bytes:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass  # e.g. concurrently removed; keep going


def main() -> None:
    """Prune the cache described by the environment variables."""
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    try:
        max_mb = float(os.environ.get("BLOND_NUMBA_CACHE_MAX_MB", ""))
    except ValueError:
        max_mb = DEFAULT_MAX_MB
    prune(cache_dir, max_mb)


if __name__ == "__main__":
    main()
