"""Keep numba's on-disk JIT cache valid across CI checkouts and report it.

numba decides whether a cached kernel is still usable by comparing the
``(st_mtime, st_size)`` of the *source file* with the stamp recorded in the
cache index (``numba.core.caching._SourceFileBackedLocatorMixin``). A fresh
CI checkout gives every file the checkout time as mtime, so a cache persisted
between pipelines (``NUMBA_CACHE_DIR``, see ``.gitlab-ci.yml``) never matches
and every kernel is silently recompiled during the test run -- numba just
compiles on a miss and never says so.

``stamp``
    Give every ``*.py`` file below the given directories an mtime derived
    from its content hash. Identical content then means an identical stamp
    on every runner, so unchanged files hit the cache; a changed file gets a
    new stamp and is recompiled. (numba additionally keys every entry on the
    function's bytecode hash, so the stamp is only the cheap first check,
    never the only safeguard.) Run this *before* anything imports ``blond``.

``report``
    Print what is in ``NUMBA_CACHE_DIR``: how many index files are valid for
    the current sources, how many are stale (will recompile) or orphaned
    (source no longer exists at that path), the data-file count and size,
    and -- when a snapshot from ``--snapshot`` taken before the tests is
    passed with ``--compare`` -- how many kernels were compiled during the
    job (the cache misses).

Pure standard library on purpose: CI runs it with a bare ``python3``,
without activating the project venv and without importing ``blond`` (which
would compile kernels into the very cache being inspected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field

#: Earliest mtime handed out by :func:`content_mtime` (2001-09-09). Any
#: value is fine for numba, which only compares for equality; a past date
#: keeps build tools that look at mtimes from thinking files are "new".
BASE_MTIME = 1_000_000_000

#: Width of the mtime window: 2^31 distinct values spread over ~68 years.
MTIME_RANGE = 2**31

#: Directories that never hold sources numba compiles from.
SKIP_DIRS = frozenset({"__pycache__", ".git"})


def content_mtime(data: bytes) -> int:
    """
    Return a deterministic mtime for file content.

    Parameters
    ----------
    data
        The file's bytes.

    Returns
    -------
    int
        Seconds since the epoch, within
        ``[BASE_MTIME, BASE_MTIME + MTIME_RANGE)``. Equal content gives an
        equal value on every machine.
    """
    digest = hashlib.sha256(data).digest()
    return BASE_MTIME + int.from_bytes(digest[:8], "big") % MTIME_RANGE


def iter_python_files(roots: list[str]):
    """Yield every ``*.py`` file below ``roots``, skipping ``SKIP_DIRS``."""
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def stamp_sources(roots: list[str]) -> list[str]:
    """
    Set the mtime of every ``*.py`` file below ``roots`` from its content.

    Parameters
    ----------
    roots
        Directories to walk.

    Returns
    -------
    list of str
        The files whose mtime was changed. A second call on an unchanged
        tree returns an empty list.
    """
    changed = []
    for path in iter_python_files(roots):
        with open(path, "rb") as file:
            mtime = content_mtime(file.read())
        if os.stat(path).st_mtime == mtime:
            continue
        os.utime(path, (mtime, mtime))
        changed.append(path)
    return changed


def cache_subdir_for(source_dir: str) -> str:
    """
    Return the cache sub-directory numba uses for sources in ``source_dir``.

    Mirrors ``numba.core.caching._CacheLocator.get_suitable_cache_subpath``:
    the directory's basename plus the SHA-1 of its absolute path.
    """
    source_dir = os.path.abspath(source_dir)
    parent = os.path.split(source_dir)[-1]
    return "_".join([parent, hashlib.sha1(source_dir.encode()).hexdigest()])


def source_stamp(path: str) -> tuple[float, int]:
    """Return the ``(mtime, size)`` stamp numba records for ``path``."""
    stat = os.stat(path)
    return stat.st_mtime, stat.st_size


@dataclass
class IndexEntry:
    """One ``.nbi`` index file of the numba cache."""

    path: str
    source: str | None
    n_overloads: int
    numba_version: str | None


@dataclass
class CacheReport:
    """What :func:`scan_cache` found in a numba cache directory."""

    cache_dir: str
    valid: list[IndexEntry] = field(default_factory=list)
    stale: list[IndexEntry] = field(default_factory=list)
    orphaned: list[IndexEntry] = field(default_factory=list)
    n_data_files: int = 0
    size_bytes: int = 0


def read_index(path: str) -> tuple[str, tuple, int] | None:
    """
    Return ``(numba_version, stamp, n_overloads)`` of an index file.

    ``None`` when the file cannot be parsed. Mirrors
    ``numba.core.caching.IndexDataCacheFile._load_index``.
    """
    try:
        with open(path, "rb") as file:
            version = pickle.load(file)
            stamp, overloads = pickle.loads(file.read())
        return str(version), tuple(stamp), len(overloads)
    except Exception:  # noqa: BLE001 - any corrupt index is just "unreadable"
        return None


def _sources_by_cache_key(roots: list[str]) -> dict[tuple[str, str], str]:
    """Map ``(cache subdir, module name)`` to the source file it belongs to."""
    mapping = {}
    for path in iter_python_files(roots):
        subdir = cache_subdir_for(os.path.dirname(path))
        module = os.path.splitext(os.path.basename(path))[0]
        mapping[(subdir, module)] = path
    return mapping


def scan_cache(cache_dir: str, roots: list[str]) -> CacheReport:
    """
    Classify every index file in ``cache_dir`` against the sources.

    Parameters
    ----------
    cache_dir
        The ``NUMBA_CACHE_DIR``.
    roots
        Source directories the cached kernels may come from.

    Returns
    -------
    CacheReport
        Valid entries match the current source stamp; stale ones belong to
        a source whose stamp differs (numba will recompile); orphaned ones
        have no source at the recorded location or cannot be read.
    """
    report = CacheReport(cache_dir=cache_dir)
    if not os.path.isdir(cache_dir):
        return report
    sources = _sources_by_cache_key(roots)
    for dirpath, _, filenames in os.walk(cache_dir):
        subdir = os.path.basename(dirpath)
        for name in filenames:
            path = os.path.join(dirpath, name)
            report.size_bytes += os.stat(path).st_size
            if name.endswith(".nbc"):
                report.n_data_files += 1
                continue
            if not name.endswith(".nbi"):
                continue
            parsed = read_index(path)
            source = sources.get((subdir, name.split(".", 1)[0]))
            if parsed is None:
                report.orphaned.append(IndexEntry(path, source, 0, None))
                continue
            version, stamp, n_overloads = parsed
            entry = IndexEntry(path, source, n_overloads, version)
            if source is None:
                report.orphaned.append(entry)
            elif stamp == source_stamp(source):
                report.valid.append(entry)
            else:
                report.stale.append(entry)
    return report


def _data_files(cache_dir: str) -> list[str]:
    """Return the ``.nbc`` files below ``cache_dir``, relative and sorted."""
    found = []
    for dirpath, _, filenames in os.walk(cache_dir):
        for name in filenames:
            if name.endswith(".nbc"):
                found.append(
                    os.path.relpath(os.path.join(dirpath, name), cache_dir)
                )
    return sorted(found)


def write_snapshot(cache_dir: str, snapshot_file: str) -> None:
    """Record the data files currently in ``cache_dir`` to a JSON file."""
    files = _data_files(cache_dir) if os.path.isdir(cache_dir) else []
    with open(snapshot_file, "w") as file:
        json.dump(files, file)


def new_data_files(cache_dir: str, snapshot_file: str) -> list[str] | None:
    """
    Return the data files added since ``snapshot_file`` was written.

    ``None`` when there is no snapshot to compare against.
    """
    try:
        with open(snapshot_file) as file:
            before = set(json.load(file))
    except (OSError, ValueError):
        return None
    current = _data_files(cache_dir) if os.path.isdir(cache_dir) else []
    return [path for path in current if path not in before]


def _by_source(entries: list[IndexEntry]) -> list[tuple[str, int]]:
    """Count entries per source file, most entries first."""
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.source or os.path.relpath(entry.path)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def format_report(
    report: CacheReport, new_files: list[str] | None, max_lines: int = 20
) -> str:
    """
    Render a :class:`CacheReport` for the CI log.

    Parameters
    ----------
    report
        What :func:`scan_cache` found.
    new_files
        Data files added since the snapshot (see :func:`new_data_files`),
        or ``None`` when no snapshot was given.
    max_lines
        Cap on the per-source detail lines.

    Returns
    -------
    str
        Multi-line text.
    """
    n_index = len(report.valid) + len(report.stale) + len(report.orphaned)
    versions = sorted(
        {
            entry.numba_version
            for entry in report.valid + report.stale + report.orphaned
            if entry.numba_version
        }
    )
    lines = [
        f"numba cache: {report.cache_dir}",
        f"  index files: {n_index} ({len(report.valid)} valid, "
        f"{len(report.stale)} stale, {len(report.orphaned)} orphaned)",
        f"  data files:  {report.n_data_files} "
        f"({report.size_bytes / 2**20:.1f} MB), "
        f"numba version in index: {', '.join(versions) or 'n/a'}",
    ]
    if report.stale:
        lines.append("  stale (source stamp changed, will recompile):")
        for source, count in _by_source(report.stale)[:max_lines]:
            lines.append(f"    {count:3d}  {source}")
    if report.orphaned:
        lines.append("  orphaned (no source at recorded path / unreadable):")
        for source, count in _by_source(report.orphaned)[:max_lines]:
            lines.append(f"    {count:3d}  {source}")
    if new_files is not None:
        lines.append(
            f"  compiled during this job: {len(new_files)} kernel(s) "
            "(cache misses, new data files)"
        )
        for path in new_files[:max_lines]:
            lines.append(f"    {path}")
        if len(new_files) > max_lines:
            lines.append(f"    ... and {len(new_files) - max_lines} more")
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    stamp = sub.add_parser("stamp", help="set content-derived mtimes")
    stamp.add_argument("roots", nargs="*", default=["blond"])

    rep = sub.add_parser("report", help="print the state of the cache")
    rep.add_argument("roots", nargs="*", default=["blond"])
    rep.add_argument(
        "--cache-dir",
        default=os.environ.get("NUMBA_CACHE_DIR"),
        help="defaults to $NUMBA_CACHE_DIR",
    )
    rep.add_argument(
        "--snapshot",
        metavar="FILE",
        help="record the current data files to FILE (run before the tests)",
    )
    rep.add_argument(
        "--compare",
        metavar="FILE",
        help="report data files added since the snapshot in FILE",
    )

    args = parser.parse_args()
    if args.command == "stamp":
        changed = stamp_sources(args.roots)
        print(
            f"numba cache: stamped {len(changed)} source file(s) with "
            f"content-derived mtimes under {', '.join(args.roots)}"
        )
        return

    if not args.cache_dir:
        print("numba cache: NUMBA_CACHE_DIR not set, nothing to report")
        return
    new_files = None
    if args.compare:
        new_files = new_data_files(args.cache_dir, args.compare)
    print(format_report(scan_cache(args.cache_dir, args.roots), new_files))
    if args.snapshot:
        write_snapshot(args.cache_dir, args.snapshot)


if __name__ == "__main__":
    _main()
