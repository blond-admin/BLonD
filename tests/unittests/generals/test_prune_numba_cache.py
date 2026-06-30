# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the numba-cache size limiter (``dev_tools/prune_numba_cache.py``)."""

import importlib.util
import os
import time

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so it can run with a bare python3 in CI).
_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "dev_tools",
    "prune_numba_cache.py",
)
_spec = importlib.util.spec_from_file_location("prune_numba_cache", _SCRIPT)
prune_numba_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune_numba_cache)


def _write(path, size_bytes, mtime):
    with open(path, "wb") as file:
        file.write(b"\0" * size_bytes)
    os.utime(path, (mtime, mtime))


def _total(directory):
    return sum(
        os.path.getsize(os.path.join(r, n))
        for r, _d, names in os.walk(directory)
        for n in names
    )


class TestPruneNumbaCache:
    def test_noop_under_budget(self, tmp_path):
        _write(str(tmp_path / "a.nbc"), 1024, time.time())
        prune_numba_cache.prune(str(tmp_path), max_mb=1)
        assert (tmp_path / "a.nbc").exists()

    def test_evicts_oldest_first_until_under_budget(self, tmp_path):
        now = time.time()
        # Three 1 MiB files, distinct ages: old < mid < new.
        _write(str(tmp_path / "old.nbc"), 1024 * 1024, now - 300)
        _write(str(tmp_path / "mid.nbc"), 1024 * 1024, now - 200)
        _write(str(tmp_path / "new.nbc"), 1024 * 1024, now - 100)
        # Budget for ~2 files -> the oldest must go, the two newest survive.
        prune_numba_cache.prune(str(tmp_path), max_mb=2.5)
        assert not (tmp_path / "old.nbc").exists()
        assert (tmp_path / "mid.nbc").exists()
        assert (tmp_path / "new.nbc").exists()
        assert _total(str(tmp_path)) <= int(2.5 * 1024 * 1024)

    def test_recurses_into_subdirs(self, tmp_path):
        # numba nests cache files under per-module subdirectories.
        sub = tmp_path / "numba_abc123"
        sub.mkdir()
        now = time.time()
        _write(str(sub / "old.nbc"), 1024 * 1024, now - 100)
        _write(str(sub / "new.nbc"), 1024 * 1024, now)
        prune_numba_cache.prune(str(tmp_path), max_mb=1.5)
        assert not (sub / "old.nbc").exists()
        assert (sub / "new.nbc").exists()

    def test_missing_dir_is_noop(self):
        prune_numba_cache.prune("/nonexistent/numba/cache")  # must not raise

    def test_none_dir_is_noop(self):
        prune_numba_cache.prune(None)  # must not raise
