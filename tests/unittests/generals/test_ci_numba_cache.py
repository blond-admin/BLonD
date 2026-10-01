# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the CI numba cache helper (``dev_tools/ci_numba_cache.py``)."""

import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap

from blond.testing.backend_testing import BLonDTestCase

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so it can run with a bare python3 in CI).
_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "dev_tools",
    "ci_numba_cache.py",
)
_spec = importlib.util.spec_from_file_location("ci_numba_cache", _SCRIPT)
ci_numba_cache = importlib.util.module_from_spec(_spec)
# dataclasses resolve postponed annotations through sys.modules.
sys.modules[_spec.name] = ci_numba_cache
_spec.loader.exec_module(ci_numba_cache)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as file:  # same bytes on Windows
        file.write(text)


def _write_index(
    cache_dir,
    source_file,
    stamp,
    n_overloads=1,
    version="0.63.1",
    qualname="kernel-1.py310",
):
    """Write a numba-format ``.nbi`` index for ``source_file``."""
    subdir = ci_numba_cache.cache_subdir_for(os.path.dirname(source_file))
    module = os.path.splitext(os.path.basename(source_file))[0]
    path = os.path.join(cache_dir, subdir, f"{module}.{qualname}.nbi")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    overloads = {f"sig{i}": f"data{i}" for i in range(n_overloads)}
    with open(path, "wb") as file:
        pickle.dump(version, file)
        file.write(pickle.dumps((stamp, overloads)))
    return path


class TestContentMtime(BLonDTestCase):
    """``content_mtime`` maps file content to a deterministic timestamp."""

    def test_same_content_same_mtime(self):
        self.assertEqual(
            ci_numba_cache.content_mtime(b"x = 1\n"),
            ci_numba_cache.content_mtime(b"x = 1\n"),
        )

    def test_different_content_different_mtime(self):
        self.assertNotEqual(
            ci_numba_cache.content_mtime(b"x = 1\n"),
            ci_numba_cache.content_mtime(b"x = 2\n"),
        )

    def test_is_a_plausible_past_timestamp(self):
        mtime = ci_numba_cache.content_mtime(b"anything")
        self.assertIsInstance(mtime, int)
        self.assertGreaterEqual(mtime, ci_numba_cache.BASE_MTIME)
        self.assertLess(
            mtime, ci_numba_cache.BASE_MTIME + ci_numba_cache.MTIME_RANGE
        )


class TestStampSources(BLonDTestCase):
    """``stamp_sources`` rewrites mtimes of ``*.py`` files below the roots."""

    def test_identical_content_gets_identical_mtime(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "a", "mod.py")
            second = os.path.join(root, "b", "mod.py")
            _write(first, "def f():\n    return 1\n")
            _write(second, "def f():\n    return 1\n")
            os.utime(first, (1_600_000_000, 1_600_000_000))
            os.utime(second, (1_700_000_000, 1_700_000_000))
            ci_numba_cache.stamp_sources([root])
            self.assertEqual(os.stat(first).st_mtime, os.stat(second).st_mtime)
            self.assertEqual(
                os.stat(first).st_mtime,
                ci_numba_cache.content_mtime(b"def f():\n    return 1\n"),
            )

    def test_returns_only_changed_files(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "mod.py")
            _write(path, "x = 1\n")
            changed_first = ci_numba_cache.stamp_sources([root])
            changed_second = ci_numba_cache.stamp_sources([root])
            self.assertEqual(changed_first, [path])
            self.assertEqual(changed_second, [])

    def test_leaves_non_python_files_alone(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "data.txt")
            _write(path, "x")
            os.utime(path, (1_600_000_000, 1_600_000_000))
            ci_numba_cache.stamp_sources([root])
            self.assertEqual(os.stat(path).st_mtime, 1_600_000_000)

    def test_skips_pycache(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "__pycache__", "mod.py")
            _write(path, "x")
            self.assertEqual(ci_numba_cache.stamp_sources([root]), [])


class TestCacheSubdir(BLonDTestCase):
    """``cache_subdir_for`` reproduces numba's cache directory naming."""

    def test_matches_numba_scheme(self):
        source_dir = os.path.abspath(os.path.join("some", "pkg"))
        expected = "pkg_" + hashlib.sha1(source_dir.encode()).hexdigest()
        self.assertEqual(ci_numba_cache.cache_subdir_for(source_dir), expected)


class TestScanCache(BLonDTestCase):
    """``scan_cache`` classifies index files as valid, stale or orphaned."""

    def _setup(self, root):
        cache_dir = os.path.join(root, "cache")
        source_root = os.path.join(root, "src")
        source = os.path.join(source_root, "pkg", "kernels.py")
        _write(source, "def kernel():\n    pass\n")
        ci_numba_cache.stamp_sources([source_root])
        return cache_dir, source_root, source

    def test_valid_entry(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            stamp = ci_numba_cache.source_stamp(source)
            _write_index(cache_dir, source, stamp, n_overloads=3)
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(len(report.valid), 1)
            self.assertEqual(report.stale, [])
            self.assertEqual(report.orphaned, [])
            self.assertEqual(report.valid[0].n_overloads, 3)
            self.assertEqual(report.valid[0].source, source)

    def test_stale_entry_when_mtime_differs(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            mtime, size = ci_numba_cache.source_stamp(source)
            _write_index(cache_dir, source, (mtime + 1.0, size))
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(report.valid, [])
            self.assertEqual(len(report.stale), 1)

    def test_stale_entry_when_size_differs(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            mtime, size = ci_numba_cache.source_stamp(source)
            _write_index(cache_dir, source, (mtime, size + 1))
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(len(report.stale), 1)

    def test_orphaned_entry_without_source(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            gone = os.path.join(source_root, "pkg", "removed.py")
            _write_index(cache_dir, gone, (1.0, 1))
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(len(report.orphaned), 1)

    def test_unreadable_index_is_orphaned_not_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            subdir = ci_numba_cache.cache_subdir_for(os.path.dirname(source))
            _write(os.path.join(cache_dir, subdir, "kernels.k-1.nbi"), "junk")
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(len(report.orphaned), 1)

    def test_counts_data_files_and_size(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, source_root, source = self._setup(root)
            stamp = ci_numba_cache.source_stamp(source)
            index = _write_index(cache_dir, source, stamp)
            data = index[: -len(".nbi")] + ".1.nbc"
            with open(data, "wb") as file:
                file.write(b"\0" * 1000)
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            self.assertEqual(report.n_data_files, 1)
            self.assertGreaterEqual(report.size_bytes, 1000)

    def test_missing_cache_dir_gives_empty_report(self):
        with tempfile.TemporaryDirectory() as root:
            report = ci_numba_cache.scan_cache(
                os.path.join(root, "nope"), [root]
            )
            self.assertEqual(report.valid, [])
            self.assertEqual(report.n_data_files, 0)


class TestSnapshot(BLonDTestCase):
    """Snapshots tell which data files appeared since before the job."""

    def test_new_data_files_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, "cache")
            old = os.path.join(cache_dir, "d", "k.1.nbc")
            _write(old, "x")
            snapshot_file = os.path.join(root, "snap.json")
            ci_numba_cache.write_snapshot(cache_dir, snapshot_file)
            with open(snapshot_file) as file:
                self.assertEqual(len(json.load(file)), 1)
            new = os.path.join(cache_dir, "d", "k.2.nbc")
            _write(new, "y")
            added = ci_numba_cache.new_data_files(cache_dir, snapshot_file)
            self.assertEqual(added, [os.path.join("d", "k.2.nbc")])

    def test_missing_snapshot_gives_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(
                ci_numba_cache.new_data_files(
                    root, os.path.join(root, "missing.json")
                )
            )


class TestFormatReport(BLonDTestCase):
    """``format_report`` renders the counts a CI log reader wants."""

    def test_mentions_counts(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, "cache")
            source_root = os.path.join(root, "src")
            source = os.path.join(source_root, "pkg", "kernels.py")
            _write(source, "def kernel():\n    pass\n")
            ci_numba_cache.stamp_sources([source_root])
            stamp = ci_numba_cache.source_stamp(source)
            _write_index(cache_dir, source, stamp, qualname="a-1.py310")
            _write_index(
                cache_dir,
                source,
                (stamp[0] + 1, stamp[1]),
                qualname="b-2.py310",
            )
            report = ci_numba_cache.scan_cache(cache_dir, [source_root])
            text = ci_numba_cache.format_report(report, new_files=["x.nbc"])
            self.assertIn("1 valid", text)
            self.assertIn("1 stale", text)
            self.assertIn("kernels.py", text)
            self.assertIn("compiled during this job: 1", text)

    def test_without_snapshot_no_compiled_line(self):
        with tempfile.TemporaryDirectory() as root:
            report = ci_numba_cache.scan_cache(root, [root])
            text = ci_numba_cache.format_report(report, new_files=None)
            self.assertNotIn("compiled during this job", text)


class TestEndToEndWithNumba(BLonDTestCase):
    """A real numba kernel is served from cache after a stamped re-checkout."""

    _MODULE = textwrap.dedent(
        """
        from numba import njit

        @njit(cache=True)
        def add_one(x):
            return x + 1
        """
    )

    def _run(self, root, cache_dir):
        """Import the module in a subprocess and tell if it hit the cache."""
        code = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {root!r})
            import mod
            start = time.perf_counter()
            mod.add_one(1)
            print(time.perf_counter() - start)
            """
        )
        env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip().splitlines()[-1])

    def test_cache_hits_after_checkout_like_mtime_change(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, "numba_cache")
            source = os.path.join(root, "mod.py")
            _write(source, self._MODULE)
            ci_numba_cache.stamp_sources([root])
            self._run(root, cache_dir)  # compiles and fills the cache
            report = ci_numba_cache.scan_cache(cache_dir, [root])
            self.assertEqual(len(report.valid), 1, report)

            # Simulate a fresh checkout: same content, new mtime.
            _write(source, self._MODULE)
            os.utime(source, None)
            report = ci_numba_cache.scan_cache(cache_dir, [root])
            self.assertEqual(len(report.stale), 1, report)

            ci_numba_cache.stamp_sources([root])
            report = ci_numba_cache.scan_cache(cache_dir, [root])
            self.assertEqual(len(report.valid), 1, report)
            self.assertLess(self._run(root, cache_dir), 0.5)
