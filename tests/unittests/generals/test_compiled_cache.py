# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the bounded LRU housekeeping of compiled-backend directories."""

import json

from blond.generals import compiled_cache


def _meta(directory):
    with open(directory / "meta.json", encoding="utf-8") as file:
        return json.load(file)


class TestMarkUsed:
    def test_creates_built_at_and_last_used(self, tmp_path):
        compiled_cache.mark_used(str(tmp_path))
        meta = _meta(tmp_path)
        assert "built_at" in meta
        assert "last_used" in meta

    def test_built_at_preserved_last_used_refreshed(self, tmp_path):
        compiled_cache.mark_used(str(tmp_path))
        first = _meta(tmp_path)
        compiled_cache.mark_used(str(tmp_path))
        second = _meta(tmp_path)
        assert second["built_at"] == first["built_at"]
        assert second["last_used"] >= first["last_used"]

    def test_never_raises_on_bad_path(self):
        # A non-existent directory must not raise (best-effort bookkeeping).
        compiled_cache.mark_used("/nonexistent/dir/xyz")

    def test_corrupt_meta_is_replaced(self, tmp_path):
        # A garbled meta.json must not crash: it is discarded and rewritten
        # with fresh, valid stamps.
        (tmp_path / "meta.json").write_text("{ this is not json")
        compiled_cache.mark_used(str(tmp_path))
        meta = _meta(tmp_path)
        assert "built_at" in meta
        assert "last_used" in meta


class TestLastUsed:
    def test_missing_dir_sorts_oldest(self):
        # Neither meta.json nor an mtime is available -> the fallback key 0.0
        # makes such (e.g. vanished) directories sort oldest.
        assert compiled_cache._last_used("/nonexistent/dir/xyz") == 0.0

    def test_falls_back_to_mtime_without_meta(self, tmp_path):
        # No meta.json -> the directory mtime is used as the key.
        d = tmp_path / "nometa"
        d.mkdir()
        import os

        assert compiled_cache._last_used(str(d)) == os.path.getmtime(str(d))


class TestPrune:
    def _make_dirs(self, root, n):
        dirs = []
        for i in range(n):
            d = root / f"hash{i:02d}"
            d.mkdir()
            (d / "meta.json").write_text(
                json.dumps(
                    {
                        "built_at": "x",
                        "last_used": f"2026-01-0{i + 1}T00:00:00",
                    }
                )
            )
            dirs.append(d)
        return dirs

    def test_keeps_active_plus_most_recently_used_siblings(self, tmp_path):
        # hash00 oldest ... hash04 newest. Prune around hash04 (the active
        # one) with keep_n=2 -> active + 1 freshest sibling survive.
        self._make_dirs(tmp_path, 5)
        compiled_cache.prune_siblings(str(tmp_path / "hash04"), keep_n=2)
        remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert remaining == ["hash03", "hash04"]

    def test_active_dir_never_evicted_even_when_oldest(self, tmp_path):
        # hash00 is the *oldest* but is the active dir -> must survive; the
        # single freshest sibling is the only other kept (keep_n=2).
        self._make_dirs(tmp_path, 5)
        compiled_cache.prune_siblings(str(tmp_path / "hash00"), keep_n=2)
        remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert "hash00" in remaining  # active, oldest, still kept
        assert remaining == ["hash00", "hash04"]

    def test_noop_when_under_limit(self, tmp_path):
        self._make_dirs(tmp_path, 3)
        compiled_cache.prune_siblings(str(tmp_path / "hash00"), keep_n=20)
        assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 3

    def test_unstamped_siblings_evicted_first(self, tmp_path):
        # Legacy siblings without meta.json sort oldest -> removed first,
        # while the active dir is always retained.
        active = tmp_path / "active"
        active.mkdir()
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        compiled_cache.prune_siblings(str(active), keep_n=1)
        assert active.exists()
        assert not legacy.exists()

    def test_never_raises_on_missing_dir(self):
        compiled_cache.prune_siblings("/nonexistent/root/active", keep_n=5)
