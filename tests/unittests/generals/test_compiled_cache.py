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


class TestPrune:
    def _make_dirs(self, root, n):
        dirs = []
        for i in range(n):
            d = root / f"hash{i:02d}"
            d.mkdir()
            dirs.append(d)
        return dirs

    def test_keeps_only_most_recently_used(self, tmp_path):
        dirs = self._make_dirs(tmp_path, 5)
        # Stamp last_used in a known order: hash00 oldest ... hash04 newest.
        for i, d in enumerate(dirs):
            (d / "meta.json").write_text(
                json.dumps(
                    {
                        "built_at": "x",
                        "last_used": f"2026-01-0{i + 1}T00:00:00",
                    }
                )
            )
        compiled_cache.prune(str(tmp_path), keep=2)
        remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert remaining == ["hash03", "hash04"]  # the two newest

    def test_noop_when_under_limit(self, tmp_path):
        self._make_dirs(tmp_path, 3)
        compiled_cache.prune(str(tmp_path), keep=20)
        assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 3

    def test_unstamped_dirs_evicted_first(self, tmp_path):
        # Legacy directories without meta.json sort oldest -> removed first.
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        stamped = tmp_path / "stamped"
        stamped.mkdir()
        (stamped / "meta.json").write_text(
            json.dumps({"built_at": "x", "last_used": "2099-01-01T00:00:00"})
        )
        compiled_cache.prune(str(tmp_path), keep=1)
        assert stamped.exists()
        assert not legacy.exists()

    def test_never_raises_on_missing_root(self):
        compiled_cache.prune("/nonexistent/root/xyz", keep=5)
