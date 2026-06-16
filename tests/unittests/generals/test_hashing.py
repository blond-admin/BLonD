# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the build-environment-aware hashing used by the cache keys."""

import shutil

import pytest

from blond.generals.hashing_ import hash_build_target


def _write_sources(folder):
    (folder / "a.py").write_text("print('hello')\n")
    (folder / "b.cpp").write_text("int main() { return 0; }\n")


class TestHashBuildTarget:
    def test_deterministic(self, tmp_path):
        _write_sources(tmp_path)
        h1 = hash_build_target(tmp_path, (".py", ".cpp"))
        h2 = hash_build_target(tmp_path, (".py", ".cpp"))
        assert h1 == h2

    def test_location_independent(self, tmp_path):
        # Identical sources in differently-named folders must hash equal:
        # the checkout path must not leak into the key (else a CI runner and
        # a local clone get different keys and never share a cache).
        a = tmp_path / "locationA"
        b = tmp_path / "a_totally_different_name"
        for d in (a, b):
            d.mkdir()
            _write_sources(d)
        assert hash_build_target(a, (".py", ".cpp")) == hash_build_target(
            b, (".py", ".cpp")
        )

    def test_source_change_changes_hash(self, tmp_path):
        _write_sources(tmp_path)
        before = hash_build_target(tmp_path, (".py", ".cpp"))
        (tmp_path / "a.py").write_text("print('changed')\n")
        assert hash_build_target(tmp_path, (".py", ".cpp")) != before

    def test_probe_output_changes_hash(self, tmp_path):
        # Different toolchain probe output -> different digest, so a binary
        # built for one toolchain is never located under another's key.
        _write_sources(tmp_path)
        base = hash_build_target(tmp_path, (".py", ".cpp"))
        with_probe = hash_build_target(
            tmp_path,
            (".py", ".cpp"),
            probe_commands=[["python", "--version"]],
        )
        assert with_probe != base

    def test_extra_changes_hash(self, tmp_path):
        _write_sources(tmp_path)
        a = hash_build_target(tmp_path, (".py", ".cpp"), extra=("sm_70",))
        b = hash_build_target(tmp_path, (".py", ".cpp"), extra=("sm_80",))
        assert a != b

    def test_missing_probe_is_safe(self, tmp_path):
        # An uncallable probe must not raise; it folds its error text in and
        # still yields a stable digest (forcing a rebuild at worst).
        _write_sources(tmp_path)
        h1 = hash_build_target(
            tmp_path,
            (".py", ".cpp"),
            probe_commands=[["definitely-not-a-real-binary-xyz"]],
        )
        h2 = hash_build_target(
            tmp_path,
            (".py", ".cpp"),
            probe_commands=[["definitely-not-a-real-binary-xyz"]],
        )
        assert h1 == h2


class TestCppCacheRendezvous:
    """The compiler and the loader must agree on the compiled directory."""

    def test_compile_and_load_dirs_match(self):
        import os

        from blond.core.backends.cpp import compiled_dir_handler as lc

        folder = os.path.dirname(os.path.abspath(lc.__file__))
        # Loader calls with the default compiler; the compiler passes its
        # (default) compiler argument explicitly. They must coincide.
        assert lc.cpp_compiled_dir(folder) == lc.cpp_compiled_dir(
            folder, compiler=lc.DEFAULT_COMPILER
        )

    @pytest.mark.skipif(
        shutil.which("g++") is None or shutil.which("gcc") is None,
        reason="needs both g++ and gcc to compare toolchains",
    )
    def test_different_compiler_changes_dir(self):
        import os

        from blond.core.backends.cpp import compiled_dir_handler as lc

        folder = os.path.dirname(os.path.abspath(lc.__file__))
        assert lc.cpp_compiled_dir(
            folder, compiler="g++"
        ) != lc.cpp_compiled_dir(folder, compiler="gcc")

    def test_caller_flags_change_dir(self):
        # Caller-supplied build parameters must land in a distinct directory
        # so a custom-flag binary never collides with the default one.
        import os

        from blond.core.backends.cpp import compiled_dir_handler as lc

        folder = os.path.dirname(os.path.abspath(lc.__file__))
        default = lc.cpp_compiled_dir(folder)
        assert lc.cpp_compiled_dir(folder, flags="-DFOO") != default
        assert lc.cpp_compiled_dir(folder, libs="-lm") != default
        assert lc.cpp_compiled_dir(folder, optimize=False) != default
        assert lc.cpp_compiled_dir(folder, with_fftw=True) != default

    def test_memoised(self):
        # The toolchain probes spawn subprocesses; repeated load-side calls
        # must hit the cache rather than re-probe.
        import os

        from blond.core.backends.cpp import compiled_dir_handler as lc

        folder = os.path.dirname(os.path.abspath(lc.__file__))
        lc.cpp_compiled_dir(folder)
        before = lc.cpp_compiled_dir.cache_info()
        lc.cpp_compiled_dir(folder)
        after = lc.cpp_compiled_dir.cache_info()
        assert after.hits == before.hits + 1
