# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the build-environment-aware hashing used by the cache keys."""

import shutil
from unittest import mock

import pytest

from blond.generals.hashing_ import (
    hash_build_target,
    hash_files,
    hash_in_folder,
)


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

    def test_defaults_match_compile_cpp_library(self):
        # The loader computes the cache dir from cpp_compiled_dir's *default*
        # build parameters; the compiler passes compile_cpp_library's. If a
        # default drifts between the two, a default build would silently land
        # in a directory the loader never looks in. Guard against that.
        import inspect

        from blond.core.backends.cpp import compiled_dir_handler as lc
        from blond.core.backends.cpp.compile import compile_cpp_library

        dir_params = inspect.signature(lc.cpp_compiled_dir).parameters
        build_params = inspect.signature(compile_cpp_library).parameters
        shared = [
            "compiler",
            "optimize",
            "flags",
            "libs",
            "with_fftw",
            "with_fftw_threads",
            "with_fftw_omp",
            "with_fftw_lib",
            "with_fftw_header",
            "boost",
        ]
        for name in shared:
            assert dir_params[name].default == build_params[name].default, (
                f"default for {name!r} drifted between cpp_compiled_dir and "
                f"compile_cpp_library; the default build would no longer "
                f"rendezvous with the loader"
            )

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


class TestHashFilesNameHandling:
    """Name folded into the digest: relative when given a base, else absolute."""

    def test_without_base_folder_absolute_path_leaks(self, tmp_path):
        # Default (base_folder=None): the full path is folded in, so identical
        # content at two different locations hashes differently.
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.py").write_text("same\n")
        (b / "x.py").write_text("same\n")
        assert hash_files([str(a / "x.py")]) != hash_files([str(b / "x.py")])

    def test_base_folder_makes_it_location_independent(self, tmp_path):
        # With base_folder, only the relative name is folded in -> identical
        # content at the same relative path hashes equally.
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.py").write_text("same\n")
        (b / "x.py").write_text("same\n")
        assert hash_files([str(a / "x.py")], base_folder=str(a)) == hash_files(
            [str(b / "x.py")], base_folder=str(b)
        )


class TestHashInFolderWindows:
    def test_windows_lowercases_paths_and_base(self, tmp_path):
        # On Windows the file paths AND the base folder are lower-cased before
        # being folded into the digest (the filesystem is case-insensitive).
        # Verify that lowercasing without depending on the filesystem: mock
        # hash_files to capture what hash_in_folder hands it. (Forcing the real
        # code path on a case-sensitive CI would fail, because it also *opens*
        # the lowercased path -- which only resolves on real Windows.)
        sub = tmp_path / "MixedCaseDir"
        sub.mkdir()
        (sub / "Kernel.py").write_text("x = 1\n")

        captured = {}

        def fake_hash_files(paths, base_folder=None):
            captured["paths"] = list(paths)
            captured["base_folder"] = base_folder
            return "0" * 64

        with (
            mock.patch("platform.system", return_value="Windows"),
            mock.patch("blond.generals.hashing_.hash_files", fake_hash_files),
        ):
            hash_in_folder(str(sub), (".py",))

        assert captured["paths"] == [p.lower() for p in captured["paths"]]
        assert captured["base_folder"] == captured["base_folder"].lower()
        # not vacuously true: the mixed-case input really was lower-cased
        assert captured["paths"][0].endswith("kernel.py")
