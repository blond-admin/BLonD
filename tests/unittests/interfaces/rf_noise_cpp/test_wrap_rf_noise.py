# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the ctypes binding in ``wrap_rf_noise``.

These exercise the interop/build error-handling branches without requiring
the external ``rf-noise-cpp`` library: the filesystem checks and the
compiler invocation are driven with temporary trees and a stubbed
``subprocess.run``.
"""

import pathlib

import numpy as np
import pytest

from blond.interfaces.rf_noise_cpp import wrap_rf_noise
from blond.interfaces.rf_noise_cpp.wrap_rf_noise import (
    RF_NOISE_REPO_URL,
    _cast_double,
    _compile_rf_noise_library,
    _get_rf_noise_dir,
    _install_hint,
    rf_noise_library_available,
)


class _StubCompletedProcess:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def _make_rf_noise_tree(
    root: pathlib.Path, *, with_sources: bool = True
) -> pathlib.Path:
    """Build a minimal ``rf-noise-cpp`` checkout under ``root``.

    Returns the path that ends in ``rf-noise-cpp`` (the value
    ``_compile_rf_noise_library`` expects).
    """
    rf_noise_dir = root / "rf-noise-cpp"
    src = rf_noise_dir / "src" / "rf-noise"
    src.mkdir(parents=True)
    if with_sources:
        (src / "dummy.cpp").write_text("// dummy source\n")
    return rf_noise_dir


# --------------------------------------------------------------------------- #
# _install_hint
# --------------------------------------------------------------------------- #
def test_install_hint_mentions_repo_and_env_var():
    hint = _install_hint()
    assert RF_NOISE_REPO_URL in hint
    assert "RF_NOISE_DIR" in hint


# --------------------------------------------------------------------------- #
# _get_rf_noise_dir
# --------------------------------------------------------------------------- #
def test_get_rf_noise_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RF_NOISE_DIR", str(tmp_path))
    assert _get_rf_noise_dir() == tmp_path.resolve()


def test_get_rf_noise_dir_env_not_a_dir_raises(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setenv("RF_NOISE_DIR", str(missing))
    with pytest.raises(AssertionError):
        _get_rf_noise_dir()


def test_get_rf_noise_dir_sibling_fallback(monkeypatch, tmp_path):
    # Without RF_NOISE_DIR the source is assumed to be a sibling of the BLonD
    # repository: ``files("blond").parent.parent / "rf-noise-cpp"``.
    monkeypatch.delenv("RF_NOISE_DIR", raising=False)
    sibling = tmp_path / "rf-noise-cpp"
    sibling.mkdir()
    fake_blond_pkg = tmp_path / "BLonD" / "blond"
    monkeypatch.setattr(wrap_rf_noise, "files", lambda _pkg: fake_blond_pkg)

    assert _get_rf_noise_dir() == sibling


# --------------------------------------------------------------------------- #
# _compile_rf_noise_library
# --------------------------------------------------------------------------- #
def test_compile_wrong_dir_name_raises(tmp_path):
    with pytest.raises(NameError):
        _compile_rf_noise_library(
            rf_noise_dir=tmp_path / "not-the-right-name",
            target_library=tmp_path / "lib.so",
        )


def test_compile_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _compile_rf_noise_library(
            rf_noise_dir=tmp_path / "rf-noise-cpp",  # never created
            target_library=tmp_path / "lib.so",
        )


def test_compile_missing_sources_dir_raises(tmp_path):
    rf_noise_dir = tmp_path / "rf-noise-cpp"
    rf_noise_dir.mkdir()  # exists, but has no src/rf-noise
    with pytest.raises(FileNotFoundError):
        _compile_rf_noise_library(
            rf_noise_dir=rf_noise_dir,
            target_library=tmp_path / "lib.so",
        )


def test_compile_no_cpp_files_raises(tmp_path):
    rf_noise_dir = _make_rf_noise_tree(tmp_path, with_sources=False)
    with pytest.raises(FileNotFoundError):
        _compile_rf_noise_library(
            rf_noise_dir=rf_noise_dir,
            target_library=tmp_path / "lib.so",
        )


def test_compile_removes_stale_library_on_success(monkeypatch, tmp_path):
    rf_noise_dir = _make_rf_noise_tree(tmp_path)
    target_library = tmp_path / "lib.so"
    target_library.write_text("stale")  # pre-existing, must be removed first

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _StubCompletedProcess(returncode=0)

    monkeypatch.setattr(wrap_rf_noise.subprocess, "run", fake_run)

    # Returns None on success; the stale library was unlinked before building.
    assert (
        _compile_rf_noise_library(
            rf_noise_dir=rf_noise_dir, target_library=target_library
        )
        is None
    )
    assert calls, "compiler was not invoked"


def test_compile_missing_compiler_raises(monkeypatch, tmp_path):
    rf_noise_dir = _make_rf_noise_tree(tmp_path)

    def fake_run(command, **kwargs):
        raise FileNotFoundError("g++ not found")

    monkeypatch.setattr(wrap_rf_noise.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="g\\+\\+"):
        _compile_rf_noise_library(
            rf_noise_dir=rf_noise_dir,
            target_library=tmp_path / "lib.so",
        )


def test_compile_nonzero_returncode_raises_and_cleans_up(
    monkeypatch, tmp_path
):
    rf_noise_dir = _make_rf_noise_tree(tmp_path)
    target_library = tmp_path / "lib.so"
    target_library.write_text("partial")  # simulate a stale/partial artefact

    def fake_run(command, **kwargs):
        return _StubCompletedProcess(returncode=1, stderr="boom")

    monkeypatch.setattr(wrap_rf_noise.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="failed"):
        _compile_rf_noise_library(
            rf_noise_dir=rf_noise_dir, target_library=target_library
        )
    # The partial output is cleaned up on failure.
    assert not target_library.exists()


# --------------------------------------------------------------------------- #
# rf_noise_library_available
# --------------------------------------------------------------------------- #
def test_library_available_false_on_load_error(monkeypatch):
    def fake_load():
        raise FileNotFoundError("no library")

    monkeypatch.setattr(wrap_rf_noise, "_load_rf_noise", fake_load)
    assert rf_noise_library_available() is False


# --------------------------------------------------------------------------- #
# _cast_double
# --------------------------------------------------------------------------- #
def test_cast_double_warns_on_wrong_dtype():
    arr = np.array([1, 2, 3], dtype=np.float32)
    with pytest.warns(UserWarning):
        out = _cast_double(arr)
    assert out.dtype == np.double


def test_cast_double_no_warning_for_double():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.double)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        out = _cast_double(arr)
    assert out.dtype == np.double
