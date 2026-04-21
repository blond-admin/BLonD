# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the LXPlus HTCondor submission helpers."""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pytest

from blond.specifics.cern.lxplus import submission
from blond.specifics.cern.lxplus.submission import (
    _build_submission_command,
    _parse_cluster_id,
    get_eos_target,
    is_on_htcondor,
    move_results_to_eos,
    save_args,
    send_results_to_host,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Layer 1: pure-function unit tests
# ---------------------------------------------------------------------------


class TestParseClusterId:
    def test_standard_output(self):
        out = "Submitting job(s).\n1 job(s) submitted to cluster 12345."
        assert _parse_cluster_id(out) == "12345"

    def test_multiple_lines(self):
        out = "garbage\n1 job(s) submitted to cluster 99."
        assert _parse_cluster_id(out) == "99"

    def test_missing_raises(self):
        with pytest.raises(RuntimeError, match="Could not parse cluster ID"):
            _parse_cluster_id("something else entirely")


class TestOnHtcondor:
    def test_true_when_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BLOND_JOB_TMPDIR", str(tmp_path))
        assert is_on_htcondor() is True

    def test_false_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("BLOND_JOB_TMPDIR", raising=False)
        assert is_on_htcondor() is False


class TestGetEosTarget:
    def test_inside_job(self, monkeypatch, tmp_path):
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setenv("BLOND_JOB_TMPDIR", "/afs/.../job_abc123")
        target = get_eos_target(tmp_path / "results")
        assert target == "/eos/user/a/alice/blond_results/job_abc123/results"

    def test_fallback_to_local(self, monkeypatch, tmp_path):
        monkeypatch.setenv("USER", "bob")
        monkeypatch.delenv("BLOND_JOB_TMPDIR", raising=False)
        target = get_eos_target(tmp_path / "out")
        assert target == "/eos/user/b/bob/blond_results/local/out"


# ---------------------------------------------------------------------------
# Layer 2: side-effecting helpers with mocks / tmp_path
# ---------------------------------------------------------------------------


class TestSaveArgs:
    def test_writes_json(self, tmp_path):
        args = argparse.Namespace(count=3, name="demo")
        save_args(args, target_dir=tmp_path)
        loaded = json.loads((tmp_path / "args.json").read_text())
        assert loaded == {"count": 3, "name": "demo"}

    def test_creates_missing_dir(self, tmp_path):
        target = tmp_path / "new" / "dir"
        save_args(argparse.Namespace(x=1), target_dir=target)
        assert (target / "args.json").exists()


class TestWriteManifest:
    def test_manifest_contains_env_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLOND_JOB_SUBMITTED_AT", "2026-04-20T10:00:00Z")
        monkeypatch.setenv("BLOND_JOB_COMMIT", "deadbeef")
        monkeypatch.setenv(
            "BLOND_JOB_REMOTE_URL", "https://gitlab.cern.ch/u/p"
        )
        monkeypatch.setenv("BLOND_JOB_TMPDIR", "/afs/.../job_xyz")
        monkeypatch.setenv("USER", "alice")

        write_manifest(tmp_path)

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["submitted_at"] == "2026-04-20T10:00:00Z"
        assert manifest["commit"] == "deadbeef"
        assert manifest["remote_url"] == "https://gitlab.cern.ch/u/p"
        assert manifest["job_id"] == "job_xyz"
        assert manifest["user"] == "alice"
        assert "started_at" in manifest
        assert "hostname" in manifest
        assert "python_version" in manifest
        assert "argv" in manifest


class TestSetResult:
    def test_noop_off_htcondor(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BLOND_JOB_TMPDIR", raising=False)
        send_results_to_host(
            {"dt": 0.4e-6}
        )  # Should not raise, not write anywhere.

    def test_writes_json_for_scalar(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLOND_JOB_TMPDIR", str(tmp_path))
        send_results_to_host(0.4e-6)
        assert json.loads((tmp_path / "blond_result.json").read_text()) == 4e-7

    def test_writes_json_for_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLOND_JOB_TMPDIR", str(tmp_path))
        payload = {"dt": 0.4e-6, "dE": 25e6}
        send_results_to_host(payload)
        assert (
            json.loads((tmp_path / "blond_result.json").read_text()) == payload
        )

    def test_writes_npy_for_ndarray(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLOND_JOB_TMPDIR", str(tmp_path))
        arr = np.array([1.0, 2.0, 3.0])
        send_results_to_host(arr)
        loaded = np.load(tmp_path / "blond_result.npy")
        np.testing.assert_array_equal(loaded, arr)


class TestResultsToEos:
    def test_invokes_eos_cp_with_mgm_url(self, tmp_path, monkeypatch):
        src = tmp_path / "results"
        src.mkdir()
        (src / "out.txt").write_text("hello")

        monkeypatch.setenv("USER", "alice")
        monkeypatch.setenv("BLOND_JOB_TMPDIR", "/afs/.../job_abc")

        calls = []

        def fake_run(cmd, check=True, env=None, **kwargs):
            calls.append((list(cmd), env or {}))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(submission.subprocess, "run", fake_run)

        result = move_results_to_eos(src, verbose=False)

        assert result == "/eos/user/a/alice/blond_results/job_abc/results"
        # Two subprocess calls: mkdir -p, then cp -r.
        assert len(calls) == 2
        mkdir_cmd, mkdir_env = calls[0]
        cp_cmd, cp_env = calls[1]

        assert mkdir_cmd[:3] == ["eos", "mkdir", "-p"]
        assert mkdir_env.get("EOS_MGM_URL") == "root://eosuser.cern.ch"
        assert cp_cmd[:3] == ["eos", "cp", "-r"]
        # Destination is the parent (ending in '/') to avoid the nesting quirk.
        assert cp_cmd[-1].endswith("/blond_results/job_abc/")
        assert cp_env.get("EOS_MGM_URL") == "root://eosuser.cern.ch"

    def test_raises_on_basename_mismatch(self, tmp_path, monkeypatch):
        src = tmp_path / "results"
        src.mkdir()
        monkeypatch.setattr(
            submission.subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 0),
        )
        with pytest.raises(ValueError, match="basename"):
            move_results_to_eos(
                src, target_eos="/eos/user/a/alice/different_name"
            )

    def test_raises_when_source_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            move_results_to_eos(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Layer 3: snapshot / shell-validity tests for the generated submission
# ---------------------------------------------------------------------------


def _extract_heredoc(script: str, marker: str) -> str:
    """Extract the body of ``cat > ... << 'MARKER' ... MARKER``."""
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if f"<< '{marker}'" in line)
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].strip() == marker
    )
    return "\n".join(lines[start + 1 : end])


@pytest.fixture
def base_build_kwargs():
    return dict(
        remote_workdir="/afs/cern.ch/user/a/alice/blond_jobs/job_abc123",
        remote_url="https://gitlab.cern.ch/alice/proj",
        commit="deadbeef1234",
        script_rel="proj/main.py",
        kwargs={"count": 1, "label": "hello world"},
        python="python3.11",
        job_flavour="espresso",
        accounting_group="group_u_BE.ABP.normal",
    )


class TestBuildSubmissionCommand:
    def test_is_syntactically_valid_bash(self, tmp_path, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        script_path = tmp_path / "submit.sh"
        script_path.write_text(script)
        # Catches: unterminated heredocs, bad quoting, stray $(), etc.
        subprocess.run(
            ["bash", "-n", str(script_path)], check=True, capture_output=True
        )

    def test_wrapper_is_syntactically_valid_bash(
        self, tmp_path, base_build_kwargs
    ):
        script = _build_submission_command(**base_build_kwargs)
        wrapper = _extract_heredoc(script, "WRAPPER_EOF")
        wrapper_path = tmp_path / "wrapper.sh"
        wrapper_path.write_text(wrapper)
        subprocess.run(
            ["bash", "-n", str(wrapper_path)], check=True, capture_output=True
        )

    def test_accounting_group_is_quoted(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        assert '+AccountingGroup      = "group_u_BE.ABP.normal"' in script

    def test_accounting_group_omitted_when_none(self, base_build_kwargs):
        base_build_kwargs["accounting_group"] = None
        script = _build_submission_command(**base_build_kwargs)
        assert "+AccountingGroup" not in script

    def test_request_gpus_absent_by_default(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        assert "request_gpus" not in script

    def test_request_gpus_present_when_set(self, base_build_kwargs):
        base_build_kwargs["request_gpus"] = 2
        script = _build_submission_command(**base_build_kwargs)
        assert "request_gpus          = 2" in script

    def test_provenance_env_vars_are_exported(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        wrapper = _extract_heredoc(script, "WRAPPER_EOF")
        assert 'export BLOND_JOB_COMMIT="deadbeef1234"' in wrapper
        assert (
            'export BLOND_JOB_REMOTE_URL="https://gitlab.cern.ch/alice/proj"'
            in wrapper
        )
        assert "export BLOND_JOB_SUBMITTED_AT=" in wrapper
        assert 'export BLOND_JOB_TMPDIR="/afs/cern.ch/user/a/alice' in wrapper

    def test_args_are_written_as_json(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        args_body = _extract_heredoc(script, "ARGS_EOF")
        assert json.loads(args_body) == base_build_kwargs["kwargs"]

    def test_script_is_invoked_without_cli_args(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        wrapper = _extract_heredoc(script, "WRAPPER_EOF")
        assert (
            '"$SCRATCH/venv/bin/python" "$SCRATCH/repo/proj/main.py"\n'
            in wrapper + "\n"
        )
        assert "--label" not in wrapper
        assert "--count" not in wrapper

    def test_job_flavour_line_present(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        assert "+JobFlavour           = espresso" in script

    def test_commit_is_checked_out(self, base_build_kwargs):
        script = _build_submission_command(**base_build_kwargs)
        wrapper = _extract_heredoc(script, "WRAPPER_EOF")
        assert "git -C \"$SCRATCH/repo\" checkout --quiet 'deadbeef1234'" in (
            wrapper
        )
