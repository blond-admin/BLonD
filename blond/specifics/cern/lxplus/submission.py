# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Submit BLonD simulation scripts to HTCondor on LXPlus."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

LXPLUS_HOST = "lxplus.cern.ch"
_RESULT_JSON = "blond_result.json"
_RESULT_NPY = "blond_result.npy"
_ENV_JOB_TMPDIR = "BLOND_JOB_TMPDIR"


def set_result(value: Any) -> None:
    """Write a result value from within a batch job.

    Serialises *value* to the job's result directory so that
    :meth:`LxplusJob.wait` can retrieve it after the job finishes.
    When called outside a batch job (e.g. during local development)
    this function is a no-op.

    Parameters
    ----------
    value
        The result to communicate back to the caller:

        * ``float`` / ``int`` / ``dict`` → saved as ``blond_result.json``
        * :class:`numpy.ndarray` → saved as ``blond_result.npy``

    Examples
    --------
    >>> set_result(0.4e-6)                        # float
    >>> set_result({'dt': 0.4e-6, 'dE': 25e6})   # dict
    >>> set_result(obs.dts[-1])                   # 1-D ndarray
    """
    tmpdir = os.environ.get(_ENV_JOB_TMPDIR)
    if tmpdir is None:
        return
    if isinstance(value, np.ndarray):
        np.save(os.path.join(tmpdir, _RESULT_NPY), value)
    else:
        with open(os.path.join(tmpdir, _RESULT_JSON), "w") as f:
            json.dump(value, f)


class LxplusJob:
    """Handle for a job submitted to HTCondor on LXPlus.

    Instances are returned by :func:`run_on_lxplus`; callers normally
    do not construct this class directly.

    Parameters
    ----------
    cluster_id
        HTCondor cluster identifier (as returned by *condor_submit*).
    remote_workdir
        Absolute AFS path on LXPlus where job outputs are written.
    ssh_host
        SSH host to connect to.  Defaults to ``lxplus.cern.ch``.
    """

    def __init__(
        self,
        cluster_id: str,
        remote_workdir: str,
        ssh_host: str = LXPLUS_HOST,
    ) -> None:
        self.cluster_id = cluster_id
        self.remote_workdir = remote_workdir
        self.ssh_host = ssh_host

    def wait(self, poll_interval: int = 30) -> Any:
        """Block until the job finishes and return its result.

        Polls HTCondor every *poll_interval* seconds until the job leaves
        the queue, then retrieves the value written by :func:`set_result`
        in the remote script.

        Parameters
        ----------
        poll_interval
            Seconds between ``condor_q`` polls.  Defaults to 30.

        Returns
        -------
        result
            The value passed to :func:`set_result` on the batch node,
            or ``None`` if the script did not call that function.

        Raises
        ------
        RuntimeError
            If the job exits with a non-zero status code or is held.
        """
        while self._job_in_queue():
            time.sleep(poll_interval)
        self._raise_on_failure()
        return self._fetch_result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_ssh(self, cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", self.ssh_host, cmd],
            check=False,
            capture_output=True,
            text=True,
        )

    def _job_in_queue(self) -> bool:
        """Return True while the job is still present in condor_q."""
        proc = self._run_ssh(
            f"condor_q {self.cluster_id} -format '%d\\n' ClusterId 2>/dev/null"
        )
        return bool(proc.stdout.strip())

    def _raise_on_failure(self) -> None:
        proc = self._run_ssh(
            f"condor_history {self.cluster_id}"
            " -format '%d\\n' ExitCode 2>/dev/null"
        )
        code_str = proc.stdout.strip()
        if code_str and code_str != "0":
            raise RuntimeError(
                f"LXPlus job {self.cluster_id} exited with code {code_str}.\n"
                f"Inspect {self.remote_workdir}/job.err for details."
            )

    def _fetch_result(self) -> Any:
        # Try JSON first (scalars and dicts)
        proc = self._run_ssh(
            f"cat {self.remote_workdir}/{_RESULT_JSON} 2>/dev/null"
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)

        # Fall back to numpy array
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
            local_npy = tf.name
        try:
            scp = subprocess.run(
                [
                    "scp",
                    "-q",
                    f"{self.ssh_host}:{self.remote_workdir}/{_RESULT_NPY}",
                    local_npy,
                ],
                check=False,
                capture_output=True,
            )
            if scp.returncode == 0:
                return np.load(local_npy)
        finally:
            if os.path.exists(local_npy):
                os.unlink(local_npy)

        return None


def run_on_lxplus(
    filepath: str, kwargs: dict[str, int | float | str | list]
) -> LxplusJob:
    """Submit a Python script to HTCondor on LXPlus.

    The script at *filepath* must:

    * Live inside a git-tracked project reachable via its ``origin`` remote
      (e.g. on ``gitlab.cern.ch``).
    * Declare its dependencies in a ``pyproject.toml`` so the project can
      be installed with ``pip install``.
    * Accept its parameters as ``argparse`` flags matching the keys in
      *kwargs*.

    Results are communicated back by calling :func:`set_result` inside
    the remote script.

    Parameters
    ----------
    filepath
        Path to the Python script to run on the batch node.
    kwargs
        Keyword arguments forwarded to the script as ``--key value``
        command-line flags.

    Returns
    -------
    job
        A :class:`LxplusJob` whose :meth:`~LxplusJob.wait` method blocks
        until the job finishes and returns the value set by
        :func:`set_result`.

    Notes
    -----
    * Requires passwordless SSH access to ``lxplus.cern.ch`` (Kerberos or
      an SSH key).
    * The git commit that is currently checked out locally is cloned on
      the batch node, so uncommitted local changes are **not** included.

    Examples
    --------
    >>> for step in range(10):
    ...     result = run_on_lxplus(
    ...         'kickdrift_test.py',
    ...         kwargs={'voltage': optimizer.suggest(),
    ...                 'output_dir': f'/eos/.../step{step}/'}
    ...     ).wait()
    ...     optimizer.update(result)
    """
    filepath = Path(filepath).resolve()
    git_root = _find_git_root(filepath)
    remote_url, commit = _get_git_info(git_root)
    script_rel = str(filepath.relative_to(git_root))

    remote_workdir = _make_remote_workdir()
    submission_cmd = _build_submission_command(
        remote_workdir=remote_workdir,
        remote_url=remote_url,
        commit=commit,
        script_rel=script_rel,
        kwargs=kwargs,
    )

    proc = subprocess.run(
        ["ssh", LXPLUS_HOST, submission_cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"LXPlus submission failed:\n{proc.stderr}")

    cluster_id = _parse_cluster_id(proc.stdout)
    return LxplusJob(cluster_id=cluster_id, remote_workdir=remote_workdir)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_git_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parent.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        f"No git repository found for {path}.  "
        "The script must reside in a git-tracked project."
    )


def _get_git_info(git_root: Path) -> tuple[str, str]:
    remote_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"],
        cwd=git_root,
        text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        text=True,
    ).strip()
    return remote_url, commit


def _make_remote_workdir() -> str:
    """Return a unique job directory path under ``~/blond_jobs/`` on LXPlus."""
    home_proc = subprocess.run(
        ["ssh", LXPLUS_HOST, "echo $HOME"],
        capture_output=True,
        text=True,
        check=True,
    )
    home = home_proc.stdout.strip()
    token = uuid.uuid4().hex[:12]
    return f"{home}/blond_jobs/job_{token}"


def _kwargs_to_cli(kwargs: dict) -> str:
    """Convert a kwargs dict to a shell-safe CLI argument string."""
    parts: list[str] = []
    for key, val in kwargs.items():
        if isinstance(val, list):
            parts.append(f"--{key}")
            parts.extend(shlex.quote(str(v)) for v in val)
        else:
            parts.append(f"--{key}")
            parts.append(shlex.quote(str(val)))
    return " ".join(parts)


def _build_submission_command(
    remote_workdir: str,
    remote_url: str,
    commit: str,
    script_rel: str,
    kwargs: dict,
) -> str:
    """Build the shell command executed on LXPlus to submit the HTCondor job.

    Uses single-quoted heredocs so that ``$SCRATCH`` and other shell
    variables are written *literally* into the generated scripts and
    expanded only when those scripts execute on the batch node.
    Python f-string interpolation (``{remote_workdir}`` etc.) takes place
    before the command is transmitted over SSH.
    """
    args_str = _kwargs_to_cli(kwargs)
    return f"""\
set -e
mkdir -p {remote_workdir}

cat > {remote_workdir}/wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
set -e
export BLOND_JOB_TMPDIR="{remote_workdir}"
SCRATCH=$(mktemp -d)
git clone --quiet '{remote_url}' "$SCRATCH/repo"
git -C "$SCRATCH/repo" checkout --quiet '{commit}'
pip install --quiet --user "$SCRATCH/repo"
python "$SCRATCH/repo/{script_rel}" {args_str}
WRAPPER_EOF
chmod +x {remote_workdir}/wrapper.sh

cat > {remote_workdir}/job.sub << 'SUB_EOF'
executable            = {remote_workdir}/wrapper.sh
output                = {remote_workdir}/job.out
error                 = {remote_workdir}/job.err
log                   = {remote_workdir}/job.log
should_transfer_files = NO
queue
SUB_EOF

condor_submit {remote_workdir}/job.sub
"""


def _parse_cluster_id(condor_output: str) -> str:
    """Extract the cluster ID from ``condor_submit`` stdout."""
    for line in condor_output.splitlines():
        if "submitted to cluster" in line:
            # "1 job(s) submitted to cluster 12345."
            return line.split("cluster")[-1].strip().rstrip(".")
    raise RuntimeError(
        f"Could not parse cluster ID from condor_submit output:\n{condor_output}"
    )
