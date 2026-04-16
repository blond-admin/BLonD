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
import logging
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


LXPLUS_HOST = "lxplus.cern.ch"
_RESULT_JSON = "blond_result.json"
_RESULT_NPY = "blond_result.npy"
_ENV_JOB_TMPDIR = "BLOND_JOB_TMPDIR"

# HTCondor JobStatus codes (see condor_q -long / JobStatus attribute).
_CONDOR_STATUS = {
    "0": "Unexpanded",
    "1": "Idle",
    "2": "Running",
    "3": "Removed",
    "4": "Completed",
    "5": "Held",
    "6": "Transferring Output",
    "7": "Suspended",
}


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

    Attributes
    ----------
    cluster_id : str
        HTCondor cluster identifier.
    remote_workdir : str
        Remote directory containing all job artefacts.
    ssh_host : str
        SSH host used for polling and fetching results.
    stdout_path : str
        Remote path of the job's stdout file (``job.out``).  Fetch with
        e.g. ``scp <ssh_host>:<stdout_path> .`` to debug a running or
        failed job.
    stderr_path : str
        Remote path of the job's stderr file (``job.err``).
    condor_log_path : str
        Remote path of HTCondor's own log (``job.log``); useful for
        queue events, exit codes and hold reasons.
    """

    def __init__(
        self,
        cluster_id: str,
        remote_workdir: str,
        ssh_host: str = LXPLUS_HOST,
    ) -> None:
        logger.info(
            f"Created LxplusJob({cluster_id=}, {remote_workdir=}, {ssh_host=})"
        )
        self.cluster_id = cluster_id
        self.remote_workdir = remote_workdir
        self.ssh_host = ssh_host
        self.stdout_path = f"{remote_workdir}/job.out"
        self.stderr_path = f"{remote_workdir}/job.err"
        self.condor_log_path = f"{remote_workdir}/job.log"
        self._stdout_lines_seen = 0

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
            If the job exits with a non-zero status code, enters the
            ``Held`` state, or is removed from the queue.
        """
        last_status: str | None = None
        t0 = time.time()
        while True:
            try:
                status = self._job_status()
            except RuntimeError as exc:
                logger.warning(
                    f"Failed to poll job {self.cluster_id}: {exc}. "
                    f"Retrying in {poll_interval}s."
                )
                time.sleep(poll_interval)
                continue

            if status is None:
                break

            status_changed = status != last_status
            if status_changed:
                logger.info(
                    f"Job {self.cluster_id} status: {status} "
                    f"(stdout: {self.ssh_host}:{self.stdout_path}, "
                    f"condor log: {self.ssh_host}:{self.condor_log_path})"
                )
                last_status = status
                t0 = time.time()

            self._log_new_stdout()

            if status in ("Held", "Removed"):
                self._raise_stuck(status)

            if not status_changed:
                logger.info(
                    f"Job {self.cluster_id} still {status}"
                    f" since {int((time.time() - t0) / 60)} minutes; "
                    f" polling again in {poll_interval}s."
                )
            time.sleep(poll_interval)
        logger.info(f"Job {self.cluster_id} left the queue.")
        self._log_new_stdout()
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

    def _job_status(self) -> str | None:
        """Return the HTCondor JobStatus as a human-readable string.

        Returns ``None`` once the job has left the queue (``condor_q``
        reports no matching cluster with a successful exit code).
        Otherwise returns one of ``"Idle"``, ``"Running"``, ``"Held"``
        etc.  Unknown numeric codes are returned verbatim as
        ``"JobStatus=<n>"``.

        For multi-proc clusters (``queue N > 1``), the status of the
        first proc is reported.

        Raises
        ------
        RuntimeError
            If the ``ssh`` / ``condor_q`` call exits non-zero (e.g.
            connection failure, schedd unreachable).  Callers are
            expected to treat this as transient and retry.
        """
        proc = self._run_ssh(
            f"condor_q {self.cluster_id} -format '%d\\n' JobStatus 2>/dev/null"
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"condor_q on {self.ssh_host} failed for cluster "
                f"{self.cluster_id} (rc={proc.returncode}): "
                f"{proc.stderr.strip() or '(no stderr)'}"
            )
        lines = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
        if not lines:
            return None
        code = lines[0]
        return _CONDOR_STATUS.get(code, f"JobStatus={code}")

    def _log_new_stdout(self) -> None:
        """Log any lines appended to the remote ``job.out`` since last call.

        Tails complete (newline-terminated) lines so partial writes are
        re-read on the next poll once finished.  Silently no-ops when
        the file does not yet exist (job hasn't started).
        """
        proc = self._run_ssh(
            f"tail -n +{self._stdout_lines_seen + 1} {self.stdout_path} "
            f"2>/dev/null"
        )
        if proc.returncode != 0 or not proc.stdout:
            return
        text = proc.stdout
        lines = text.splitlines()
        if lines and not text.endswith("\n"):
            lines = lines[:-1]
        if not lines:
            return
        self._stdout_lines_seen += len(lines)
        for line in lines:
            logger.info(f"[job {self.cluster_id} stdout] {line}")

    def _raise_stuck(self, status: str) -> None:
        """Raise RuntimeError for a ``Held`` or ``Removed`` job.

        Includes the ``HoldReason`` (when available) and remote paths to
        the job's stdout, stderr, and condor log so the caller can
        debug.
        """
        reason = ""
        reason_proc = self._run_ssh(
            f"condor_q {self.cluster_id} "
            f"-format '%s\\n' HoldReason 2>/dev/null"
        )
        if reason_proc.returncode == 0 and reason_proc.stdout.strip():
            reason = reason_proc.stdout.strip().splitlines()[0].strip()
        msg = (
            f"LXPlus job {self.cluster_id} is {status} and will not "
            f"complete."
            f"\nstdout: {self.ssh_host}:{self.stdout_path}"
            f"\nstderr: {self.ssh_host}:{self.stderr_path}"
            f"\ncondor log: {self.ssh_host}:{self.condor_log_path}"
        )
        if reason:
            msg += f"\nHoldReason: {reason}"
        raise RuntimeError(msg)

    def _raise_on_failure(self) -> None:
        proc = self._run_ssh(
            f"condor_history {self.cluster_id}"
            " -format '%d\\n' ExitCode 2>/dev/null"
        )
        code_str = proc.stdout.strip()
        if code_str and code_str != "0":
            stderr_proc = self._run_ssh(f"cat {self.stderr_path} 2>/dev/null")
            stderr = stderr_proc.stdout.strip()
            msg = (
                f"LXPlus job {self.cluster_id} exited with code {code_str}."
                f"\nstdout: {self.ssh_host}:{self.stdout_path}"
                f"\nstderr: {self.ssh_host}:{self.stderr_path}"
                f"\ncondor log: {self.ssh_host}:{self.condor_log_path}"
            )
            if stderr:
                msg += f"\n--- job.err ---\n{stderr}"
            raise RuntimeError(msg)

    def _fetch_result(self) -> Any:
        # Try JSON first (scalars and dicts)
        proc = self._run_ssh(
            f"cat {self.remote_workdir}/{_RESULT_JSON} 2>/dev/null"
        )
        logger.debug(f"{proc.stdout=}")
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
    filepath: str,
    kwargs: dict[str, int | float | str | list],
    python: str = "python3.11",
    job_flavour: Literal[
        "espresso",
        "microcentury",
        "longlunch",
        "workday",
        "tomorrow",
        "testmatch",
        "nextweek",
    ] = "espresso",
    accounting_group="batch-u-abp-ext-rf",
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
    python
        Python interpreter to use on the batch node for both
        ``pip install`` and script execution.  Defaults to
        ``"python3.12"``.
    job_flavour
        espresso     = 20 minutes
        microcentury = 1 hour
        longlunch    = 2 hours
        workday      = 8 hours
        tomorrow     = 1 day
        testmatch    = 3 days
        nextweek     = 1 week
    accounting_group
        Should remain unchanged for BLonD users.

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
    assert filepath.exists(), f"{filepath} does not exist."
    git_root = _find_git_root(filepath)
    _assert_git_clean(git_root)
    remote_url, commit = _get_git_info(git_root)
    script_rel = str(filepath.relative_to(git_root))

    remote_workdir = _make_remote_workdir()
    submission_cmd = _build_submission_command(
        remote_workdir=remote_workdir,
        remote_url=remote_url,
        commit=commit,
        script_rel=script_rel,
        kwargs=kwargs,
        python=python,
        job_flavour=job_flavour,
        accounting_group=accounting_group,
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


def _assert_git_clean(git_root: Path) -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=git_root,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Uncommitted local changes detected.  "
            "Commit or stash them before submitting to LXPlus, "
            "otherwise the batch node (which clones from the remote) "
            "will run a different version.\n"
            f"Changed files:\n{dirty}"
        )

    unpushed_proc = subprocess.run(
        ["git", "rev-list", "@{u}..HEAD"],
        check=False,
        cwd=git_root,
        capture_output=True,
        text=True,
    )
    if unpushed_proc.returncode != 0:
        raise RuntimeError(
            "Could not determine whether local commits are pushed "
            "(no upstream branch configured).  "
            "Set a tracking branch with "
            "'git push --set-upstream origin <branch>' first."
        )
    if unpushed_proc.stdout.strip():
        raise RuntimeError(
            "Local commits have not been pushed to the remote.  "
            "Push them before submitting to LXPlus.\n"
            f"Unpushed commits:\n{unpushed_proc.stdout.strip()}"
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
    proc = subprocess.run(
        ["ssh", LXPLUS_HOST, "echo $HOME"],
        capture_output=True,
        text=True,
        check=True,
    )
    home = proc.stdout.strip()
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
    python: str = "python3.11",
    job_flavour=None,
    accounting_group=None,
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
# Exit immediately if any command fails (safer for automation/scripts)
set -e

# Set a temporary working directory variable
export BLOND_JOB_TMPDIR="{remote_workdir}"

# Create a temporary scratch directory and store its path
SCRATCH=$(mktemp -d)

# Clone the remote repository into the scratch directory (quiet mode suppresses output)
git clone --quiet '{remote_url}' "$SCRATCH/repo"

# Check out a specific commit inside the cloned repository (ensures reproducibility)
git -C "$SCRATCH/repo" checkout --quiet '{commit}'

# Install into a venv on local scratch. Installing to $HOME/.local (AFS)
# is slow and flakes with transient I/O errors on the batch nodes.
{python} -m venv "$SCRATCH/venv"
"$SCRATCH/venv/bin/pip" install --quiet "$SCRATCH/repo"

# Run the target Python script from the repository with provided arguments
"$SCRATCH/venv/bin/python" "$SCRATCH/repo/{script_rel}" {args_str}
WRAPPER_EOF
chmod +x {remote_workdir}/wrapper.sh

cat > {remote_workdir}/job.sub << 'SUB_EOF'
executable            = {remote_workdir}/wrapper.sh
output                = {remote_workdir}/job.out
error                 = {remote_workdir}/job.err
log                   = {remote_workdir}/job.log
should_transfer_files = NO
getenv                = True
+JobFlavour           = {job_flavour}
{f"+AccountingGroup      = {accounting_group}" if accounting_group else ""}

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
