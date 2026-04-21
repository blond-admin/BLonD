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
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from argparse import Namespace
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

_JOB_FALVOURS = (
    "espresso",
    "microcentury",
    "longlunch",
    "workday",
    "tomorrow",
    "testmatch",
    "nextweek",
)


def is_on_htcondor() -> bool:
    """
    Check whether the current program is executed on HTCondor.

    Returns
    -------
    bool
        True when running inside an HTCondor batch job.
    """
    tmpdir = os.environ.get(_ENV_JOB_TMPDIR)
    return tmpdir is not None


def move_results_to_eos(
    source_local: str | os.PathLike,
    target_eos: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Copy a file or directory from the worker node to EOS via ``eos cp``.

    Intended to be called from within a batch job to persist results
    that would otherwise vanish when the worker's scratch disk is
    cleaned up.

    Parameters
    ----------
    source_local
        Path (file or directory) on the worker node.
    target_eos
        Destination on EOS (must start with ``/eos/``).  When *None*,
        defaults to
        ``/eos/user/<u>/<user>/blond_results/<basename(source_local)>``,
        with ``<user>`` taken from ``$USER``.
    verbose
        When *True* (default), print the resolved source/target, the
        ``eos`` commands being executed, and the size of the payload.
        Output goes to stdout so it ends up in the job's ``job.out``.

    Returns
    -------
    str
        The resolved destination path on EOS.

    Raises
    ------
    FileNotFoundError
        If *source_local* does not exist.
    subprocess.CalledProcessError
        If the underlying ``eos`` command exits non-zero.
    """
    src = Path(source_local)
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    if target_eos is None:
        target_eos = get_eos_target(source_local)

    if verbose:
        kind = "directory" if src.is_dir() else "file"
        nbytes = (
            sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
            if src.is_dir()
            else src.stat().st_size
        )
        print(
            f"[move_results_to_eos] copying {kind} {src} ({nbytes / 1024:.1f} KiB) "
            f"-> {target_eos}"
        )

    # Worker nodes don't auto-discover the MGM the way lxplus login nodes do.
    env = {**os.environ, "EOS_MGM_URL": "root://eosuser.cern.ch"}

    # `eos cp -r src dest` drops src *inside* dest, so we target the parent
    # and rely on the source's basename to land at target_eos.
    parent = str(Path(target_eos).parent)
    if Path(target_eos).name != src.name:
        raise ValueError(
            f"target_eos basename ({Path(target_eos).name!r}) must match "
            f"source basename ({src.name!r})"
        )
    mkdir_cmd = ["eos", "mkdir", "-p", parent]
    if verbose:
        print(f"[move_results_to_eos] $ {' '.join(mkdir_cmd)}")
    subprocess.run(mkdir_cmd, check=True, env=env)

    copy_cmd = ["eos", "cp"]
    if src.is_dir():
        copy_cmd.append("-r")
    copy_cmd.extend([str(src), parent + "/"])
    if verbose:
        print(f"[move_results_to_eos] $ {' '.join(copy_cmd)}")
    t0 = time.time()
    subprocess.run(copy_cmd, check=True, env=env)
    if verbose:
        print(
            f"[move_results_to_eos] done in {time.time() - t0:.1f}s -> {target_eos}"
        )

    return target_eos


def get_eos_target(
    source_local: str | os.PathLike,
) -> str:
    """
    Build the default EOS destination for a per-job result directory.

    Produces
    ``/eos/user/<u>/<user>/blond_results/<job_id>/<basename(source_local)>``
    where ``<job_id>`` is taken from ``BLOND_JOB_TMPDIR`` (or ``"local"``
    when running outside a batch job).

    Parameters
    ----------
    source_local
        Local path whose basename is reused as the final path component.

    Returns
    -------
    str
        The resolved EOS destination path.
    """
    src = Path(source_local)

    user = os.environ["USER"]
    job_dir = os.environ.get(_ENV_JOB_TMPDIR)
    job_id = Path(job_dir).name if job_dir else "local"
    target_eos = (
        f"/eos/user/{user[0]}/{user}/blond_results/{job_id}/{src.name}"
    )
    return target_eos


def save_args(args, target_dir):
    """
    Dump an ``argparse`` namespace as ``args.json`` in *target_dir*.

    Parameters
    ----------
    args
        An ``argparse.Namespace`` (as returned by ``parser.parse_args()``).
    target_dir
        Directory the JSON file is written into. Created if missing.
    """
    # args from parser.parse_args()
    # Convert argparse Namespace to dict
    args_dict = vars(args)

    os.makedirs(target_dir, exist_ok=True)

    # Dump to JSON file
    output_path = os.path.join(target_dir, "args.json")
    with open(output_path, "w") as f:
        json.dump(args_dict, f, indent=4)


def write_manifest(target_dir: str | os.PathLike) -> str:
    """
    Write a ``manifest.json`` describing the current run.

    Captures provenance (commit, repo URL, submission time) and
    runtime context (hostname, start time, Python/BLonD versions,
    condor cluster, scratch dir) so that the produced results folder
    is self-describing once copied to EOS.

    Parameters
    ----------
    target_dir
        Directory the manifest is written into. Created if missing.

    Returns
    -------
    str
        Path to the written ``manifest.json``.
    """
    import platform
    import socket
    import sys
    from datetime import datetime, timezone

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # `import blond; blond.__version__` is unreliable — the top-level
    # module doesn't expose it. Read from installed package metadata.
    from importlib.metadata import PackageNotFoundError, version

    try:
        blond_version = version("blond")
    except PackageNotFoundError:
        blond_version = None

    manifest = {
        "submitted_at": os.environ.get("BLOND_JOB_SUBMITTED_AT"),
        "started_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "commit": os.environ.get("BLOND_JOB_COMMIT"),
        "remote_url": os.environ.get("BLOND_JOB_REMOTE_URL"),
        "job_tmpdir": os.environ.get(_ENV_JOB_TMPDIR),
        "job_id": (
            Path(os.environ[_ENV_JOB_TMPDIR]).name
            if os.environ.get(_ENV_JOB_TMPDIR)
            else None
        ),
        # HTCondor doesn't export ClusterId/ProcId into the job env by
        # default; run_on_htcondor propagates them via the submit file's
        # `environment = ...` line.
        "condor_cluster": os.environ.get("CONDOR_CLUSTER_ID"),
        "condor_proc": os.environ.get("CONDOR_PROC_ID"),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "blond_version": blond_version,
        "argv": sys.argv,
    }

    output_path = target / "manifest.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return str(output_path)


def send_results_to_host(value: Any) -> None:
    """
    Write a result value from within a batch job.

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
    >>> send_results_to_host(0.4e-6)                        # float
    >>> send_results_to_host({'dt': 0.4e-6, 'dE': 25e6})   # dict
    >>> send_results_to_host(obs.dts[-1])                   # 1-D ndarray
    """
    if not is_on_htcondor():
        return

    tmpdir = os.environ.get(_ENV_JOB_TMPDIR)
    if isinstance(value, np.ndarray):
        np.save(os.path.join(tmpdir, _RESULT_NPY), value)
    else:
        with open(os.path.join(tmpdir, _RESULT_JSON), "w") as f:
            json.dump(value, f)


class HTCondorJob:
    """
    Handle for a job submitted to HTCondor on LXPlus.

    Instances are returned by :func:`run_on_htcondor`; callers normally
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

    Examples
    --------
    You need two files, 'launch_lxplus.py' and 'main.py'.
    They must sit within a pip-installable project
    that is available via `git clone`.
    An example project is available via https://gitlab.cern.ch/slauber/lxplussubmissiondemo

    'launch_lxplus.py'
    >>> # 'launch_lxplus.py'
    >>> import logging
    >>> from pathlib import Path
    >>>
    >>> from blond.specifics.cern.lxplus.submission import run_on_htcondor
    >>>
    >>> logging.basicConfig(level=logging.DEBUG)
    >>>
    >>> future = run_on_htcondor(
    ...     filepath=str(Path(__file__).parent / "main.py"),
    ...     kwargs=dict(count=1),
    ...     request_gpus=1,
    ... )
    >>> result = future.wait()

    'main.py'
    >>> # main.py
    >>> import blond
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from blond import setup_backend
    >>> from blond.handle_results.helpers import callers_relative_path
    >>> from blond.specifics.cern.lxplus.submission import (
    ...     is_on_htcondor,
    ...     move_results_to_eos,
    ...     write_manifest,
    ...     load_args,
    ...     send_results_to_host,
    ...     save_args,
    ... )
    >>> from blond.testing.simulation import SimulationTwoRFStations
    >>>
    >>> REMOTE_RESULTS = (
    ...     "/home/slauber/cernbox/blond_results/job_9d78f490dc58/results/"
    ... )
    >>>
    >>> if is_on_htcondor():
    ...     args = load_args()
    >>> else:
    ...     args = load_args(REMOTE_RESULTS)
    >>>
    >>> setup_backend("auto")
    >>>
    >>> RESULTS_LOCAL = callers_relative_path("results/", stacklevel=1)
    >>>
    >>> print(f"{blond=}")
    >>> print(f"{args.count}")
    >>>
    >>> helper = SimulationTwoRFStations()
    >>> sim = helper.simulation
    >>> helper.beam1.setup_beam(
    ...     dt=np.linspace(1e-3, 2e-3),
    ...     dE=np.linspace(1e-3, 2e-3),
    ... )
    >>> bunch_obs = blond.BeamObservationOncePerTurn(each_turn_i=1)
    >>> observables = (bunch_obs,)
    >>>
    >>> if is_on_htcondor():
    ...     sim.run_simulation(beams=helper.beam1, n_turns=2, observe=observables)
    ...
    ...     sim.save_results(
    ...         observe=observables,
    ...         common_name=RESULTS_LOCAL,
    ...     )
    ...     write_manifest(target_dir=RESULTS_LOCAL)
    ...     save_args(args=args, target_dir=RESULTS_LOCAL)
    ...     target_eos = move_results_to_eos(source_local=RESULTS_LOCAL)
    ...     send_results_to_host(123)
    ... else:
    ...     sim.load_results(
    ...         beams=helper.beam1,
    ...         n_turns=2,
    ...         observe=observables,
    ...         common_name=REMOTE_RESULTS,
    ...     )
    ... plt.scatter(bunch_obs.dts, bunch_obs.dEs)
    ... plt.show()
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

    def wait(
        self,
        poll_interval: int = 30,
        archive_to_eos: bool = True,
        cleanup_afs: bool = True,
    ) -> Any:
        """
        Block until the job finishes and return its result.

        Polls HTCondor every *poll_interval* seconds until the job leaves
        the queue, then retrieves the value written by
        :func:`send_results_to_host` in the remote script.

        Parameters
        ----------
        poll_interval
            Seconds between ``condor_q`` polls.  Defaults to 30.
        archive_to_eos
            After the job succeeds, copy the AFS workdir (wrapper.sh,
            job.sub, job.out, job.err, job.log, result files) to
            ``/eos/user/<u>/<user>/blond_results/<job_id>/submission/``
            so the full run context is persisted alongside the results.
        cleanup_afs
            After a successful archive, ``rm -rf`` the AFS workdir to
            keep AFS quota free.  Ignored if *archive_to_eos* is False
            or the archive step fails.

        Returns
        -------
        result
            The value passed to :func:`send_results_to_host` on the batch node,
            or ``None`` if the script did not call that function.

        Raises
        ------
        RuntimeError
            If the job exits with a non-zero status code, enters the
            ``Held`` state, or is removed from the queue.
        KeyboardInterrupt
            Re-raised after ``condor_rm`` is issued so that a local
            interrupt (e.g. PyCharm stop button) also tears down the
            remote job instead of leaving it orphaned in the queue.
        """
        last_status: str | None = None
        t0 = time.time()
        try:
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
                        f"[Job {self.cluster_id} status] {status} "
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
                        f"[Job {self.cluster_id} status] still {status}"
                        f" since {int((time.time() - t0) / 60)} minutes; "
                        f" polling again in {poll_interval}s."
                    )
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.warning(
                f"[Job {self.cluster_id}] KeyboardInterrupt received while "
                f"polling; issuing condor_rm to kill the remote job."
            )
            self._condor_rm()
            raise
        logger.info(f"[Job {self.cluster_id} status] left the queue.")
        self._log_new_stdout()
        self._raise_on_failure()
        result = self._fetch_result()
        if archive_to_eos:
            archived = self._archive_submission_to_eos()
            if archived and cleanup_afs:
                self._cleanup_afs()
        return result

    def _run_ssh(self, cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", self.ssh_host, cmd],
            check=False,
            capture_output=True,
            text=True,
        )

    def _condor_rm(self) -> None:
        """
        Remove this cluster from the HTCondor queue via ``condor_rm``.

        SIGINT is ignored for the duration of the SSH call so a second
        Ctrl-C (e.g. the user hammering the IDE stop button) cannot
        abort cleanup and leave the job orphaned in the queue.

        Failures are logged (warning) but not raised: this is called
        from an interrupt handler where re-raising a secondary error
        would mask the original ``KeyboardInterrupt``.
        """
        # Only safe to swap signal handlers from the main thread;
        # if we're on a worker thread, skip the guard and accept
        # that a second interrupt may abort cleanup.
        can_block_sigint = (
            threading.current_thread() is threading.main_thread()
        )
        prev_handler = (
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if can_block_sigint
            else None
        )
        try:
            proc = self._run_ssh(f"condor_rm {self.cluster_id}")
        finally:
            if can_block_sigint:
                signal.signal(signal.SIGINT, prev_handler)
        if proc.returncode != 0:
            logger.warning(
                f"[Job {self.cluster_id}] condor_rm failed "
                f"(rc={proc.returncode}): "
                f"{proc.stderr.strip() or '(no stderr)'}. "
                f"Check the queue manually on {self.ssh_host}."
            )
            return
        logger.info(
            f"[Job {self.cluster_id}] condor_rm issued: "
            f"{proc.stdout.strip() or '(no stdout)'}"
        )

    def _job_status(self) -> str | None:
        """
        Return the HTCondor JobStatus as a human-readable string.

        Unknown numeric codes are returned verbatim as
        ``"JobStatus=<n>"``.  For multi-proc clusters (``queue N > 1``),
        the status of the first proc is reported.

        Returns
        -------
        str or None
            ``None`` once the job has left the queue (``condor_q``
            reports no matching cluster with a successful exit code),
            otherwise one of ``"Idle"``, ``"Running"``, ``"Held"``, etc.

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
        """
        Log any lines appended to the remote ``job.out`` since last call.

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
            logger.info(f"[Job {self.cluster_id} stdout] {line}")

    def _raise_stuck(self, status: str) -> None:
        """
        Raise RuntimeError for a ``Held`` or ``Removed`` job.

        Includes the ``HoldReason`` (when available) and remote paths to
        the job's stdout, stderr, and condor log so the caller can
        debug.

        Parameters
        ----------
        status
            The observed condor status (``"Held"`` or ``"Removed"``),
            included in the raised error message.
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

    def _archive_submission_to_eos(self) -> str | None:
        """
        Copy the AFS workdir to EOS using ``eos cp``.

        Runs on the lxplus login node. Over non-interactive SSH the
        login profile is not sourced, so ``EOS_MGM_URL`` is set
        explicitly (otherwise ``eos`` defaults to ``root://localhost``).
        FUSE ``/eos`` is avoided because it is known to silently drop
        data under load; ``eos cp`` is the CERN-recommended path for
        reliable writes. Files are copied individually to sidestep
        ``eos cp -r``'s "copy into" nesting.

        Returns
        -------
        str or None
            The EOS target path on success, or ``None`` on failure
            (logged as a warning; non-fatal).
        """
        job_id = Path(self.remote_workdir).name
        target_tpl = (
            f"/eos/user/${{USER:0:1}}/$USER/blond_results/{job_id}/submission"
        )
        remote_cmd = (
            "set -e; "
            "export EOS_MGM_URL=root://eosuser.cern.ch; "
            f'target="{target_tpl}"; '
            'eos mkdir -p "$target"; '
            f"for f in {self.remote_workdir}/*; do "
            '  [ -f "$f" ] || continue; '
            '  eos cp "$f" "$target/"; '
            "done; "
            'echo "$target"'
        )
        proc = self._run_ssh(remote_cmd)
        if proc.returncode != 0:
            logger.warning(
                f"[Job {self.cluster_id}] archive to EOS failed: "
                f"{proc.stderr.strip() or '(no stderr)'}"
            )
            return None
        target = proc.stdout.strip().splitlines()[-1]
        logger.info(f"[Job {self.cluster_id}] archived submission to {target}")
        return target

    def _cleanup_afs(self) -> None:
        """Remove the AFS workdir after a successful archive."""
        proc = self._run_ssh(f"rm -rf -- {shlex.quote(self.remote_workdir)}")
        if proc.returncode != 0:
            logger.warning(
                f"[Job {self.cluster_id}] AFS cleanup failed: "
                f"{proc.stderr.strip() or '(no stderr)'}"
            )
            return
        logger.info(
            f"[Job {self.cluster_id}] removed AFS workdir {self.remote_workdir}"
        )

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


def run_on_htcondor(
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
    accounting_group="group_u_BE.ABP.normal",
    request_gpus: int | None = None,
) -> HTCondorJob:
    """
    Submit a Python script to HTCondor on LXPlus.

    The script at *filepath* must:

    * Live inside a git-tracked project reachable via its ``origin`` remote
      (e.g. on ``gitlab.cern.ch``).
    * Declare its dependencies in a ``pyproject.toml`` so the project can
      be installed with ``pip install``.
    * Read its parameters with :func:`load_args` (which deserialises
      ``args.json`` from ``$BLOND_JOB_TMPDIR``).

    Results are communicated back by calling :func:`send_results_to_host`
    inside the remote script.

    Parameters
    ----------
    filepath
        Path to the Python script to run on the batch node.
    kwargs
        Keyword arguments serialised to ``args.json`` in the job's
        remote working directory; the script loads them with
        :func:`load_args`.
    python
        Python interpreter to use on the batch node for both
        ``pip install`` and script execution.  Defaults to
        ``"python3.12"``.
    job_flavour
        Condor queue flavour selecting the job's max wall time.
        Allowed values: ``espresso`` (20 min), ``microcentury``
        (1 h), ``longlunch`` (2 h), ``workday`` (8 h), ``tomorrow``
        (1 day), ``testmatch`` (3 days), ``nextweek`` (1 week).
    accounting_group
        Should remain unchanged for BLonD users.
    request_gpus
        If set, adds ``request_gpus = N`` to the submit file so the job
        is matched to a GPU-equipped worker. Defaults to *None* (CPU).

    Returns
    -------
    job
        A :class:`LxplusJob` whose :meth:`~LxplusJob.wait` method blocks
        until the job finishes and returns the value set by
        :func:`send_results_to_host`.

    Notes
    -----
    * Requires passwordless SSH access to ``lxplus.cern.ch`` (Kerberos or
      an SSH key).
    * The git commit that is currently checked out locally is cloned on
      the batch node, so uncommitted local changes are **not** included.

    Examples
    --------
    >>> for step in range(10):
    ...     result = run_on_htcondor(
    ...         'kickdrift_test.py',
    ...         kwargs={'voltage': optimizer.suggest(),
    ...                 'output_dir': f'/eos/.../step{step}/'}
    ...     ).wait()
    ...     optimizer.update(result)
    """
    assert job_flavour in _JOB_FALVOURS, (
        f"{job_flavour=}, but must be in {_JOB_FALVOURS}."
    )
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
        request_gpus=request_gpus,
    )

    logger.info(f"Submitting {submission_cmd}")

    proc = subprocess.run(
        ["ssh", LXPLUS_HOST, submission_cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"LXPlus submission failed:\n{proc.stderr}")

    cluster_id = _parse_cluster_id(proc.stdout)
    return HTCondorJob(cluster_id=cluster_id, remote_workdir=remote_workdir)


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
    """
    Return a unique job directory path under ``~/blond_jobs/`` on LXPlus.

    Returns
    -------
    str
        Absolute AFS path of a per-job working directory on LXPlus.
    """
    proc = subprocess.run(
        ["ssh", LXPLUS_HOST, "echo $HOME"],
        capture_output=True,
        text=True,
        check=True,
    )
    home = proc.stdout.strip()
    token = uuid.uuid4().hex[:12]
    return f"{home}/blond_jobs/job_{token}"


def load_args(location: str | os.PathLike | None = None) -> Namespace:
    """
    Load an ``args.json`` file previously written by :func:`save_args`.

    Parameters
    ----------
    location
        Directory containing an ``args.json`` file. When *None* (the
        default), the directory is taken from ``$BLOND_JOB_TMPDIR``,
        which :func:`run_on_htcondor` sets on the batch node to point
        at the job's workdir.

    Returns
    -------
    Namespace
        An ``argparse.Namespace`` reconstructed from the JSON contents,
        suitable for drop-in use in place of ``parser.parse_args()``.
    """
    if location is None:
        location = os.environ.get(_ENV_JOB_TMPDIR)
        if location is None:
            raise RuntimeError(
                f"load_args() called without a location and "
                f"${_ENV_JOB_TMPDIR} is not set. Either run under "
                f"run_on_htcondor (which sets it) or pass an explicit "
                f"directory."
            )
    with open(os.path.join(location, "args.json")) as f:
        args = json.load(f)
    return Namespace(**args)


def _build_submission_command(
    remote_workdir: str,
    remote_url: str,
    commit: str,
    script_rel: str,
    kwargs: dict,
    python: str = "python3.11",
    job_flavour=None,
    accounting_group=None,
    request_gpus: int | None = None,
) -> str:
    """
    Build the shell command executed on LXPlus to submit the HTCondor job.

    Uses single-quoted heredocs so that ``$SCRATCH`` and other shell
    variables are written *literally* into the generated scripts and
    expanded only when those scripts execute on the batch node.
    Python f-string interpolation (``{remote_workdir}`` etc.) takes place
    before the command is transmitted over SSH.

    Parameters
    ----------
    remote_workdir
        Absolute AFS path on LXPlus where the wrapper and job.sub
        files are written and HTCondor writes its logs.
    remote_url
        Git remote URL cloned on the batch node.
    commit
        Git commit SHA checked out on the batch node for reproducibility.
    script_rel
        Path to the target Python script relative to the git root.
    kwargs
        Script arguments, serialised to ``args.json`` in
        *remote_workdir* so the batch script can load them with
        :func:`load_args`.
    python
        Python interpreter used on the batch node.
    job_flavour
        Condor ``+JobFlavour`` value controlling max wall time.
    accounting_group
        Value for ``+AccountingGroup``. Omitted from the submit file
        when falsy.
    request_gpus
        When set, emits ``request_gpus = N`` into the submit file.

    Returns
    -------
    str
        The complete shell command to be executed over SSH on LXPlus.
    """
    from datetime import datetime, timezone

    args_json = json.dumps(kwargs, indent=2)
    submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"""\
set -e
mkdir -p {remote_workdir}

cat > {remote_workdir}/args.json << 'ARGS_EOF'
{args_json}
ARGS_EOF

cat > {remote_workdir}/wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
# Exit immediately if any command fails (safer for automation/scripts)
set -e

# Set a temporary working directory variable
export BLOND_JOB_TMPDIR="{remote_workdir}"

# Propagate provenance info to the job so it can write a manifest.
export BLOND_JOB_COMMIT="{commit}"
export BLOND_JOB_REMOTE_URL="{remote_url}"
export BLOND_JOB_SUBMITTED_AT="{submitted_at}"
# Force matplotlib to a non-interactive backend on the headless worker,
# so any plt.show() in user code is a no-op instead of blocking.
export MPLBACKEND="agg"
{'export BLOND_BACKEND_MODE="cuda"' if request_gpus else ""}

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
{'"$SCRATCH/venv/bin/pip" install --quiet cupy-cuda12x' if request_gpus else ""}

# Run the target Python script from the repository; the script reads
# its parameters from args.json via blond.specifics.cern.lxplus.load_args.
"$SCRATCH/venv/bin/python" "$SCRATCH/repo/{script_rel}"
WRAPPER_EOF
chmod +x {remote_workdir}/wrapper.sh

cat > {remote_workdir}/job.sub << 'SUB_EOF'

{f"request_gpus          = {request_gpus}" if request_gpus else ""}

executable            = {remote_workdir}/wrapper.sh
output                = {remote_workdir}/job.out
error                 = {remote_workdir}/job.err
log                   = {remote_workdir}/job.log


should_transfer_files = NO
getenv                = True

# HTCondor doesn't put ClusterId/ProcId in the job's env by default;
# expose them so the job can include them in its manifest.
environment           = "CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"


+JobFlavour           = {job_flavour}
{f'+AccountingGroup      = "{accounting_group}"' if accounting_group else ""}

queue
SUB_EOF

condor_submit {remote_workdir}/job.sub
"""


def _parse_cluster_id(condor_output: str) -> str:
    """
    Extract the cluster ID from ``condor_submit`` stdout.

    Parameters
    ----------
    condor_output
        The stdout produced by ``condor_submit``.

    Returns
    -------
    str
        The cluster ID reported by HTCondor.
    """
    for line in condor_output.splitlines():
        if "submitted to cluster" in line:
            # "1 job(s) submitted to cluster 12345."
            return line.split("cluster")[-1].strip().rstrip(".")
    raise RuntimeError(
        f"Could not parse cluster ID from condor_submit output:\n{condor_output}"
    )
