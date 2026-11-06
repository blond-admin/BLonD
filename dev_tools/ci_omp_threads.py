"""Print the OpenMP thread count a containerised CI job should use.

In a container libgomp sizes its thread pool from the *host* CPU count: it
has no idea the job is limited by a cgroup CPU quota. A Kubernetes runner
entitled to two CPUs therefore happily starts sixteen OpenMP workers, and
every barrier leaves fourteen of them fighting the same quota. CFS
throttling then deschedules workers constantly, which is exactly the
condition that makes the short per-turn kernels collapse (see
``blond/core/backends/openmp_env.py`` for the wait-policy half of this
problem).

This script prints a single integer -- the thread count to export as
``OMP_NUM_THREADS`` -- derived from the cgroup quota, the CPU affinity mask
and a cap. It is pure standard library on purpose: CI runs it with a bare
``python3``, without activating the project venv and without importing
``blond`` (which would load an OpenMP runtime before the variable is set).

Configuration (environment variables):

``BLOND_CI_MAX_OMP_THREADS``
    Upper bound on the printed value. Defaults to
    :data:`DEFAULT_MAX_THREADS`.
"""

from __future__ import annotations

import argparse
import os

#: Upper bound on the thread count. The test suite runs many small arrays,
#: where the per-barrier cost outweighs what extra workers contribute, so a
#: modest cap is faster than "one thread per available CPU" even on a
#: generously sized runner. It still exercises the parallel code paths.
DEFAULT_MAX_THREADS = 8

#: Where the cgroup filesystem is mounted. Overridable for testing.
CGROUP_ROOT = "/sys/fs/cgroup"

#: Sentinel for an argument that should be detected from the environment.
#: Distinct from ``None``, which is a real answer ("there is no quota").
DETECT: object = object()


def _read(path: str) -> str | None:
    """Return the stripped content of ``path``, or ``None`` if unreadable."""
    try:
        with open(path) as file:
            return file.read().strip()
    except OSError:
        return None


def cgroup_cpu_quota(cgroup_root: str = CGROUP_ROOT) -> float | None:
    """
    Return the cgroup CPU quota in whole CPUs, or ``None`` if unlimited.

    Both cgroup v2 (``cpu.max``) and cgroup v1
    (``cpu/cpu.cfs_quota_us`` and ``cpu/cpu.cfs_period_us``) are
    supported. Anything unreadable or unparsable counts as "no quota", so
    this never raises.

    Parameters
    ----------
    cgroup_root
        Mount point of the cgroup filesystem.

    Returns
    -------
    float or None
        The quota expressed in CPUs, e.g. ``2.0`` for a two-CPU limit.
        ``None`` when no quota applies.
    """
    quota_period = _read(os.path.join(cgroup_root, "cpu.max"))
    if quota_period is not None:
        try:
            quota, period = quota_period.split()
        except ValueError:  # not the documented "<quota> <period>" form
            return None
        if quota == "max":
            return None
        return _divide(quota, period)

    quota = _read(os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us"))
    period = _read(os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us"))
    if quota is None or period is None or quota.startswith("-"):
        return None
    return _divide(quota, period)


def _divide(quota: str, period: str) -> float | None:
    """Return ``quota / period`` as CPUs, or ``None`` if not usable."""
    try:
        quota_value = float(quota)
        period_value = float(period)
    except ValueError:
        return None
    if quota_value <= 0 or period_value <= 0:
        return None
    return quota_value / period_value


def visible_cpu_count() -> int:
    """
    Return the number of CPUs this process may actually run on.

    Prefers the scheduler affinity mask over :func:`os.cpu_count`, since a
    runner may pin the job to a subset of the host's cores.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def omp_num_threads(
    visible_cpus: int | None = None,
    quota: float | None | object = DETECT,
    max_threads: int | None = None,
    divide: int = 1,
) -> int:
    """
    Return the OpenMP thread count to use, at least ``1``.

    The result is the smallest of the visible CPU count, the cgroup CPU
    quota (rounded down, so the pool never exceeds what the scheduler will
    actually hand out) and ``max_threads``.

    Parameters
    ----------
    visible_cpus
        CPUs the process may run on. Detected when ``None``.
    quota
        cgroup CPU quota in CPUs. Detected when left at :data:`DETECT`;
        pass ``None`` to state that no quota applies.
    max_threads
        Upper bound. Defaults to ``BLOND_CI_MAX_OMP_THREADS``, else
        :data:`DEFAULT_MAX_THREADS`.
    divide
        Number of concurrent processes sharing the entitlement, e.g. the
        number of MPI ranks. The result is divided by it, rounded down.

    Returns
    -------
    int
        The thread count, never below one.
    """
    if visible_cpus is None:
        visible_cpus = visible_cpu_count()
    if quota is DETECT:
        quota = cgroup_cpu_quota()
    if max_threads is None:
        max_threads = int(
            os.environ.get("BLOND_CI_MAX_OMP_THREADS", DEFAULT_MAX_THREADS)
        )

    threads = min(visible_cpus, max_threads)
    if quota is not None:
        threads = min(threads, int(quota))
    return max(1, threads // max(1, divide))


def _main() -> None:
    """Print the thread count, honouring an optional ``--divide N``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--divide",
        type=int,
        default=1,
        metavar="N",
        help="split the entitlement across N concurrent processes "
        "(e.g. the number of MPI ranks)",
    )
    print(omp_num_threads(divide=parser.parse_args().divide))


if __name__ == "__main__":
    _main()
