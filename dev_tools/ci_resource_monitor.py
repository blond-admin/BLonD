"""Sample what a containerised CI job uses, from its own cgroup.

Host-level tools are misleading inside a Kubernetes pod: ``vmstat`` on a
192-CPU node reports a job saturating its 8 CPUs as "4 % busy", and its
"free memory" is the node's, not the job's ``memory.max``. The cgroup v2
files under ``/sys/fs/cgroup`` are the job's own accounting, so this
script samples those instead:

``sample``
    Append one CSV row every ``--interval`` seconds until killed: memory in
    use (total and anonymous -- the total includes page cache, which the
    kernel can drop, the anonymous part it cannot), swap in use, and the
    cumulative CPU time and CFS throttling counters.
``digest``
    Print a one-line summary of such a CSV: peak memory against
    ``memory.max``, CPU in cores against the job's entitlement, and how
    much of the run the CFS quota throttled the job. Throttling is the
    number that says "the tests are CPU-starved", which no host-wide tool
    can show.

Pure standard library on purpose, like ``ci_omp_threads.py`` (whose CPU
entitlement logic it reuses): CI runs it with a bare ``python3`` in the
background, without the project venv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time

from ci_omp_threads import CGROUP_ROOT, cgroup_cpu_quota, visible_cpu_count

#: CSV columns, in order. ``time`` is seconds since the epoch.
FIELDS = (
    "time",
    "memory_current",
    "memory_anon",
    "swap_current",
    "cpu_usage_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
)

#: Samples needed before a rate (CPU cores, throttling) can be computed.
MIN_SAMPLES_FOR_RATES = 2


def _read_int(path: str) -> int | None:
    """Return the integer in ``path``, or ``None`` if unreadable."""
    try:
        with open(path) as file:
            return int(file.read().strip())
    except (OSError, ValueError):
        return None


def _read_keyed(path: str) -> dict[str, int]:
    """Return the ``key value`` lines of a cgroup stat file as a dict."""
    values = {}
    try:
        with open(path) as file:
            for line in file:
                key, _, value = line.partition(" ")
                try:
                    values[key] = int(value)
                except ValueError:
                    continue
    except OSError:
        pass
    return values


def read_sample(cgroup_root: str = CGROUP_ROOT, now: float | None = None):
    """
    Read one sample of the job's cgroup counters.

    Parameters
    ----------
    cgroup_root
        Mount point of the cgroup v2 filesystem.
    now
        Timestamp to record. Defaults to :func:`time.time`.

    Returns
    -------
    dict
        One value per entry of :data:`FIELDS`; ``None`` where the kernel
        does not expose the counter.
    """
    memory_stat = _read_keyed(os.path.join(cgroup_root, "memory.stat"))
    cpu_stat = _read_keyed(os.path.join(cgroup_root, "cpu.stat"))
    return {
        "time": time.time() if now is None else now,
        "memory_current": _read_int(
            os.path.join(cgroup_root, "memory.current")
        ),
        "memory_anon": memory_stat.get("anon"),
        "swap_current": _read_int(
            os.path.join(cgroup_root, "memory.swap.current")
        ),
        "cpu_usage_usec": cpu_stat.get("usage_usec"),
        "nr_periods": cpu_stat.get("nr_periods"),
        "nr_throttled": cpu_stat.get("nr_throttled"),
        "throttled_usec": cpu_stat.get("throttled_usec"),
    }


def append_sample(path: str, sample: dict) -> None:
    """Append ``sample`` to the CSV at ``path``, writing a header first."""
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow(
            "" if sample[field] is None else sample[field] for field in FIELDS
        )


def read_samples(path: str) -> list[dict]:
    """Read back a CSV written by :func:`append_sample`."""
    samples = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            try:
                samples.append(
                    {
                        field: None if row[field] == "" else float(row[field])
                        for field in FIELDS
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue  # row truncated by the kill at the end of the job
    return samples


def _peak(samples: list[dict], field: str):
    values = [s[field] for s in samples if s[field] is not None]
    return max(values) if values else None


def _delta(first: dict, last: dict, field: str):
    if first[field] is None or last[field] is None:
        return None
    return last[field] - first[field]


def summarize(samples: list[dict], cpu_entitlement: float) -> dict:
    """
    Condense a timeline into job-relative peaks and rates.

    Parameters
    ----------
    samples
        Samples in time order, as from :func:`read_samples`.
    cpu_entitlement
        CPUs the job may use (cgroup quota, else visible CPUs).

    Returns
    -------
    dict
        Peaks in bytes, CPU use in cores and as a fraction of
        ``cpu_entitlement``, total throttled seconds and the fraction of
        CFS periods that were throttled. Rates are ``None`` with fewer
        than two samples.
    """
    summary = {
        "n_samples": len(samples),
        "peak_memory": _peak(samples, "memory_current"),
        "peak_anon": _peak(samples, "memory_anon"),
        "peak_swap": _peak(samples, "swap_current"),
        "cpu_entitlement": cpu_entitlement,
        "mean_cpus": None,
        "peak_cpus": None,
        "mean_cpu_fraction": None,
        "throttled_seconds": None,
        "throttled_period_fraction": None,
    }
    if len(samples) < MIN_SAMPLES_FOR_RATES:
        return summary

    first, last = samples[0], samples[-1]
    elapsed = last["time"] - first["time"]
    used = _delta(first, last, "cpu_usage_usec")
    if used is not None and elapsed > 0:
        summary["mean_cpus"] = used / 1e6 / elapsed
        summary["mean_cpu_fraction"] = summary["mean_cpus"] / cpu_entitlement
        summary["peak_cpus"] = max(
            (later["cpu_usage_usec"] - earlier["cpu_usage_usec"])
            / 1e6
            / (later["time"] - earlier["time"])
            for earlier, later in itertools.pairwise(samples)
            if later["time"] > earlier["time"]
        )

    throttled = _delta(first, last, "throttled_usec")
    if throttled is not None:
        summary["throttled_seconds"] = throttled / 1e6
    periods = _delta(first, last, "nr_periods")
    throttled_periods = _delta(first, last, "nr_throttled")
    if periods and throttled_periods is not None:
        summary["throttled_period_fraction"] = throttled_periods / periods
    return summary


def _gigabytes(value) -> str:
    return "n/a" if value is None else f"{value / 1e9:.2f} GB"


def format_digest(summary: dict, memory_max: int | None) -> str:
    """Render :func:`summarize` output as one ``resources:`` log line."""
    peak = summary["peak_memory"]
    of_max = (
        f" ({100 * peak / memory_max:.0f}% of memory.max)"
        if peak is not None and memory_max
        else ""
    )
    parts = [
        f"resources: {summary['n_samples']} samples",
        f"peak_mem={_gigabytes(peak)}{of_max}",
        f"peak_anon={_gigabytes(summary['peak_anon'])}",
        f"peak_swap={_gigabytes(summary['peak_swap'])}",
    ]
    if summary["mean_cpus"] is not None:
        parts.append(
            f"cpu mean={summary['mean_cpus']:.1f}"
            f"/peak={summary['peak_cpus']:.1f}"
            f" of {summary['cpu_entitlement']:g} cores"
            f" ({100 * summary['mean_cpu_fraction']:.0f}% mean)"
        )
    if summary["throttled_seconds"] is not None:
        fraction = summary["throttled_period_fraction"]
        parts.append(
            f"throttled={summary['throttled_seconds']:.1f} s"
            + (
                ""
                if fraction is None
                else f" ({100 * fraction:.0f}% of periods)"
            )
        )
    return "  ".join(parts)


def _cpu_entitlement() -> float:
    quota = cgroup_cpu_quota()
    cpus = visible_cpu_count()
    return cpus if quota is None else min(quota, cpus)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="append samples until killed")
    sample.add_argument("output")
    sample.add_argument("--interval", type=float, default=5.0)
    digest = commands.add_parser("digest", help="summarise a timeline")
    digest.add_argument("input")
    arguments = parser.parse_args()

    if arguments.command == "sample":
        while True:
            append_sample(arguments.output, read_sample())
            time.sleep(arguments.interval)
    else:
        memory_max = _read_int(os.path.join(CGROUP_ROOT, "memory.max"))
        print(
            format_digest(
                summarize(read_samples(arguments.input), _cpu_entitlement()),
                memory_max,
            )
        )


if __name__ == "__main__":
    _main()
