# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helpers to measure the runtime of backend functions."""

from __future__ import annotations

import dataclasses
import datetime
import glob
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from blond.core.backends.backend import backend
from blond.core.backends.helpers import setup_backend

if TYPE_CHECKING:
    from matplotlib.lines import Line2D

ALL_MODES = (
    # "python",
    "numba",
    "cpp",
    "cpp_single_core",
    "cuda",
)

#: Macro-particle counts scanned by the benchmarks, in half-decade steps.
#: 1e8 particles need 0.8 GB per array, which still fits a 4 GB GPU.
N_MACROPARTICLES_SCAN = [int(round(value)) for value in np.logspace(3, 8, 11)]

#: Directory the benchmark plots and their raw data are written to.
RESULTS_DIRECTORY = Path(__file__).resolve().parents[1] / "results"

#: Sysfs files holding the current clock of every core, in kHz. Present
#: on Linux with a cpufreq driver, absent elsewhere.
CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"

#: Peak-to-peak spread of the repetitions above which a measurement is
#: reported as unstable, relative to their median. A throttling CPU
#: slows down from one repetition to the next and exceeds this.
MAX_RELATIVE_SPREAD = 0.05


def synchronize_device() -> None:
    """
    Block until all queued work on the active device has finished.

    CuPy launches kernels asynchronously, so a host timer stopped right
    after the call would only measure the kernel launch. A device-wide
    synchronization waits on every stream of the current device. On CPU
    backends this is a no-op.
    """
    if not backend.is_gpu:
        return
    import cupy as cp  # type: ignore # import only if needed

    cp.cuda.Device().synchronize()


def free_device_memory() -> None:
    """
    Return cached device memory to the driver.

    CuPy keeps freed blocks in a memory pool. Consecutive scan points
    allocate arrays of different sizes, so the pool would grow instead of
    reusing its blocks and the largest scan points can run out of memory.
    On CPU backends this is a no-op.
    """
    if not backend.is_gpu:
        return
    import cupy as cp  # type: ignore # import only if needed

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _timed_calls(
    fun: Callable[..., Any],
    kwargs: Mapping[str, Any],
    n_calls: int,
) -> float:
    """
    Return the duration of `n_calls` calls of `fun(**kwargs)`.

    The device is synchronized once, after the last call. On a GPU the
    calls therefore pipeline, exactly as they do in the tracking loop,
    which does not synchronize between kernels either. Synchronizing
    after every call instead would measure the launch-plus-synchronize
    latency, which exceeds the kernel duration for small arrays.

    Parameters
    ----------
    fun
        Function to be timed.
    kwargs
        Keyword arguments passed to `fun`.
    n_calls
        Number of calls inside the timed window.

    Returns
    -------
    duration
        Wall-clock duration of all `n_calls` calls, in seconds.
    """
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fun(**kwargs)
    synchronize_device()
    return time.perf_counter() - t0


def read_cpu_clock_ghz() -> float | None:
    """
    Return the current mean clock of all CPU cores, in GHz.

    The clock is read from sysfs, which only exists on Linux with a
    cpufreq driver. It is recorded alongside every measurement so that a
    run slowed down by thermal throttling can be recognized afterwards
    instead of being mistaken for a slower kernel.

    Returns
    -------
    clock_ghz
        Mean clock over all cores, or None where sysfs is unavailable.
    """
    clocks_ghz = []
    for path in glob.glob(CPUFREQ_GLOB):
        try:
            with open(path) as file:
                clocks_ghz.append(float(file.read()) / 1e6)
        except (OSError, ValueError):
            continue
    if not clocks_ghz:
        return None
    return statistics.mean(clocks_ghz)


def _relative_spread(durations: Sequence[float]) -> float:
    """
    Return the peak-to-peak spread of `durations`, relative to the median.

    Parameters
    ----------
    durations
        Durations of the individual repetitions, in seconds.

    Returns
    -------
    relative_spread
        ``(max - min) / median``, zero for identical durations.
    """
    median = statistics.median(durations)
    if median <= 0.0:
        return 0.0
    return (max(durations) - min(durations)) / median


@dataclasses.dataclass
class RuntimeResult:
    """
    One measured runtime together with the state of the machine.

    Attributes
    ----------
    runtime_per_call
        Mean runtime per call of the median repetition, in seconds.
    repeat_durations
        Duration of every repetition, in seconds. A monotonically rising
        sequence is the signature of a throttling CPU.
    clock_ghz_start
        Mean CPU clock before the measurement, in GHz, or None.
    clock_ghz_end
        Mean CPU clock after the measurement, in GHz, or None.
    """

    runtime_per_call: float
    repeat_durations: list[float]
    clock_ghz_start: float | None
    clock_ghz_end: float | None

    @property
    def relative_spread(self) -> float:
        """Peak-to-peak spread of the repetitions, relative to the median."""
        return _relative_spread(self.repeat_durations)

    @property
    def is_unstable(self) -> bool:
        """Whether the repetitions disagree by more than the tolerance."""
        return bool(self.relative_spread > MAX_RELATIVE_SPREAD)

    def as_dict(self) -> dict[str, Any]:
        """
        Return the result as a JSON-serializable dictionary.

        Returns
        -------
        as_dict
            The runtime and its diagnostics, for the benchmark metadata.
        """
        return {
            "runtime_per_call_seconds": self.runtime_per_call,
            "repeat_durations_seconds": list(self.repeat_durations),
            "relative_spread": self.relative_spread,
            "is_unstable": self.is_unstable,
            "clock_ghz_start": self.clock_ghz_start,
            "clock_ghz_end": self.clock_ghz_end,
        }


def runtime(
    fun: Callable[..., Any],
    kwargs: Mapping[str, Any],
    n_warmup: int,
    n_runs: int = 100_000,
    n_repeats: int = 5,
    target_repeat_duration: float = 0.1,
    probe_duration_fraction: float = 0.1,
    warmup_time_s: float = 0.5,
) -> RuntimeResult:
    """
    Measure the runtime per call of `fun(**kwargs)`.

    The measurement is preceded by `warmup_time_s` of sustained load,
    so that it is taken at the clock the device settles at rather than
    on its way there. Without it, a CPU is measured while it still runs
    at its boost clock and a GPU while it still idles in a low power
    state, and neither number reproduces.

    The timed calls are repeated `n_repeats` times and the median
    repetition is reported, which suppresses outliers from OS scheduling
    or garbage collection. The median rather than the minimum is used
    because the clock changes under load, so the fastest repetition is
    not representative. How much the repetitions still disagree is
    reported in the result, see `RuntimeResult.is_unstable`.

    Parameters
    ----------
    fun
        Function to be timed.
    kwargs
        Keyword arguments passed to `fun`.
    n_warmup
        Number of untimed calls before the measurement, e.g. to trigger
        JIT compilation (numba).
    n_runs
        Upper limit of timed calls per repetition. The number of calls is
        normally set by `target_repeat_duration`; this limit only guards
        against a pathologically short call. Keep it large, because a
        small limit makes the repetition shorter than intended and the
        single synchronization at its end is then amortized over few
        calls, which biases fast kernels.
    n_repeats
        Number of repetitions of the timed calls.
    target_repeat_duration
        Duration one repetition aims for, in seconds.
    probe_duration_fraction
        Duration of the probe that sizes a repetition, as a fraction of
        `target_repeat_duration`.
    warmup_time_s
        Time spent calling `fun` back to back before the measurement, in
        seconds. This clears the clock ramp on both CPU and GPU. It does
        not reach thermal steady state, which takes seconds to tens of
        seconds of sustained load; raise it if a scan still drifts.

    Returns
    -------
    result
        The runtime per call and the state of the machine during the
        measurement.
    """
    for _ in range(n_warmup):
        fun(**kwargs)
    # Don't let pending warmup work leak into the timed window
    synchronize_device()
    # One synchronized call, only used to size the batches below. On a
    # GPU this overestimates the per-call duration, because it cannot
    # pipeline and pays the synchronization latency.
    latency = _timed_calls(fun, kwargs, 1)
    _warmup_sustained_load(fun, kwargs, latency, warmup_time_s)
    # The clock speed has changed, so the first estimate is stale.
    latency = _timed_calls(fun, kwargs, 1)
    n_probe_calls = _calls_for_duration(
        probe_duration_fraction * target_repeat_duration, latency, n_runs
    )
    probe_duration = _timed_calls(fun, kwargs, n_probe_calls)
    runs_per_repeat = _calls_for_duration(
        target_repeat_duration, probe_duration / n_probe_calls, n_runs
    )
    clock_ghz_start = read_cpu_clock_ghz()
    repeat_durations = [
        _timed_calls(fun, kwargs, runs_per_repeat) for _ in range(n_repeats)
    ]
    return RuntimeResult(
        runtime_per_call=(
            statistics.median(repeat_durations) / runs_per_repeat
        ),
        repeat_durations=repeat_durations,
        clock_ghz_start=clock_ghz_start,
        clock_ghz_end=read_cpu_clock_ghz(),
    )


def _calls_for_duration(
    target_duration: float,
    duration_per_call: float,
    n_calls_max: int,
) -> int:
    """
    Return how many calls fill `target_duration`, at least one.

    Parameters
    ----------
    target_duration
        Duration to be filled, in seconds.
    duration_per_call
        Estimated duration of one call, in seconds.
    n_calls_max
        Upper limit of the returned number of calls.

    Returns
    -------
    n_calls
        Number of calls, within ``[1, n_calls_max]``.
    """
    n_calls = math.ceil(target_duration / max(duration_per_call, 1e-9))
    return max(1, min(n_calls_max, n_calls))


def _warmup_sustained_load(
    fun: Callable[..., Any],
    kwargs: Mapping[str, Any],
    duration_per_call: float,
    warmup_time_s: float,
    batch_duration: float = 0.05,
) -> None:
    """
    Call `fun` under sustained load until the clock has settled.

    A CPU starts a burst of work at its boost clock and drops to its
    sustained clock once it heats up, a GPU idles in a low power state
    and only clocks up under load. Both move the measured runtime, in
    opposite directions, so the measurement is taken after this warmup
    rather than during the ramp.

    The calls are issued in batches that are synchronized before the
    clock is read again. Without that synchronization the host would
    queue GPU kernels far faster than the device retires them, and the
    warmup would keep the device busy for much longer than
    `warmup_time_s`. On CPU backends the synchronization is a no-op.

    Parameters
    ----------
    fun
        Function to be called.
    kwargs
        Keyword arguments passed to `fun`.
    duration_per_call
        Estimated duration of one call, in seconds.
    warmup_time_s
        Duration of the warmup, in seconds.
    batch_duration
        Duration one batch of calls aims for, in seconds. It sets by how
        much the warmup can overshoot `warmup_time_s`.
    """
    calls_per_batch = _calls_for_duration(
        batch_duration, duration_per_call, n_calls_max=100_000
    )
    warmup_end = time.perf_counter() + warmup_time_s
    while time.perf_counter() < warmup_end:
        for _ in range(calls_per_batch):
            fun(**kwargs)
        synchronize_device()


def scan_performance(
    fun: Callable[..., Any],
    make_kwargs: Callable[[Any], Mapping[str, Any]],
    scan_values: Sequence[Any],
    n_warmup: int,
    n_runs: int = 100_000,
    n_repeats: int = 5,
    warmup_time_s: float = 0.5,
    on_result: Callable[[Any, RuntimeResult], None] | None = None,
) -> list[RuntimeResult]:
    """
    Measure the runtime of `fun` for each value of a scanned parameter.

    Parameters
    ----------
    fun
        Function to be timed.
    make_kwargs
        Builds the keyword arguments of `fun` for one scan value,
        e.g. allocates the input arrays for a given array size. It is
        called once per scan value and is not timed, so each scan point
        starts from fresh arrays.
    scan_values
        Values of the scanned parameter.
    n_warmup
        Number of untimed calls per scan value.
    n_runs
        Upper limit of timed calls per repetition, see `runtime`.
    n_repeats
        Number of repetitions of the timed calls.
    warmup_time_s
        Duration of the sustained-load warmup per scan point, in seconds,
        see `runtime`.
    on_result
        Called with ``(scan_value, result)`` as soon as a scan point is
        measured, e.g. to update a live plot.

    Returns
    -------
    results
        Runtime and machine state for each entry of `scan_values`.
    """
    results = []
    for scan_value in scan_values:
        kwargs = make_kwargs(scan_value)
        result = runtime(
            fun=fun,
            kwargs=kwargs,
            n_warmup=n_warmup,
            n_runs=n_runs,
            n_repeats=n_repeats,
            warmup_time_s=warmup_time_s,
        )
        # Release this scan point's arrays before the next, larger, ones
        # are allocated
        del kwargs
        free_device_memory()
        if result.is_unstable:
            print(
                f"Unstable measurement at {scan_value}: the repetitions "
                f"spread by {100 * result.relative_spread:.1f} % "
                f"(clock {result.clock_ghz_start} -> "
                f"{result.clock_ghz_end} GHz). Suspect thermal "
                f"throttling; raise warmup_time_s.",
                flush=True,
            )
        results.append(result)
        if on_result is not None:
            on_result(scan_value, result)
    return results


def _is_interactive_backend() -> bool:
    """
    Report whether matplotlib draws into a window.

    Returns
    -------
    is_interactive
        False for file-only backends such as ``Agg``.
    """
    return plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps")


def _live_line_updater(
    line_per_call: Line2D,
    line_per_element: Line2D,
    marker_unstable: Line2D,
) -> tuple[
    list[Any], list[RuntimeResult], Callable[[Any, RuntimeResult], None]
]:
    """
    Create a callback that appends a point to a mode's lines and redraws.

    Parameters
    ----------
    line_per_call
        Line of one backend mode in the runtime-per-call panel.
    line_per_element
        Line of the same mode in the runtime-per-element panel.
    marker_unstable
        Marker-only line in the runtime-per-call panel that highlights
        the scan points whose repetitions disagreed, see
        `RuntimeResult.is_unstable`.

    Returns
    -------
    scanned_values
        List the callback appends the scan values to.
    results
        List the callback appends the measured results to.
    update_plot
        Callback for `scan_performance`'s `on_result`.
    """
    scanned_values: list[Any] = []
    results: list[RuntimeResult] = []

    def update_plot(scan_value: Any, result: RuntimeResult) -> None:
        scanned_values.append(scan_value)
        results.append(result)
        runtimes = [result_.runtime_per_call for result_ in results]
        line_per_call.set_data(scanned_values, runtimes)
        line_per_element.set_data(
            scanned_values,
            [
                runtime_ / value
                for runtime_, value in zip(
                    runtimes, scanned_values, strict=False
                )
            ],
        )
        unstable = [
            (value, result_.runtime_per_call)
            for value, result_ in zip(scanned_values, results, strict=False)
            if result_.is_unstable
        ]
        marker_unstable.set_data(
            [value for value, _ in unstable],
            [runtime_ for _, runtime_ in unstable],
        )
        for line in (line_per_call, line_per_element):
            line.axes.relim()
            line.axes.autoscale_view()
        if _is_interactive_backend():
            plt.draw()
            plt.pause(0.01)

    return scanned_values, results, update_plot


def _save_results(
    save_name: str,
    figure: Any,
    metadata: dict[str, Any],
) -> None:
    """
    Write the figure as PNG and the measured data as JSON.

    Parameters
    ----------
    save_name
        File name stem inside `RESULTS_DIRECTORY`.
    figure
        Matplotlib figure to save.
    metadata
        JSON-serializable description and results of the benchmark.
    """
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(RESULTS_DIRECTORY / f"{save_name}.png", dpi=150)
    with open(RESULTS_DIRECTORY / f"{save_name}.json", "w") as file:
        json.dump(metadata, file, indent=2)


def plot_performance(
    kernel_name: str,
    make_kwargs: Callable[[Any], Mapping[str, Any]],
    scan_values: Sequence[Any],
    n_warmup: int,
    n_runs: int = 100_000,
    n_repeats: int = 5,
    warmup_time_s: float = 0.5,
    modes: Sequence[str] = ALL_MODES,
    xlabel: str = "scan value",
    title: str | None = None,
    save_name: str | None = None,
) -> dict[str, list[RuntimeResult]]:
    """
    Scan the runtime of a `Specials` kernel on several backends and plot it.

    The plot is redrawn after every scan point, so slow kernels show
    progress while the scan is still running. Call ``plt.show()``
    afterwards to keep the window open.

    Scan points whose repetitions disagreed are drawn as hollow markers
    and their diagnostics are written to the JSON, so that a run spoiled
    by thermal throttling is visible rather than silently wrong.

    Parameters
    ----------
    kernel_name
        Name of the `Specials` method, e.g. ``"histogram"``.
    make_kwargs
        Builds the keyword arguments of the kernel for one scan value.
        It is called after the backend of each mode is activated, so
        arrays created with ``backend.<fn>`` land on the right device.
    scan_values
        Values of the scanned parameter.
    n_warmup
        Number of untimed calls per scan value.
    n_runs
        Upper limit of timed calls per repetition, see `runtime`.
    n_repeats
        Number of repetitions of the timed calls.
    warmup_time_s
        Duration of the sustained-load warmup per scan point, in seconds,
        see `runtime`. Raise it if the scan reports unstable points.
    modes
        Backend specials modes to compare. Modes that fail, e.g. because
        CuPy is not available, are reported and skipped.
    xlabel
        Label of the scanned parameter.
    title
        Figure title, defaults to `kernel_name`.
    save_name
        If given, the plot (PNG) and the measured data (JSON) are written
        to `RESULTS_DIRECTORY` under this name after every mode, so that
        a scan interrupted later keeps its finished modes.

    Returns
    -------
    results
        Runtime and machine state for each scan value, per mode. Skipped
        modes contain the points measured before the failure.
    """
    title = kernel_name if title is None else title
    figure, (axis_per_call, axis_per_element) = plt.subplots(
        1, 2, figsize=(13, 5)
    )
    figure.suptitle(title)
    axis_per_call.set_ylabel("runtime per call [s]")
    axis_per_element.set_ylabel(f"runtime per call / {xlabel} [s]")
    for axis in (axis_per_call, axis_per_element):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.grid(True, which="both", alpha=0.3)

    results: dict[str, list[RuntimeResult]] = {}
    measured_values: dict[str, list[Any]] = {}
    failures: dict[str, str] = {}
    started = datetime.datetime.now().isoformat(timespec="seconds")
    for mode in modes:
        (line_per_call,) = axis_per_call.plot([], [], "o-", label=mode)
        (line_per_element,) = axis_per_element.plot([], [], "o-", label=mode)
        # Hollow, oversized markers on top of the line, so that an
        # unstable point stays recognizable in the saved PNG
        (marker_unstable,) = axis_per_call.plot(
            [],
            [],
            "o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor=line_per_call.get_color(),
            linestyle="none",
            label=f"{mode} (unstable)",
        )
        axis_per_call.legend()
        mode_values, mode_results, update_plot = _live_line_updater(
            line_per_call, line_per_element, marker_unstable
        )
        results[mode] = mode_results
        measured_values[mode] = mode_values
        try:
            setup_backend(mode)
            scan_performance(
                fun=getattr(backend.specials, kernel_name),
                make_kwargs=make_kwargs,
                scan_values=scan_values,
                n_warmup=n_warmup,
                n_runs=n_runs,
                n_repeats=n_repeats,
                warmup_time_s=warmup_time_s,
                on_result=update_plot,
            )
        except Exception as exc:  # NOQA: BLE001 benchmark continues
            message = f"{type(exc).__name__}: {exc}"
            print(f"Skipped mode {mode!r}: {message}")
            failures[mode] = message
            line_per_call.set_label(f"{mode} (failed)")
            axis_per_call.legend()
        free_device_memory()
        runtimes_per_call = [
            result.runtime_per_call for result in mode_results
        ]
        print(f"{mode} runtimes per call [s]: {runtimes_per_call}", flush=True)
        if save_name is not None:
            _save_results(
                save_name,
                figure,
                _build_metadata(
                    kernel_name=kernel_name,
                    title=title,
                    xlabel=xlabel,
                    started=started,
                    scan_values=scan_values,
                    measured_values=measured_values,
                    results=results,
                    failures=failures,
                ),
            )
    return results


def _build_metadata(
    kernel_name: str,
    title: str,
    xlabel: str,
    started: str,
    scan_values: Sequence[Any],
    measured_values: Mapping[str, Sequence[Any]],
    results: Mapping[str, Sequence[RuntimeResult]],
    failures: Mapping[str, str],
) -> dict[str, Any]:
    """
    Assemble the JSON-serializable description of a benchmark run.

    Parameters
    ----------
    kernel_name
        Name of the benchmarked `Specials` method.
    title
        Figure title.
    xlabel
        Label of the scanned parameter.
    started
        ISO timestamp of the start of the run.
    scan_values
        Values of the scanned parameter that were requested.
    measured_values
        Scan values actually measured, per mode.
    results
        Runtime and machine state of every scan point, per mode.
    failures
        Error message of every mode that could not be measured.

    Returns
    -------
    metadata
        Dictionary written next to the plot as JSON.

    Notes
    -----
    The per-repetition durations and CPU clocks are kept so that a
    suspicious runtime can be checked against what the machine was doing
    at the time, rather than only against a rerun.
    """
    return {
        "kernel_name": kernel_name,
        "title": title,
        "xlabel": xlabel,
        "host": platform.node(),
        "started": started,
        "max_relative_spread": MAX_RELATIVE_SPREAD,
        "scan_values": list(scan_values),
        "measured_values": {
            mode: list(values) for mode, values in measured_values.items()
        },
        "runtimes_per_call_seconds": {
            mode: [result.runtime_per_call for result in mode_results]
            for mode, mode_results in results.items()
        },
        "diagnostics": {
            mode: [result.as_dict() for result in mode_results]
            for mode, mode_results in results.items()
        },
        "failures": dict(failures),
    }
