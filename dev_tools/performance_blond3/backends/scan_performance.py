# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helpers to measure the runtime of backend functions."""

from __future__ import annotations

import datetime
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


def runtime(
    fun: Callable[..., Any],
    kwargs: Mapping[str, Any],
    n_warmup: int,
    n_runs: int = 100_000,
    n_repeats: int = 5,
    target_repeat_duration: float = 0.1,
    probe_duration_fraction: float = 0.1,
    gpu_warmup_duration: float = 3.0,
) -> float:
    """
    Measure the runtime per call of `fun(**kwargs)`.

    The timed calls are repeated `n_repeats` times and the median
    repetition is reported, which suppresses outliers from OS scheduling
    or garbage collection. The median rather than the minimum is used
    because GPUs change their power state under load, so the fastest
    repetition is not representative.

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
    gpu_warmup_duration
        On a GPU backend, time spent calling `fun` back to back before
        the measurement, in seconds. A GPU idles in a low power state and
        only speeds up under sustained load; a kernel that synchronizes
        after every call would otherwise be measured in the slow state.

    Returns
    -------
    runtime
        Mean runtime per call of the median repetition, in seconds.
    """
    for _ in range(n_warmup):
        fun(**kwargs)
    # Don't let pending warmup work leak into the timed window
    synchronize_device()
    # One synchronized call, only used to size the batches below. On a
    # GPU this overestimates the per-call duration, because it cannot
    # pipeline and pays the synchronization latency.
    latency = _timed_calls(fun, kwargs, 1)
    if backend.is_gpu:
        _warmup_gpu(fun, kwargs, latency, gpu_warmup_duration)
        # The clock speed has changed, so the first estimate is stale.
        latency = _timed_calls(fun, kwargs, 1)
    n_probe_calls = _calls_for_duration(
        probe_duration_fraction * target_repeat_duration, latency, n_runs
    )
    probe_duration = _timed_calls(fun, kwargs, n_probe_calls)
    runs_per_repeat = _calls_for_duration(
        target_repeat_duration, probe_duration / n_probe_calls, n_runs
    )
    repeat_durations = [
        _timed_calls(fun, kwargs, runs_per_repeat) for _ in range(n_repeats)
    ]
    return statistics.median(repeat_durations) / runs_per_repeat


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


def _warmup_gpu(
    fun: Callable[..., Any],
    kwargs: Mapping[str, Any],
    duration_per_call: float,
    gpu_warmup_duration: float,
    batch_duration: float = 0.05,
) -> None:
    """
    Call `fun` under sustained load until the GPU has clocked up.

    The calls are issued in batches that are synchronized before the
    clock is read again. Without that synchronization the host would
    queue kernels far faster than the device retires them, and the
    warmup would keep the device busy for much longer than
    `gpu_warmup_duration`.

    Parameters
    ----------
    fun
        Function to be called.
    kwargs
        Keyword arguments passed to `fun`.
    duration_per_call
        Estimated duration of one call, in seconds.
    gpu_warmup_duration
        Duration of the warmup, in seconds.
    batch_duration
        Duration one batch of calls aims for, in seconds. It sets by how
        much the warmup can overshoot `gpu_warmup_duration`.
    """
    calls_per_batch = _calls_for_duration(
        batch_duration, duration_per_call, n_calls_max=100_000
    )
    warmup_end = time.perf_counter() + gpu_warmup_duration
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
    on_result: Callable[[Any, float], None] | None = None,
) -> list[float]:
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
    on_result
        Called with ``(scan_value, runtime)`` as soon as a scan point is
        measured, e.g. to update a live plot.

    Returns
    -------
    runtimes
        Runtime per call for each entry of `scan_values`, in seconds.
    """
    runtimes = []
    for scan_value in scan_values:
        kwargs = make_kwargs(scan_value)
        mean_runtime = runtime(
            fun=fun,
            kwargs=kwargs,
            n_warmup=n_warmup,
            n_runs=n_runs,
            n_repeats=n_repeats,
        )
        # Release this scan point's arrays before the next, larger, ones
        # are allocated
        del kwargs
        free_device_memory()
        runtimes.append(mean_runtime)
        if on_result is not None:
            on_result(scan_value, mean_runtime)
    return runtimes


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
) -> tuple[list[Any], list[float], Callable[[Any, float], None]]:
    """
    Create a callback that appends a point to a mode's lines and redraws.

    Parameters
    ----------
    line_per_call
        Line of one backend mode in the runtime-per-call panel.
    line_per_element
        Line of the same mode in the runtime-per-element panel.

    Returns
    -------
    scanned_values
        List the callback appends the scan values to.
    runtimes
        List the callback appends the measured runtimes to.
    update_plot
        Callback for `scan_performance`'s `on_result`.
    """
    scanned_values: list[Any] = []
    runtimes: list[float] = []

    def update_plot(scan_value: Any, mean_runtime: float) -> None:
        scanned_values.append(scan_value)
        runtimes.append(mean_runtime)
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
        for line in (line_per_call, line_per_element):
            line.axes.relim()
            line.axes.autoscale_view()
        if _is_interactive_backend():
            plt.draw()
            plt.pause(0.01)

    return scanned_values, runtimes, update_plot


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
    modes: Sequence[str] = ALL_MODES,
    xlabel: str = "scan value",
    title: str | None = None,
    save_name: str | None = None,
) -> dict[str, list[float]]:
    """
    Scan the runtime of a `Specials` kernel on several backends and plot it.

    The plot is redrawn after every scan point, so slow kernels show
    progress while the scan is still running. Call ``plt.show()``
    afterwards to keep the window open.

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
    runtimes
        Runtime per call in seconds for each scan value, per mode.
        Skipped modes contain the points measured before the failure.
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

    runtimes: dict[str, list[float]] = {}
    measured_values: dict[str, list[Any]] = {}
    failures: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "kernel_name": kernel_name,
        "title": title,
        "xlabel": xlabel,
        "host": platform.node(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "scan_values": list(scan_values),
        "measured_values": measured_values,
        "runtimes_per_call_seconds": runtimes,
        "failures": failures,
    }
    for mode in modes:
        (line_per_call,) = axis_per_call.plot([], [], "o-", label=mode)
        (line_per_element,) = axis_per_element.plot([], [], "o-", label=mode)
        axis_per_call.legend()
        mode_values, mode_runtimes, update_plot = _live_line_updater(
            line_per_call, line_per_element
        )
        runtimes[mode] = mode_runtimes
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
                on_result=update_plot,
            )
        except Exception as exc:  # NOQA: BLE001 benchmark continues
            message = f"{type(exc).__name__}: {exc}"
            print(f"Skipped mode {mode!r}: {message}")
            failures[mode] = message
            line_per_call.set_label(f"{mode} (failed)")
            axis_per_call.legend()
        free_device_memory()
        print(f"{mode} runtimes per call [s]: {mode_runtimes}", flush=True)
        if save_name is not None:
            _save_results(save_name, figure, metadata)
    return runtimes
