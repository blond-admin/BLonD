# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""
Benchmark deferred (queued, fused) kernel execution against eager execution.

Runs the EX_23 PSB setup (two wakefields, one RF station, one drift) for a
fixed number of turns and reports milliseconds/turn, the number of queued
flushes per turn, and a checksum of the final ``dE`` -- which must be
identical across modes on the same device, since deferred execution changes
only *when* a kernel runs, never *what* it computes.

Usage
-----
.. code-block:: bash

    .venv/bin/python dev_tools/performance_blond3/backends/deferred_psb.py \\
        cpp_deferred --one-histogram

    # single-core cycle counts (pin threads through BLonD's own mechanism,
    # NOT OMP_NUM_THREADS -- ``import blond`` overrides it):
    perf stat -e cycles -- .venv/bin/python \\
        dev_tools/performance_blond3/backends/deferred_psb.py \\
        cpp --single-core --n-turns 10

Run each configuration multiple times, interleaved, to average out turbo
drift -- see ``task-11-brief.md`` for the exact loop used to produce the
spec's benchmark table.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

# EX_23 calls `logging.basicConfig(level=logging.INFO)` at import time,
# which would otherwise drown the benchmark output in per-element setup
# logs. Quiet it down before importing it.
logging.basicConfig(level=logging.WARNING)


@dataclass
class BenchmarkResult:
    """Outcome of one benchmark run."""

    mode: str
    n_macroparticles: int
    n_turns: int
    ms_per_turn: float
    flushes_per_turn: float
    batches_per_turn: float
    checksum: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark deferred vs eager kernel execution on the EX_23 "
            "PSB setup."
        )
    )
    parser.add_argument(
        "mode",
        choices=("cpp", "cpp_deferred", "cuda", "cuda_deferred"),
        help="Specials mode to benchmark.",
    )
    parser.add_argument(
        "--n-macroparticles",
        type=float,
        default=1e7,
        help="Number of macroparticles (default: 1e7).",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=30,
        help="Number of timed turns, after warm-up (default: 30).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10_000,
        help="Number of profile bins (default: 10000).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of untimed warm-up turns (default: 3).",
    )
    parser.add_argument(
        "--one-histogram",
        action="store_true",
        help=(
            "Set track_profile=False on both wakefields, so only the "
            "StaticProfile element histograms each turn. Reproduces the "
            "spec section 9 rows."
        ),
    )
    parser.add_argument(
        "--single-core",
        action="store_true",
        help=(
            "cpp / cpp_deferred only: load the non-OMP ('_noOMP') C++ "
            "library instead of setting BLOND_BACKEND_MODE, which "
            "guarantees single-threaded execution regardless of "
            "OMP_NUM_THREADS (importing blond overrides that variable "
            "before user code can set it)."
        ),
    )
    return parser.parse_args(argv)


def _set_up_backend(mode: str, single_core: bool):
    """Activate `mode`, returning the active `backend` module object."""
    from blond.core.backends.backend import backend

    if single_core:
        if mode not in ("cpp", "cpp_deferred"):
            raise ValueError(
                f"--single-core only supports cpp/cpp_deferred, got {mode!r}."
            )
        from blond.core.backends.backend import Numpy64Bit
        from blond.core.backends.cpp.callables import reload_cpp_backend

        backend.change_backend(Numpy64Bit)
        backend.specials = reload_cpp_backend(
            backend.float,
            parallel=False,
            deferred=(mode == "cpp_deferred"),
        )
        backend.specials_mode = mode
    else:
        from blond.core.backends.helpers import setup_backend

        setup_backend(mode)
    return backend


def _wrap_flush_counter(specials) -> list[int]:
    """
    Count flush calls and executed batches, returning the counter cells.

    `flush` is replaced on the active specials, so this intercepts every
    explicit ``specials.flush()`` -- queued or not -- without touching
    `Specials` itself. The deferred flush looks ``_execute_batch`` up on
    its class at call time, so wrapping that counts every batch actually
    run, including the implicit flushes of flush-then-call methods such
    as ``histogram``.

    Returns ``[calls, batches]``.
    """
    count = [0, 0]
    original_flush = specials.flush

    def counting_flush() -> None:
        count[0] += 1
        original_flush()

    specials.flush = counting_flush

    # The cpp specials are a class, the cuda ones may be an instance.
    deferred_class = specials if isinstance(specials, type) else type(specials)
    if "_execute_batch" in vars(deferred_class):
        original_execute_batch = deferred_class._execute_batch

        def counting_execute_batch(*args, **kwargs) -> None:
            count[1] += 1
            original_execute_batch(*args, **kwargs)

        deferred_class._execute_batch = staticmethod(counting_execute_batch)
    return count


def run_benchmark(
    mode: str,
    n_macroparticles: int,
    n_turns: int,
    n_bins: int,
    n_warmup: int,
    one_histogram: bool,
    single_core: bool,
) -> BenchmarkResult:
    """Build the EX_23 PSB setup and time `n_turns` of tracking."""
    from blond import WakeField
    from blond.examples.scripts.EX_23_Main_long_ps_booster import build
    from blond.generals.cupy_.no_cupy_import import copy_to_cpu

    active_backend = _set_up_backend(mode, single_core)

    sim, beam = build(
        n_macroparticles=n_macroparticles,
        n_bins=n_bins,
    )

    if one_histogram:
        wakefields = sim.ring.elements.get_elements(WakeField, recursive=False)
        if not wakefields:
            raise RuntimeError("EX_23 build() no longer has WakeFields.")
        for wakefield in wakefields:
            wakefield.track_profile = False

    flush_count = _wrap_flush_counter(active_backend.specials)

    # Warm-up: JIT/compile caches, first-touch allocations, not timed.
    if n_warmup:
        sim.run_simulation(
            beams=(beam,),
            n_turns=n_warmup,
            show_progressbar=False,
            verbose=False,
        )
    flush_count[0] = flush_count[1] = 0

    start = time.perf_counter()
    sim.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        show_progressbar=False,
        verbose=False,
    )
    if active_backend.is_gpu:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - start

    dE = copy_to_cpu(beam.read_partial_dE())
    checksum = float(dE.sum())

    return BenchmarkResult(
        mode=mode,
        n_macroparticles=n_macroparticles,
        n_turns=n_turns,
        ms_per_turn=elapsed * 1e3 / n_turns,
        flushes_per_turn=flush_count[0] / n_turns,
        batches_per_turn=flush_count[1] / n_turns,
        checksum=checksum,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Run one benchmark configuration and print its result."""
    args = _parse_args(argv)
    result = run_benchmark(
        mode=args.mode,
        n_macroparticles=int(args.n_macroparticles),
        n_turns=args.n_turns,
        n_bins=args.n_bins,
        n_warmup=args.n_warmup,
        one_histogram=args.one_histogram,
        single_core=args.single_core,
    )
    print(
        f"mode={result.mode} "
        f"n_macroparticles={result.n_macroparticles:.0f} "
        f"n_turns={result.n_turns} "
        f"single_core={args.single_core} "
        f"one_histogram={args.one_histogram}"
    )
    print(f"ms/turn: {result.ms_per_turn:.3f}")
    print(f"flushes/turn: {result.flushes_per_turn:.2f}")
    print(f"batches/turn: {result.batches_per_turn:.2f}")
    print(f"checksum(dE): {result.checksum!r}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
