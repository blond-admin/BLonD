# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Runtime of BLonD 2 vs BLonD 3 for kick, drift, profile and wake.

A PSB ramp (7069 turns, single harmonic RF, one broadband resonator)
is tracked with the same particles in both codes, per backend. Only the
main loop is timed; setup and a short warmup run are excluded. The
result is printed and plotted as a bar chart.

Usage::

    python run_comparison.py --backends numba cpp cuda --n-runs 3

``cpp_deferred`` and ``cuda_deferred`` run BLonD 3 with the queued
C++ / CUDA kernels (CPU chunk size from ``BLOND_DEFERRED_CHUNK_SIZE``).
BLonD 2 has no deferred mode, so it runs its plain C++ / GPU code there.

Without ``--backends``, numba and cpp are run, plus cuda if a GPU is
available. The C++ backends must be compiled first
(``blond-compile-cpp`` for BLonD 3, ``blond/legacy/blond2/compile.py``
for BLonD 2).

Adapted from the IPAC 2026 study by Oliver Muller Smedt and Simon Lauber.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blond import backend
from blond.core.backends.backend import backend_class_for_mode
from blond.legacy.blond2.utils import bmath as bm

RESOURCES = Path(__file__).parent / "resources" / "psb_ramp.npz"

CIRCUMFERENCE = 2 * np.pi * 100  # m
VOLTAGE = 200e3  # V
HARMONIC = 8
INTENSITY = 600e10  # roughly TOF intensity
SHUNT_IMPEDANCE = 1e4  # Ohm
RESONANCE_FREQUENCY = 4e6  # Hz
QUALITY_FACTOR = 3
PROFILE_LENGTH = 2.124873604201372e-06  # s
N_BINS = 1024
N_TURNS_WARMUP = 100
GPU_TARGETS = ("cuda", "cuda_deferred")


class Params:
    """Machine ramp and initial bunch shared by both codes."""

    def __init__(self, n_macroparticles: int, n_turns: int | None):
        ramp = np.load(RESOURCES)
        self.phi_rf = ramp["phi_rf"]  # rad, per turn
        self.transition_gamma = ramp["transition_gamma"]
        self.momentum = ramp["momentum"]  # eV/c
        self.n_turns_ramp = len(self.momentum) - 1
        self.n_turns = 100  # self.n_turns_ramp if n_turns is None else n_turns

        rng = np.random.default_rng(seed=4)
        distribution = rng.standard_normal((n_macroparticles, 2))
        self.initial_dE = distribution[:, 1] * 25e6  # eV
        self.initial_dt = distribution[:, 0] * 1e-8 + 0.35e-6  # s


def _gpu_available() -> bool:
    try:
        import cupy  # noqa: PLC0415  (optional dependency)

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _synchronize_gpu() -> None:
    import cupy  # noqa: PLC0415  (only reached on GPU runs)

    cupy.cuda.Device().synchronize()


def run_blond2(params: Params, n_turns: int, use_gpu: bool) -> float:
    """Track with BLonD 2 and return the main loop runtime in s."""
    from blond.legacy.blond2.beam.beam import Beam, Proton  # noqa: PLC0415
    from blond.legacy.blond2.beam.profile import (  # noqa: PLC0415
        CutOptions,
        Profile,
    )
    from blond.legacy.blond2.impedances.impedance import (  # noqa: PLC0415
        InducedVoltageFreq,
        TotalInducedVoltage,
    )
    from blond.legacy.blond2.impedances.impedance_sources import (  # noqa: PLC0415
        Resonators,
    )
    from blond.legacy.blond2.input_parameters.rf_parameters import (  # noqa: PLC0415
        RFStation,
    )
    from blond.legacy.blond2.input_parameters.ring import (  # noqa: PLC0415
        Ring,
    )
    from blond.legacy.blond2.trackers.tracker import (  # noqa: PLC0415
        FullRingAndRF,
        RingAndRFTracker,
    )

    # objects are built on the host and moved to the GPU afterwards
    if use_gpu:
        bm.use_py()
    ring = Ring(
        CIRCUMFERENCE,
        1 / params.transition_gamma**2,
        params.momentum.copy(),
        Proton(),
        n_turns=params.n_turns_ramp,
    )
    beam = Beam(
        ring,
        len(params.initial_dt),
        INTENSITY,
        dt=params.initial_dt.copy(),
        dE=params.initial_dE.copy(),
    )
    rf_station = RFStation(ring, HARMONIC, VOLTAGE, params.phi_rf.copy())
    profile = Profile(beam, CutOptions(0, PROFILE_LENGTH, N_BINS))
    resonator = Resonators(
        SHUNT_IMPEDANCE, RESONANCE_FREQUENCY, QUALITY_FACTOR
    )
    induced_voltage = InducedVoltageFreq(beam, profile, [resonator])
    total_induced_voltage = TotalInducedVoltage(
        beam, profile, [induced_voltage]
    )
    rf_tracker = RingAndRFTracker(rf_station, beam, solver="simple")
    full_tracker = FullRingAndRF([rf_tracker])
    if use_gpu:
        # `FullRingAndRF.to_gpu` copies the potential well, so it must exist
        full_tracker.potential_well_generation()
        for obj in (
            rf_station,
            profile,
            induced_voltage,
            total_induced_voltage,
            rf_tracker,
            full_tracker,
        ):
            obj.to_gpu()
        bm.use_gpu()

    t0 = time.perf_counter()
    for _ in range(n_turns):
        full_tracker.track()
        profile.track()
        total_induced_voltage.track()
    if use_gpu:
        _synchronize_gpu()
    return time.perf_counter() - t0


def run_blond3(params: Params, n_turns: int, use_gpu: bool) -> float:
    """Track with BLonD 3 and return the main loop runtime in s."""
    from blond import (  # noqa: PLC0415
        Beam,
        DriftSimple,
        MagneticCyclePerTurn,
        Resonators,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
        StaticProfile,
        WakeField,
        momentum_compaction_factor,
        proton,
    )
    from blond.physics.impedances.solvers import (  # noqa: PLC0415
        PeriodicFreqSolver,
    )

    ring = Ring(circumference=CIRCUMFERENCE)
    magnetic_cycle = MagneticCyclePerTurn(
        value_init=float(params.momentum[0]),
        values_after_turn=params.momentum[1:].copy(),
        reference_particle=proton,
    )
    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = HARMONIC
    rf_station.voltage = VOLTAGE
    rf_station.schedule("phi_rf_design", params.phi_rf[:-1].copy())
    drift = DriftSimple(orbit_length=CIRCUMFERENCE)
    drift.schedule(
        "momentum_compaction_factor",
        momentum_compaction_factor(params.transition_gamma[1:].copy()),
    )
    profile = StaticProfile(
        cut_left=0, cut_right=PROFILE_LENGTH, n_bins=N_BINS
    )
    wakefield = WakeField(
        sources=(
            Resonators(
                np.array([SHUNT_IMPEDANCE]),
                np.array([RESONANCE_FREQUENCY]),
                np.array([QUALITY_FACTOR]),
            ),
        ),
        solver=PeriodicFreqSolver(PROFILE_LENGTH, allow_next_fast_len=True),
        profile=profile,
    )
    ring.add_elements(
        (rf_station, drift, wakefield), reorder=False, section_index=0
    )
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

    beam = Beam(intensity=INTENSITY, particle_type=proton)
    beam.setup_beam(
        dt=params.initial_dt.copy(),
        dE=params.initial_dE.copy(),
        reference_total_energy=magnetic_cycle.get_total_energy_init(
            particle_type=beam.particle_type
        ),
    )
    profile.track(beam=beam)
    simulation.finalize(beams=(beam,), n_turns=n_turns)

    t0 = time.perf_counter()
    simulation.mainloop(beams=(beam,), n_turns=n_turns, show_progressbar=False)
    if use_gpu:
        _synchronize_gpu()
    return time.perf_counter() - t0


def set_backends(target: str) -> None:
    """Activate `target` in both BLonD 3 and BLonD 2."""
    backend.change_backend(backend_class_for_mode(target))
    backend.set_specials(target)
    {
        "cpp": bm.use_cpp,
        # BLonD 2 has no deferred mode; its plain C++ is the reference
        "cpp_deferred": bm.use_cpp,
        "cuda": bm.use_gpu,
        "cuda_deferred": bm.use_gpu,
        "numba": bm.use_numba,
        "python": bm.use_py,
    }[target]()


def measure(
    params: Params, targets: list[str], n_runs: int
) -> dict[str, dict[str, list[float]]]:
    """Time `n_runs` main loops per code and backend, after a warmup."""
    runners = {"BLonD 2": run_blond2, "BLonD 3": run_blond3}
    runtimes: dict[str, dict[str, list[float]]] = {
        code: {} for code in runners
    }
    for target in targets:
        use_gpu = target in GPU_TARGETS
        for code, runner in runners.items():
            set_backends(target)
            runner(params, N_TURNS_WARMUP, use_gpu)
            runtimes[code][target] = []
            for run_i in range(n_runs):
                runtime = runner(params, params.n_turns, use_gpu)
                runtimes[code][target].append(runtime)
                print(
                    f"{code} [{target}] run {run_i + 1}/{n_runs}: "
                    f"{runtime:.3f} s"
                )
    return runtimes


def plot(
    runtimes: dict[str, dict[str, list[float]]],
    targets: list[str],
    title: str,
    output: Path,
) -> None:
    """Bar chart of mean runtime with standard deviation per backend."""
    x = np.arange(len(targets))
    width = 0.35
    fig, ax = plt.subplots()
    for code_i, (code, per_target) in enumerate(runtimes.items()):
        means = [np.mean(per_target[t]) for t in targets]
        stds = [np.std(per_target[t]) for t in targets]
        bars = ax.bar(
            x + (code_i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=code,
            capsize=4,
        )
        ax.bar_label(
            bars,
            labels=[
                f"{m:.2f}\n±{s:.2f}" for m, s in zip(means, stds, strict=True)
            ],
            fontsize=7,
        )
    ax.set_xticks(x, targets)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(title)
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Saved {output}")
    plt.show()


def main() -> None:
    """Parse the command line, measure and plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=[
            "numba",
            "cpp",
            "cpp_deferred",
            "cuda",
            "cuda_deferred",
            "python",
        ],
        help="default: numba cpp, plus cuda if a GPU is available",
    )
    parser.add_argument("--n-macroparticles", type=float, default=1e4)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--n-turns", type=int, default=None, help="default: full ramp"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("blond2_vs_blond3.png")
    )
    args = parser.parse_args()
    if args.backends is None:
        args.backends = ["cpp"]
        if _gpu_available():
            args.backends.append("cuda")
        print(f"Backends: {' '.join(args.backends)}")

    params = Params(int(args.n_macroparticles), args.n_turns)
    runtimes = measure(params, args.backends, args.n_runs)
    title = (
        f"PSB, {params.n_turns} turns, "
        f"{args.n_macroparticles:.0e} macroparticles"
    )
    plot(runtimes, args.backends, title, args.output)


if __name__ == "__main__":
    main()
