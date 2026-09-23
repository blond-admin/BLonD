"""
Timing and profiling harness for the LHC cavity feedback.

Builds a 72-bunch LHC scenario, reports the wall-clock time of each stage,
and checks the detuning and RF power against the half-detuning model.
Set `PROFILE_PRETRACK` to profile the no-beam pre-tracking and stop there.
"""

import copy
import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    proton,
)
from blond.physics.feedbacks.accelerators.lhc import (
    LHCCavityFeedback,
    LHCCavityFeedbackCommissioning,
)

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
rf_voltage = 5e6
rf_phase = 0.0
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

energy = np.sqrt(momentum**2 + proton.mass**2)
rel_gamma = energy / proton.mass
rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

n_macroparticles = 100_000
tau_bunch = 1.2e-9
number_of_bunches = 72
bunch_spacing = 10
bucket_shift = 0

g_a = 6.79e-6
g_d = 10
g_o = 10
tau_a = 170e-6
tau_d = 400e-6
tau_o = 110e-6
tau_loop = 650e-9
tau_otfb = 1200e-9


# Profile the no-beam pre-tracking and stop right after it -- the per-turn
# cost of a pretrack turn is within ~10 % of a full track turn, so it is a
# cheap, beam-free proxy for the hot loop.
PROFILE_PRETRACK = False
PROFILE_N_ROWS = 25

timings: dict[str, float] = {}
pretrack_profile = cProfile.Profile()


@contextmanager
def timed(label: str):
    """
    Measure the wall-clock time of a block and report it.

    Parameters
    ----------
    label
        Name the measured section is reported and summed under.

    Yields
    ------
    None
        Control returns to the measured block.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        timings[label] = timings.get(label, 0.0) + duration
        print(f"[time] {label:<34s} {duration:8.3f} s", flush=True)


def print_profile_results(profile: cProfile.Profile, label: str):
    """
    Print a cProfile result, by total time and by cumulative time.

    Parameters
    ----------
    profile
        The profiler holding the collected statistics.
    label
        Name of the profiled section.
    """
    for sort_key in ("tottime", "cumtime"):
        stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stream)
        stats.sort_stats(sort_key).print_stats(PROFILE_N_ROWS)
        print(f"\n[profile] {label} -- sorted by {sort_key}")
        print(stream.getvalue())


def print_timing_summary():
    """
    Print all measured sections, slowest first.

    Labels ending in "(total)" wrap other measured sections, so they are
    excluded from the total to avoid double counting.
    """
    total = sum(
        duration
        for label, duration in timings.items()
        if not label.endswith("(total)")
    )
    print("\n[time] ---- summary (slowest first) ----")
    for label, duration in sorted(timings.items(), key=lambda item: -item[1]):
        if label.endswith("(total)"):
            print(f"[time] {label:<34s} {duration:8.3f} s  (wrapper)")
            continue
        share = 100.0 * duration / total if total else 0.0
        print(f"[time] {label:<34s} {duration:8.3f} s  ({share:5.1f} %)")
    print(f"[time] {'TOTAL (measured)':<34s} {total:8.3f} s")


def create_scenario(
    commissioning: LHCCavityFeedbackCommissioning = None,
    disable_fine_grid: bool = False,
    n_turns: int = 20,
    q_l: float = 20_000,
    n_pretrack: int = 100,
    detuning: float = 0.0,
):
    """
    Build a 72-bunch LHC scenario with an active cavity feedback.

    Parameters
    ----------
    commissioning
        Feedback gains, delays and hardware switches.
    disable_fine_grid
        Whether to skip the fine-grid cavity response.
    n_turns
        Number of turns the simulation is prepared for.
    q_l
        Cavity loaded quality factor.
    n_pretrack
        Number of turns to pre-track without beam.
    detuning
        Offset [Hz] added to the central cavity frequency.

    Returns
    -------
    simulation
        The prepared simulation.
    beam
        The prepared bunch train.
    """
    beam = Beam(
        intensity,
        proton,
    )

    cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

    lattice = DriftSimple(
        orbit_length=circumference, momentum_compaction_factor=alpha
    )

    cavity = SingleHarmonicRFStation(
        voltage=rf_voltage,
        phi_rf=rf_phase,
        harmonic=h,
    )

    f_rf = cavity.calc_main_harmonic_omega_rf_design(
        rel_beta, lattice.orbit_length
    ) / (2 * np.pi)
    t_rf = 1 / f_rf

    profile = StaticProfile(
        cut_left=(-5.5 + bucket_shift) * t_rf,
        cut_right=(6.5 + number_of_bunches * bunch_spacing + bucket_shift)
        * t_rf,
        n_bins=(10 * number_of_bunches + 12) * 2**5,
    )

    bigaussian = BiGaussian(
        n_macroparticles=n_macroparticles,
        sigma_dt=tau_bunch / 4,
        seed=1234,
    )

    with timed("LHCCavityFeedback init (pretrack)"):
        cavity_feedback = LHCCavityFeedback(
            profile,
            tau_loop=tau_loop,
            tau_otfb=tau_otfb,
            commissioning=commissioning,
            q_l=q_l,
            n_pretrack=n_pretrack,
            f_c=detuning + 400.789e6,
        )
    cavity_feedback.disable_fine_grid = disable_fine_grid

    cavity.attach_cavity_feedback(cavity_feedback)

    ring = Ring(
        circumference,
    )

    ring.add_elements(
        [profile, cavity, lattice],
    )

    simulation = Simulation(
        ring,
        cycle,
    )

    with timed("simulation.prepare_beam"):
        simulation.prepare_beam(beam, bigaussian)

    with timed("beam replication (bunch train)"):
        beam_copy = copy.deepcopy(beam)

        for _i in range(1, number_of_bunches):
            _dt = beam_copy.write_partial_dt()
            _dt += bunch_spacing * t_rf
            beam.add_beam(beam_copy)

        _dt = beam.write_partial_dt()

    # The n_pretrack no-beam turns are run from
    # LHCCavityFeedback.on_run_simulation(), i.e. inside finalize() --
    # wrap the call so that cost shows up as its own entry.
    untimed_track_no_beam = cavity_feedback.track_no_beam

    def timed_track_no_beam(n_pretrack: int = 1) -> None:
        with timed(f"pretrack, no beam ({n_pretrack} turns)"):
            if PROFILE_PRETRACK:
                pretrack_profile.enable()
            try:
                untimed_track_no_beam(n_pretrack)
            finally:
                if PROFILE_PRETRACK:
                    pretrack_profile.disable()

    cavity_feedback.track_no_beam = timed_track_no_beam

    with timed("simulation.finalize (total)"):
        simulation.finalize(
            (beam,),
            n_turns,
        )

    with timed("initial profile.track"):
        profile.track(beam)

    return simulation, beam


commissioning = LHCCavityFeedbackCommissioning(
    g_a=g_a,
    g_d=g_d,
    g_o=g_o,
    tau_a=tau_a,
    tau_d=tau_d,
    tau_o=tau_o,
    open_tuner=False,
    mu=-10,
    open_otfb=False,
    enable_klystron=False,
    clamping=False,
    saturation=False,
)
with timed("create_scenario (total)"):
    simulation, beam = create_scenario(
        commissioning=commissioning, n_pretrack=50, disable_fine_grid=True
    )

if PROFILE_PRETRACK:
    print_profile_results(pretrack_profile, "pretrack, no beam")
    print_timing_summary()
    sys.exit(0)

rf_station = simulation.ring.elements.get_element(SingleHarmonicRFStation)
cavity_feedback: LHCCavityFeedback = (
    rf_station.get_main_harmonic_cavity_feedback()
)
with timed("rf_beam_current"):
    cavity_feedback.rf_beam_current(beam)

theoretical_detuning = cavity_feedback.half_detuning(
    imag_peak_beam_current=np.max(
        np.abs(cavity_feedback.buffers_coarse.i_beam.curr)
    ),
    r_over_q=cavity_feedback.r_over_q,
    rf_frequency=rf_station.omega_rf / 2 / np.pi,
    voltage=np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr)),
)

q_l_optimum = cavity_feedback.optimum_Q_L(
    detuning=theoretical_detuning,
    rf_frequency=rf_station.omega_rf / 2 / np.pi,
)
cavity_feedback.q_l = q_l_optimum

theoretical_rf_power = cavity_feedback.half_detuning_power(
    peak_beam_current=np.max(
        np.abs(cavity_feedback.buffers_coarse.i_beam.curr)
    ),
    voltage=np.mean(np.abs(cavity_feedback.buffers_coarse.v_ant.curr)),
)

n_tracking_turns = 100
turn_durations = np.zeros(n_tracking_turns)

with timed("cavity_feedback.track loop"):
    for i in range(n_tracking_turns):
        turn_start = time.perf_counter()
        cavity_feedback.track(beam)
        turn_durations[i] = time.perf_counter() - turn_start

print(
    f"[time] per turn: mean {turn_durations.mean() * 1e3:.1f} ms, "
    f"min {turn_durations.min() * 1e3:.1f} ms, "
    f"max {turn_durations.max() * 1e3:.1f} ms, "
    f"first turn {turn_durations[0] * 1e3:.1f} ms"
)

model_detuning = cavity_feedback.d_omega / 2 / np.pi
model_rf_power = np.mean(cavity_feedback.generator_power())

# `places=5` in the unittest sense: agreement to 5 decimal places,
# i.e. an absolute tolerance of 0.5e-5 on the ratio.
places = 5
tolerance = 0.5 * 10 ** (-places)

assert np.isclose(
    theoretical_detuning / model_detuning,
    1.0136520554274586,
    rtol=0.0,
    atol=tolerance,
), theoretical_detuning / model_detuning

assert np.isclose(
    theoretical_rf_power / float(model_rf_power),
    1.0229780380697582,
    rtol=0.0,
    atol=tolerance,
), theoretical_rf_power / float(model_rf_power)

print_timing_summary()
