# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Multi-bunch train matched all bunches at once, tracked for stationarity.

The same LHC 450 GeV train as ``EX_31_Sequential_multibunch_matcher``
(same bunches, intensities, and impedance) is generated with the
``SelfConsistentMultiBunchMatcher``: all bunches are (re)matched
together against the induced voltage of the entire train, iterated to a
global fixed point. The train profile inside the iteration is the
accumulated smooth line density from the per-bunch 2D density matrices
— no macroparticles are sampled while iterating, unlike the original
BLonD 2 ``match_beam_from_distribution`` which histogrammed generated
particles (injecting sampling noise into the fixed point).

With the default open-boundary (causal) wake, the sequential
bunch-by-bunch method of EX_31 already sits at the self-consistent
fixed point: running both scripts demonstrates that the two approaches
give the same matched train.

With ``PERIODIC_CONDITIONS = True`` the impedance is solved in the
frequency domain with periodic boundary conditions
(``PeriodicFreqSolver``) over one train period, and the matcher treats
the induced voltage as periodic (``train_periodicity``): the wake of
the trailing bunches wraps around onto the head of the train — a beam
configuration the sequential method cannot represent (the first bunch
is then matched off the bare-bucket position too).
"""

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    AnalyticDistributionMatcher,
)
from blond.experimental.beam_preparation.analytic_multibunch import (
    SelfConsistentMultiBunchMatcher,
)
from blond.physics.impedances.solvers import (
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

N_TURNS = 10_000
N_MACROPARTICLES_PER_BUNCH = 2.5e4
N_POINTS_GRID = 1000

# The train: four bunches, individually parameterized (identical to
# EX_31, to compare the two multi-bunch matching methods).
N_BUNCHES = 4
BUNCH_SPACING_BUCKETS = 10
BUNCH_LENGTHS = [1.2e-9, 1.1e-9, 1.3e-9, 1.2e-9]  # s, 4 sigma rms
BUNCH_INTENSITIES = [2.0e11, 1.6e11, 2.4e11, 2.0e11]  # particles

# Long-memory resonator: the wake decay time 2Q/omega ~ 16 ns spans
# several buckets, coupling the bunches along the train.
RESONATOR_R_SHUNT = 1e5  # Ohm
RESONATOR_FREQUENCY = 2e8  # Hz
RESONATOR_QUALITY_FACTOR = 10.0

# Periodic conditions: frequency-domain solver over one train period,
# the wake wraps around the train (see the module docstring).
PERIODIC_CONDITIONS = False


def main():
    # Machine: LHC at 450 GeV, single harmonic ------------------------
    ring = Ring(26658.883)
    rf_station = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=sum(BUNCH_INTENSITIES), particle_type=proton)
    # The RF period sets the profile span and the per-bunch analysis
    # windows below: derive it from the machine objects (the design
    # revolution frequency needs the reference energy).
    beam.reference.total_energy = magnetic_cycle.get_total_energy_init(
        particle_type=proton
    )
    rf_period = (
        2.0
        * np.pi
        / rf_station.calc_main_harmonic_omega_rf_design(
            beam_beta=beam.reference.beta,
            ring_circumference=ring.circumference,
        )
    )
    if PERIODIC_CONDITIONS:
        # Periodic wake: the tracked profile and the solver period span
        # exactly one train period.
        train_periodicity = N_BUNCHES * BUNCH_SPACING_BUCKETS * rf_period
        profile = StaticProfile(
            cut_left=0.0, cut_right=train_periodicity, n_bins=1024
        )
        solver = PeriodicFreqSolver(t_periodicity=train_periodicity)
    else:
        # Open boundary: the profile spans the train plus one bucket
        # for the wake tail behind the last bunch.
        train_periodicity = None
        profile = StaticProfile(
            cut_left=0.0,
            cut_right=(N_BUNCHES * BUNCH_SPACING_BUCKETS + 1) * rf_period,
            n_bins=1024,
        )
        solver = TimeDomainFftSolver()
    wakefield = WakeField(
        sources=(
            Resonators(
                RESONATOR_R_SHUNT,
                RESONATOR_FREQUENCY,
                RESONATOR_QUALITY_FACTOR,
            ),
        ),
        solver=solver,
        profile=profile,
    )
    ring.add_elements([rf_station, drift, wakefield, profile], reorder=True)
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

    # Matched train ---------------------------------------------------
    # Per-bunch parameters are single-bunch matcher instances (clones
    # of a common template); the driver disables their internal
    # self-wake iteration and supplies the full train wake globally.
    template = AnalyticDistributionMatcher(
        n_macroparticles=N_MACROPARTICLES_PER_BUNCH,
        distribution_type="parabolic_amplitude",
        bunch_length=1.2e-9,
        seed=0,
        n_points_grid=N_POINTS_GRID,
        allow_inner_buckets=True,
    )
    matcher = SelfConsistentMultiBunchMatcher(
        bunch_matchers=[
            template.clone(bunch_length=bunch_length, seed=bunch_i)
            for bunch_i, bunch_length in enumerate(BUNCH_LENGTHS)
        ],
        n_bunches=N_BUNCHES,
        bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
        bunch_intensities=BUNCH_INTENSITIES,
        relaxation_factor=0.5,
        train_periodicity=train_periodicity,
        verbose=True,
        # Draws the train line density and its induced voltage at the
        # converged fixed point.
        plot=True,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)

    bucket_indices = matcher.bucket_indices
    bucket_edges = [
        (bucket_index * rf_period, (bucket_index + 1) * rf_period)
        for bucket_index in bucket_indices
    ]

    # Track and record per-bunch length/position every turn -----------
    bunch_length_turns = []
    bunch_position_turns = []

    def record_bunches(simulation, beam):
        # NumPy reductions dispatch to the beam array on any backend.
        dt = beam.read_partial_dt()
        lengths, positions = [], []
        for left_edge, right_edge in bucket_edges:
            selection = (dt > left_edge) & (dt < right_edge)
            bunch_dt = dt[selection]
            positions.append(float(np.mean(bunch_dt)))
            lengths.append(float(4.0 * np.std(bunch_dt)))
        bunch_length_turns.append(lengths)
        bunch_position_turns.append(positions)

    simulation.run_simulation(
        beams=(beam,), n_turns=N_TURNS, callbacks=record_bunches
    )

    bunch_length_turns = np.array(bunch_length_turns)  # (n_turns, n_bunches)
    bunch_position_turns = np.array(bunch_position_turns)

    # Stationarity summary --------------------------------------------
    print()
    for bunch_i, bucket_index in enumerate(bucket_indices):
        lengths = bunch_length_turns[:, bunch_i]
        positions = bunch_position_turns[:, bunch_i]
        length_mean = lengths.mean()
        matched_shift = (
            (bucket_index + 0.5) * rf_period - positions[0]
        ) * 1e12
        print(
            f"Bunch {bunch_i} (bucket {bucket_index:2d}): length mean "
            f"{length_mean * 1e9:.4f} ns "
            f"(matched {BUNCH_LENGTHS[bunch_i] * 1e9:.1f} ns), rms "
            f"variation {lengths.std() / length_mean:.2%}, drift "
            f"{(lengths[-100:].mean() - lengths[:100].mean()) / length_mean:+.2%}, "
            f"matched shift {matched_shift:+.2f} ps, position "
            f"excursion {np.ptp(positions) * 1e12:.2f} ps"
        )

    # Stationarity plot -----------------------------------------------
    fig, (ax_length, ax_position) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, num="EX_32_stationarity"
    )
    turns = np.arange(len(bunch_length_turns))
    for bunch_i, bucket_index in enumerate(bucket_indices):
        ax_length.plot(
            turns,
            bunch_length_turns[:, bunch_i] * 1e9,
            lw=0.8,
            label=f"bunch {bunch_i} (bucket {bucket_index})",
        )
        ax_position.plot(
            turns,
            (
                bunch_position_turns[:, bunch_i]
                - bunch_position_turns[0, bunch_i]
            )
            * 1e12,
            lw=0.8,
        )
    ax_length.set_ylabel("Bunch length [ns]")
    ax_length.legend(loc="upper right", fontsize=8, ncols=2)
    ax_length.grid(alpha=0.3)
    ax_position.set_xlabel("Turn")
    ax_position.set_ylabel("Position deviation [ps]")
    ax_position.grid(alpha=0.3)
    fig.suptitle(
        f"Stationarity of the self-consistently matched {N_BUNCHES}-bunch "
        f"train over {N_TURNS} turns"
        + (" (periodic wake)" if PERIODIC_CONDITIONS else "")
    )
    fig.tight_layout()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
