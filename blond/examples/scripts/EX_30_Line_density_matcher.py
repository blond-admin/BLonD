# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Matched bunch from a measured line density, tracked for stationarity.

A bunch profile — here synthesized, in practice loaded from a
measurement (e.g. a wall-current-monitor trace) — is fed to the
``LineDensityMatcher``: the profile is recentred in the LHC 450 GeV
bucket (measured profiles are arbitrarily positioned) and Abel-inverted
over the analytic potential well into the phase-space distribution that
reproduces it. The generated bunch is tracked (10 000 turns by default)
and must keep its bunch length and position stationary.

Measured profiles must be clean: the Abel transform differentiates the
profile, so noise is amplified — baseline subtraction and any filtering
are the user's responsibility (an automatic filter can bias the
profile).

With ``INTENSITY_EFFECTS = True`` a broadband resonator impedance is
added to the ring: the matcher then iterates the profile centering and
the induced potential to self-consistency, and the bunch is matched at
the wake-shifted stable position. A symmetric measured profile cannot
be exactly stationary in the distorted (asymmetric) well: with
``half_option="first"`` the inversion uses the left well branch only,
so the matcher plot (peak-normalized) shows the reconstruction
overlaying the measured first half exactly while the second half
deviates; ``half_option="both"`` averages the two branches instead,
spreading the deviation. The reported profile-reconstruction error
quantifies it either way.
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
from blond.experimental.beam_preparation.analytic_distributions import (
    line_density,
)
from blond.experimental.beam_preparation.analytic_matcher import (
    LineDensityMatcher,
)
from blond.physics.impedances.solvers import TimeDomainFftSolver
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

N_TURNS = 10_000
N_MACROPARTICLES = 1e5
N_POINTS_GRID = 1000
N_POINTS_ABEL = 10_000
HALF_OPTION = "first"

# The synthetic "measured" profile: a binomial bunch sampled on a
# scope-like axis (25 ps bins), deliberately off-centre with a small
# constant baseline — the matcher recentres and baseline-subtracts.
PROFILE_FULL_LENGTH = 1.6e-9  # s
PROFILE_EXPONENT = 1.5
PROFILE_POSITION_OFFSET = 0.15e-9  # s, arbitrary measurement timing
PROFILE_BASELINE = 0.02
PROFILE_N_SAMPLES = 81

# Intensity effects: simple broadband resonator, matched
# self-consistently and kept active during tracking.
INTENSITY_EFFECTS = True
BEAM_INTENSITY = 2e11  # particles per bunch
RESONATOR_R_SHUNT = 1e5  # Ohm
RESONATOR_FREQUENCY = 8e8  # Hz
RESONATOR_QUALITY_FACTOR = 1.0


def main():
    # "Measured" bunch profile ----------------------------------------
    # In practice: load the time axis and profile from your measurement
    # and pass them straight to the matcher.
    measured_time = np.linspace(-1.0e-9, 1.0e-9, PROFILE_N_SAMPLES)
    measured_profile = (
        line_density(
            measured_time,
            "binomial",
            PROFILE_FULL_LENGTH,
            bunch_position=PROFILE_POSITION_OFFSET,
            exponent=PROFILE_EXPONENT,
        )
        + PROFILE_BASELINE
    )

    # Machine: LHC at 450 GeV, single harmonic ------------------------
    ring = Ring(26658.883)
    rf_station = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
    drift = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
    )
    elements = [rf_station, drift]
    if INTENSITY_EFFECTS:
        # Simple impedance model: one broadband resonator, driven by
        # the tracked bunch profile over one RF bucket.
        profile = StaticProfile(
            cut_left=0.0,
            cut_right=2.4951e-9,  # one RF period at 450 GeV
            n_bins=512,
        )
        wakefield = WakeField(
            sources=(
                Resonators(
                    RESONATOR_R_SHUNT,
                    RESONATOR_FREQUENCY,
                    RESONATOR_QUALITY_FACTOR,
                ),
            ),
            solver=TimeDomainFftSolver(),
            profile=profile,
        )
        elements += [wakefield, profile]
    ring.add_elements(elements, reorder=True)
    magnetic_cycle = ConstantMagneticCycle(
        value=450e9, reference_particle=proton
    )
    beam = Beam(intensity=BEAM_INTENSITY, particle_type=proton)
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

    # Matched beam from the measured profile --------------------------
    # The profile is always recentred onto the potential-well minimum;
    # with INTENSITY_EFFECTS the centering and the induced potential
    # iterate together (see `relaxation_factor` if a stronger impedance
    # makes the iteration oscillate).
    matcher = LineDensityMatcher(
        n_macroparticles=N_MACROPARTICLES,
        time_array=measured_time,
        line_density_values=measured_profile,
        half_option=HALF_OPTION,
        n_points_abel=N_POINTS_ABEL,
        seed=0,
        n_points_grid=N_POINTS_GRID,
        verbose=True,
        # Draws the input line density against the reconstructed
        # density and the generated macroparticle profile (and the
        # RF vs distorted well when intensity effects are active).
        plot=True,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)
    target_bunch_length = matcher.matched_bunch_length

    # Phase space before/after figure ---------------------------------
    fig1, (ax_before, ax_after) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharex=True,
        sharey=True,
        num="EX_30_phase_space",
    )
    plt.sca(ax_before)
    beam.plot_hist2d(bins=150)
    ax_before.set_title("before tracking")
    ax_before.set_xlabel("Time [s]")
    ax_before.set_ylabel("Energy offset [eV]")

    # Track and record bunch length/position every turn ---------------
    bunch_length_turns = []
    bunch_position_turns = []

    def record_bunch(simulation, beam):
        # NumPy reductions dispatch to the beam array on any backend.
        dt = beam.read_partial_dt()
        bunch_position_turns.append(float(np.mean(dt)))
        bunch_length_turns.append(float(4.0 * np.std(dt)))

    simulation.run_simulation(
        beams=(beam,), n_turns=N_TURNS, callbacks=record_bunch
    )

    bunch_length_turns = np.array(bunch_length_turns)
    bunch_position_turns = np.array(bunch_position_turns)

    plt.sca(ax_after)
    beam.plot_hist2d(bins=150)
    ax_after.set_title(f"after {N_TURNS} turns")
    fig1.suptitle("Bunch matched from a measured profile, LHC 450 GeV")
    fig1.tight_layout()

    # Stationarity summary --------------------------------------------
    length_mean = bunch_length_turns.mean()
    length_variation = bunch_length_turns.std() / length_mean
    length_drift = (
        bunch_length_turns[-100:].mean() - bunch_length_turns[:100].mean()
    ) / length_mean
    position_span = np.ptp(bunch_position_turns)
    print(
        f"\nBunch length : mean {length_mean * 1e9:.4f} ns "
        f"(matched {target_bunch_length * 1e9:.4f} ns, 4 sigma rms), "
        f"rms variation {length_variation:.2%}, "
        f"drift over the run {length_drift:+.2%}"
    )
    print(
        f"Bunch position: mean {bunch_position_turns.mean() * 1e9:.4f} "
        f"ns, peak-to-peak excursion {position_span * 1e12:.2f} ps"
    )
    print(
        "Profile reconstruction error "
        f"{matcher.profile_reconstruction_error:.2%} (input vs "
        "Abel-reconstructed line density"
        + (
            "; with intensity effects a symmetric input cannot be "
            "exactly stationary in the distorted well)"
            if INTENSITY_EFFECTS
            else ")"
        )
    )

    # Stationarity plot -----------------------------------------------
    fig2, (ax_length, ax_position) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, num="EX_30_stationarity"
    )
    turns = np.arange(len(bunch_length_turns))
    ax_length.plot(turns, bunch_length_turns * 1e9, color="C0", lw=0.8)
    ax_length.axhline(
        target_bunch_length * 1e9,
        color="k",
        ls="--",
        lw=1.0,
        label="matched (4 sigma rms)",
    )
    ax_length.set_ylabel("Bunch length [ns]")
    ax_length.legend(loc="upper right")
    ax_length.grid(alpha=0.3)

    ax_position.plot(turns, bunch_position_turns * 1e9, color="C1", lw=0.8)
    ax_position.axhline(
        bunch_position_turns[0] * 1e9,
        color="k",
        ls="--",
        lw=1.0,
        label="initial position",
    )
    ax_position.set_xlabel("Turn")
    ax_position.set_ylabel("Bunch position [ns]")
    ax_position.legend(loc="upper right")
    ax_position.grid(alpha=0.3)
    fig2.suptitle(f"Stationarity of the matched bunch over {N_TURNS} turns")
    fig2.tight_layout()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
