# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Analytically matched bunch, tracked to check stationarity.

An LHC 450 GeV bunch is matched to a 1.2 ns (4 sigma rms)
parabolic-amplitude distribution with the ``AnalyticDistributionMatcher``
and tracked (10 000 turns by default, ~200 turns per synchrotron period
at these settings). A matched bunch must keep its bunch length and
position stationary: the script records both every turn and reports
their variation.

With ``INTENSITY_EFFECTS = True`` a broadband resonator impedance is
added to the ring: the matcher then iterates the induced potential of
the smooth candidate line density to self-consistency (the distorted
potential well is shown in the matcher plot), and the bunch is tracked
with the wakefield active — the stationarity check then validates the
intensity-effect matching. For impedances strong enough to make the
iteration oscillate, see the matcher's ``relaxation_factor`` option.
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
from blond.physics.impedances.solvers import TimeDomainFftSolver
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

N_TURNS = 10_000
N_MACROPARTICLES = 1e5
N_POINTS_GRID = 1000
TARGET_BUNCH_LENGTH = 1.2e-9  # s, 4 sigma rms
DISTRIBUTION_TYPE = "parabolic_amplitude"

# Intensity effects: simple broadband resonator, matched
# self-consistently and kept active during tracking.
INTENSITY_EFFECTS = True
BEAM_INTENSITY = 2e11  # particles per bunch
RESONATOR_R_SHUNT = 1e5  # Ohm
RESONATOR_FREQUENCY = 8e8  # Hz
RESONATOR_QUALITY_FACTOR = 1.0


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

    # Matched beam ----------------------------------------------------
    # With INTENSITY_EFFECTS the matcher auto-detects the wakefield and
    # iterates the induced potential to self-consistency. If a stronger
    # impedance makes that iteration oscillate, lower the matcher's
    # `relaxation_factor` (e.g. 0.5).
    simulation.prepare_beam(
        beam=beam,
        preparation_routine=AnalyticDistributionMatcher(
            n_macroparticles=N_MACROPARTICLES,
            distribution_type=DISTRIBUTION_TYPE,
            bunch_length=TARGET_BUNCH_LENGTH,
            seed=0,
            n_points_grid=N_POINTS_GRID,
            verbose=True,
            # Draws the requested (matched density) line density
            # against the generated macroparticle profile (and the
            # RF vs distorted well when intensity effects are active).
            plot=True,
        ),
    )
    # Phase space before/after figure -----------------------------------
    fig1, (ax_before, ax_after) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharex=True,
        sharey=True,
        num="EX_29_phase_space",
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
    ax_after.set_xlabel("Time [s]")
    fig1.suptitle("Matched parabolic-amplitude bunch, LHC 450 GeV")
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
        f"(target {TARGET_BUNCH_LENGTH * 1e9:.4f} ns, 4 sigma rms), "
        f"rms variation {length_variation:.2%}, "
        f"drift over the run {length_drift:+.2%}"
    )
    print(
        f"Bunch position: mean {bunch_position_turns.mean() * 1e9:.4f} "
        f"ns, peak-to-peak excursion {position_span * 1e12:.2f} ps"
    )
    print(
        "NB for the parabolic-amplitude family the full bunch length "
        f"is sqrt(6)/2 = 1.2247x the 4 sigma length, i.e. "
        f"{1.2247 * length_mean * 1e9:.3f} ns here."
    )

    # Stationarity plot -----------------------------------------------
    fig2, (ax_length, ax_position) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, num="EX_29_stationarity"
    )
    turns = np.arange(len(bunch_length_turns))
    ax_length.plot(turns, bunch_length_turns * 1e9, color="C0", lw=0.8)
    ax_length.axhline(
        TARGET_BUNCH_LENGTH * 1e9,
        color="k",
        ls="--",
        lw=1.0,
        label="target (4 sigma rms)",
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
