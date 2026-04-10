# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Debunching into a barrier RF system example with BLonD3.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    BarrierGenerator,
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    StaticProfileObservation,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
)
from blond.core.base import ScheduledArray
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"


def main(
    run_n_turns: int | None = None, n_macroparticles: int | None = None
) -> None:
    momentum = 3.9051e9
    circumference = 2 * np.pi * 100
    transition_gamma = 6.1
    each_turn_i_profile = 100
    target_n_turns = 3000
    n_turns = target_n_turns if run_n_turns is None else run_n_turns
    n_macroparticles = (
        int(1e5) if n_macroparticles is None else n_macroparticles
    )

    main_amplitude = 10e3
    barrier_amplitude = -5e3
    barrier_width = 200e-9

    ring = Ring(circumference=circumference)

    energy_cycle = ConstantMagneticCycle(
        value=momentum,
        reference_particle=proton,
    )

    t_rev = energy_cycle.get_t_rev_init(ring.circumference, proton)

    main_rf = SingleHarmonicRFStation()
    main_rf.harmonic = 16
    main_rf.phi_rf_design = 0
    main_voltage = np.zeros(target_n_turns)
    main_voltage[:1000] = np.linspace(main_amplitude, 0, 1000)
    main_rf_schedule = ScheduledArray(main_voltage)

    main_rf.schedule(
        attribute="voltage",
        value=main_rf_schedule,
    )

    barrier_rf = BarrierGenerator(
        t_center=t_rev, t_width=barrier_width, n_bins=256
    )
    barrier_voltage = np.zeros(target_n_turns)
    barrier_voltage[500:] = barrier_amplitude
    barrier_schedule = ScheduledArray(barrier_voltage)

    barrier_rf.schedule(attribute="peak", value=barrier_schedule)

    drift = DriftSimple(
        orbit_length=circumference,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma
        ),
    )

    beam = Beam(
        intensity=0,
        particle_type=proton,
    )

    profile = StaticProfile(
        cut_left=0,
        cut_right=t_rev,
        n_bins=2**11,
    )

    ring.add_elements([profile, main_rf, barrier_rf, drift])

    sim = Simulation(ring, energy_cycle)

    n_turns = 3000

    time_limit = [0.5 * t_rev / 16, 15.5 * t_rev / 16]

    preparation_routine = SemiEmpiricMatcher(
        time_limit,
        n_macroparticles,
        hamilton_to_density_kwargs={
            "density_modifier": 1.0,
            "hamilton_max": 100,
        },
    )

    sim.prepare_beam(beam=beam, preparation_routine=preparation_routine)

    profile_observation = StaticProfileObservation(
        each_turn_i=each_turn_i_profile,
        profile=profile,
    )

    sim.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        observe=(profile_observation,),
    )

    bunch_time = copy_to_cpu(profile.hist_x) * 1e9
    cycle_time = np.arange(0, n_turns + 1, each_turn_i_profile) * t_rev * 1e3
    plt.figure("bunch evolution")
    plt.clf()
    plt.imshow(
        profile_observation.hist_y,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[bunch_time[0], bunch_time[-1], cycle_time[0], cycle_time[-1]],
    )
    plt.xlabel("Time [ns]")
    plt.ylabel("Cycle time [ms]")
    plt.tight_layout()
    os.makedirs("results/EX_11_BarrierBucket/", exist_ok=True)
    plt.savefig("results/EX_11_BarrierBucket/barrier_debunching.png")


if __name__ == "__main__":  # pragma: no cover
    main()
