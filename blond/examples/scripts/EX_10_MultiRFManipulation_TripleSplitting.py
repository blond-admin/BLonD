# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
PS triple splitting example with BLonD3.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    "/home/alasheen/cernbox/Work/FCCMatchingAndBLonD3Testing/BLonD"
)
from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    StaticProfile,
    StaticProfileObservation,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.core.base import ScheduledInterpolation
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

momentum = 3.9051e9
circumference = 2 * np.pi * 100
transition_gamma = 6.1
bunch_population = 2.6e11 * 12
n_macroparticles = 1e4
each_turn_i_profile = 100


ring = Ring(
    circumference=circumference,
)

energy_cycle = ConstantMagneticCycle(
    value=momentum,
    reference_particle=proton,
)

main_rf_station = MultiHarmonicRFStation(n_harmonics=3, main_harmonic_idx=0)
main_rf_station.harmonic = np.array([7, 14, 21])
main_rf_station.phi_rf_design = np.array([0, np.pi, 0])

voltage_h7_prog = np.loadtxt(
    f"{this_directory}/resources/EX_10_programs/voltage_h7.txt"
)
voltage_h14_prog = np.loadtxt(
    f"{this_directory}/resources/EX_10_programs/voltage_h14.txt"
)
voltage_h21_prog = np.loadtxt(
    f"{this_directory}/resources/EX_10_programs/voltage_h21.txt"
)

scheduler = ScheduledInterpolation(
    times=voltage_h7_prog[:, 0] * 1e-3,
    values=np.vstack(
        (
            voltage_h7_prog[:, 1] * 1e3,
            voltage_h14_prog[:, 1] * 1e3,
            voltage_h21_prog[:, 1] * 1e3,
        )
    ),
)

main_rf_station.schedule(
    attribute="voltage",
    value=scheduler,
)

drift = DriftSimple(
    orbit_length=circumference,
    momentum_compaction_factor=momentum_compaction_factor(transition_gamma),
)

beam = Beam(
    intensity=bunch_population,
    particle_type=proton,
)

t_rev = energy_cycle.get_t_rev_init(ring.circumference, proton)
t_rf = t_rev / main_rf_station.get_main_harmonic()

profile = StaticProfile(
    cut_left=-t_rf / 2,
    cut_right=t_rf / 2,
    n_bins=2**11,
)

sim = Simulation.from_locals(locals())
sim.print_one_turn_execution_order()

n_turns = int((voltage_h7_prog[-1, 0] * 1e-3) / t_rev)

time_limit = [-t_rf / 2, t_rf / 2]

preparation_routine = SemiEmpiricMatcher(
    time_limit,
    n_macroparticles,
    hamilton_to_density_kwargs={"density_modifier": 1.0, "hamilton_max": 400},
)

sim.prepare_beam(beam=beam, preparation_routine=preparation_routine)

# sim.plot_potential_well_empiric(np.linspace(time_limit[0], time_limit[-1], 10000), proton)

profile_observation = StaticProfileObservation(
    each_turn_i=each_turn_i_profile,
    profile=profile,
)


sim.run_simulation(
    beams=(beam,),
    n_turns=n_turns,
    observe=(profile_observation,),
    # callbacks=[custom_analysis],
)

bunch_time = profile.hist_x * 1e9
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
os.makedirs(
    "results/EX_10_MultiRFManipulation_TripleSplitting/", exist_ok=True
)
plt.savefig(
    "results/EX_10_MultiRFManipulation_TripleSplitting/triple_splitting.png"
)
