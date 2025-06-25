# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
SPS simulation with intensity effects in time and frequency domains using
a table of resonators. The input beam has been cloned to show that the two
methods are equivalent (compare the two figure folders). Note that to create an
exact clone of the beam, the option seed=0 in the generation has been used.
This script shows also an example of how to use the class SliceMonitor (check
the corresponding h5 files).

:Authors: **Danilo Quartullo**
"""

import os

import matplotlib as mpl
import numpy as np

from blond.impedances.induced_voltage_analytical import analytical_gaussian_resonator
from blond3 import (
    BiGaussian,
    Beam,
    proton,
    Ring,
    Simulation,
    EnergyCycle,
    ConstantProgram,
    SingleHarmonicCavity,
    DriftSimple,
    BunchObservation,
    WakeField,
    StaticProfile,
)
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.sovlers import (
    SingleTurnWakeSolverTimeDomain,
    PeriodicFreqSolver,
    AnalyticSingleTurnResonatorSolver,
)

DRAFT_MODE = bool(int(os.environ.get("BLOND_EXAMPLES_DRAFT_MODE", False)))
# To check if executing correctly, rather than to run the full simulation

mpl.use("Agg")

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

os.makedirs(this_directory + "../output_files/EX_05_fig", exist_ok=True)


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
n_particles = 1e10
n_macroparticles = 1001 if DRAFT_MODE else 5 * 1e6
tau_0 = 2e-9  # [s]

# Machine and RF parameters
gamma_transition = 1 / np.sqrt(0.00192)  # [1]
C = 6911.56  # [m]

# Tracking details
n_turns = 2
dt_plt = 1

# Derived parameters
sync_momentum = 25.92e9  # [eV / c]
momentum_compaction = 1 / gamma_transition**2  # [1]

# Cavities parameters
n_rf_systems = 1
harmonic_number = 4620
voltage_program = 0.9e6  # [V]
phi_offset = 0.0


table = np.loadtxt(
    this_directory + "../input_files/EX_05_new_HQ_table.dat", comments="!"
)

R_shunt = table[:, 2] * 10**6
f_res = table[:, 0] * 10**9
Q_factor = table[:, 1]

for wake_solver in (
    SingleTurnWakeSolverTimeDomain,
    PeriodicFreqSolver,
    AnalyticSingleTurnResonatorSolver,
):
    ring = Ring(circumference=C)
    ring.set_energy_cycle(EnergyCycle.from_linspace(sync_momentum, sync_momentum, 2))
    cavity1 = SingleHarmonicCavity(
        harmonic=harmonic_number,
        rf_program=ConstantProgram(effective_voltage=voltage_program, phase=phi_offset),
    )
    beam = Beam(
        n_particles=n_particles, particle_type=proton
    )
    drift = DriftSimple(
        transition_gamma=gamma_transition,
        share_of_circumference=1.0,
    )
    wakefield = WakeField(
        sources=(Resonators(R_shunt, f_res, Q_factor),), solver=wake_solver
    )
    profile = StaticProfile.from_rad(0, 2 * np.pi, 2**8, ring.get_t_rev(turn_i=0))
    ring.add_elements((drift, cavity1, wakefield), reorder=True)
    sim = Simulation(ring=ring)
    sim.prepare_beam(BiGaussian(tau_0 / 4, seed=1, n_macroparticles=n_macroparticles))
    bunch_observable = BunchObservation(each_turn_i=10)
    sim.run_simulation(observe=(bunch_observable,))


# Analytic result-----------------------------------------------------------
VindGauss = np.zeros(len(profile.bin_centers))
for r in range(len(Q_factor)):
    # Notice that the time-argument of inducedVoltageGauss is shifted by
    # mean(my_slices.bin_centers), because the analytical equation assumes the
    # Gauss to be centered at t=0, but the line density is centered at
    # mean(my_slices.bin_centers)
    tmp = analytical_gaussian_resonator(
        tau_0 / 4,
        Q_factor[r],
        R_shunt[r],
        2 * np.pi * f_res[r],
        profile.bin_centers - np.mean(profile.bin_centers),
        beam.intensity,
    )
    VindGauss += tmp.real
