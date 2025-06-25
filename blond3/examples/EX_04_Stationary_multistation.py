# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example input for simulating a ring with multiple RF stations
No intensity effects

:Authors: **Helga Timko**
"""

import matplotlib.pyplot as plt

from blond3 import (
    BiGaussian,
    Beam,
    proton,
    Ring,
    Simulation,
    ConstantProgram,
    ProfileObservation,
    SingleHarmonicCavity,
    DriftSimple,
    BoxLosses,
)
from blond3.examples.EX_02_Acceleration import energy_cycle
from blond3.physics.losses import SeparatrixLosses
from blond3.physics.profiles import DynamicProfileConstNBins

# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1.0e9  # Intensity
N_p = 1001

tau_0 = 0.4e-9  # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883  # Machine circumference [m]
p_s = 450.0e9  # Synchronous momentum [eV]
h = 35640  # Harmonic number
V1 = 2e6  # RF voltage, station 1 [eV]
V2 = 4e6  # RF voltage, station 1 [eV]
dphi = 0  # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

# Tracking details
N_t = 2000  # Number of turns to track
dt_plt = 200  # Time steps between plots


ring = Ring(circumference=C)
beam = Beam(n_particles=N_b, particle_type=proton)
one_turn_execution_order = (
    DriftSimple(transition_gamma=gamma_t, share_of_circumference=0.3, section_index=0),
    SingleHarmonicCavity(
        harmonic=h,
        rf_program=ConstantProgram(phase=dphi, effective_voltage=V1),
        section_index=0,
    ),
    DriftSimple(transition_gamma=gamma_t, share_of_circumference=0.7, section_index=1),
    SingleHarmonicCavity(
        harmonic=h,
        rf_program=ConstantProgram(phase=dphi, effective_voltage=V2),
        section_index=1,
    ),
    BoxLosses(t_min=0, t_max=2.5e-9),
    SeparatrixLosses(),
    DynamicProfileConstNBins(n_bins=100),
)
ring.add_elements(one_turn_execution_order, reorder=False)
sim = Simulation(ring=ring, beams=(beam,), energy_cycle=energy_cycle)
sim.on_prepare_beam(
    preparation_routine=BiGaussian(
        sigma_dt=tau_0 / 4, reinsertion=True, seed=1, n_macroparticles=N_p
    )
)
profile_observable = ProfileObservation(each_turn_i=10)
sim.run_simulation(observe=(profile_observable,))
#############################################
plt.plot(profile_observable.turns_array, profile_observable.hist_ys)
