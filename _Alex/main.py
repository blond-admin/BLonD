'''




SPS test case





'''

from __future__ import division
import numpy as np

from input_parameters.simulation_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters, SumRFSectionParameters
from trackers.longitudinal_tracker import FullRingAndRF, RingAndRFSection
from beams.beams import Beam
from beams.longitudinal_distributions import longitudinal_gaussian_matched
from longitudinal_plots.longitudinal_plots import plot_long_phase_space

import time

# Simulation parameters --------------------------------------------------------
# Simulation parameters
n_turns = 2000          # Number of turns to track
plot_step = 100          # Time steps between plots
# output_step = 100   # Time steps between outputs

# General parameters
particle_type = 'proton'
circumference = 6911.56                         # Machine circumference [m]
gamma_transition = 1/np.sqrt(0.00192)           # Transition gamma
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array

# RF parameters
n_rf_systems_2 = 1                                      # Number of rf systems second section
harmonic_numbers_2 = 4620                               # Harmonic number second section
voltage_program_2 = 0.9e6                               # RF voltage [V] second section
sync_momentum_2 = np.linspace(25.92e9, 27e9, n_turns+1) # Synchronous momentum program [eV/c] second section
# sync_momentum_2 = 25.92e9 * np.ones(n_turns+1)          # Synchronous momentum program [eV/c] second section
phi_offset_2 = 0

# Beam parameters
intensity = 1.e10           # Intensity
n_macroparticles = 100000   # Macro-particles
tau_0 = 0.5                  # Initial bunch length, 4 sigma [ns]


# Simulation setup -------------------------------------------------------------
# Gathering and pre-processing parameters
section_params = RFSectionParameters(n_turns, n_rf_systems_2, circumference, harmonic_numbers_2, voltage_program_2, phi_offset_2, sync_momentum_2)
general_params = GeneralParameters(particle_type, n_turns, circumference, momentum_compaction, section_params.momentum_program)
 
# # RF tracker
my_accelerator_section = RingAndRFSection(general_params, section_params)
 
# # Bunch generation
my_beam = Beam(general_params, n_macroparticles, intensity)
longitudinal_gaussian_matched(general_params, my_accelerator_section, my_beam, tau_0, unit='ns')
 
# Map
map_ = [my_accelerator_section]
 
 
# Tracking ---------------------------------------------------------------------
 
for i in range(n_turns):
      
    if i % 100 == 0:
        print i
        t0 = time.clock()
       
    # Track
    for m in map_:
        m.track(my_beam)
    general_params.counter[0] += 1
      
    if i % 100 == 0:
        t1 = time.clock()
        print t1-t0
      
    if i % plot_step == 0:
        plot_long_phase_space(my_beam, general_params, my_accelerator_section, 0., 5., -150, 150, xunit='ns')

             

                   




