'''



Several sections test case (SPS based, two RFs for a total voltage that
is the same than in main.py, no acceleration)




'''

from __future__ import division
import numpy as np

from input_parameters.general_parameters import General_parameters
from input_parameters.rf_parameters import RFSectionParameters, SumRFSectionParameters
from trackers.longitudinal_tracker import FullRingAndRF, RingAndRFSection
from beams.beams import Beam
from beams.longitudinal_distributions import longitudinal_gaussian_matched
from longitudinal_plots.longitudinal_plots import plot_long_phase_space

import time

# Simulation parameters --------------------------------------------------------
# Simulation parameters
n_turns = 2000          # Number of turns to track
plot_step = 200          # Time steps between plots
# output_step = 100   # Time steps between outputs

# General parameters
particle_type = 'proton'
circumference = 6911.56                         # Machine circumference [m]
gamma_transition = 1/np.sqrt(0.00192)           # Transition gamma
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array

# RF parameters
n_rf_systems_1 = 2                                  # Number of rf systems first section
harmonic_numbers_1_1 = np.array([4620])             # Harmonic number first section, first RF system
harmonic_numbers_1_2 = np.array([4620*4])           # Harmonic number first section, second RF system
harmonic_numbers_1_list = [harmonic_numbers_1_1, harmonic_numbers_1_2]
voltage_program_1_1 = 0.45e6 * np.ones([n_turns])        # RF voltage [V] first section, first RF system
voltage_program_1_2 = np.array([0.e6])                  # RF voltage [V] first section, second RF system
voltage_program_1_list = [voltage_program_1_1, voltage_program_1_2]
phi_offset_1_1 = 0 * np.ones([n_turns])                 # Phase offset first section, first RF system
phi_offset_1_2 = 0 * np.ones([n_turns])                 # Phase offset first section, second RF system
phi_offset_1_list = [phi_offset_1_1, phi_offset_1_2]
sync_momentum_1 = 25.92e9                               # Synchronous momentum [eV/c] first section
    
n_rf_systems_2 = 1                                      # Number of rf systems second section
harmonic_numbers_2 = 4620                               # Harmonic number second section
voltage_program_2 = 0.45e6                               # RF voltage [V] second section
sync_momentum_2 = 25.92e9                                # Synchronous momentum program [eV/c] second section
# sync_momentum_2 = 25.92e9 * np.ones(n_turns+1)          # Synchronous momentum program [eV/c] second section
phi_offset_2 = 0

# Beam parameters
intensity = 1.e10           # Intensity
n_macroparticles = 10000   # Macro-particles
tau_0 = 0.5                  # Initial bunch length, 4 sigma [ns]


# Simulation setup -------------------------------------------------------------
#Gathering and pre-processing parameters
general_params = General_parameters(n_turns, [circumference/2, circumference/2], 
                                    [[momentum_compaction], [momentum_compaction]], 
                                    [sync_momentum_1*np.ones(n_turns+1), sync_momentum_2*np.ones(n_turns+1)], number_of_sections = 2,
                                    particle_type)


section_1_params = RFSectionParameters(n_turns, n_rf_systems_1, circumference/2, harmonic_numbers_1_list, voltage_program_1_list, phi_offset_1_list, sync_momentum_1)
section_2_params = RFSectionParameters(n_turns, n_rf_systems_2, circumference/2, harmonic_numbers_2, voltage_program_2, phi_offset_2, sync_momentum_2)
full_rf_params = SumRFSectionParameters([section_1_params, section_2_params])
#general_params = General_parameters(particle_type, n_turns, circumference, momentum_compaction, full_rf_params.momentum_program_matrix)

 
# # RF tracker
my_accelerator_rf= FullRingAndRF(general_params, full_rf_params)
 
# # Bunch generation
my_beam = Beam(general_params, n_macroparticles, intensity)
# We need a bunch generator for the case of several RF sections/systems
# As well as a better way to calculate phi_s (for debunching, for several sections and RF...)
fake_full_voltage = RFSectionParameters(n_turns, 1, circumference, 4620, 2*0.45e6, 0, 25.92e9)
fake_full_voltage.index_section = 0
RingAndRFSection_fake = RingAndRFSection(general_params, fake_full_voltage)
longitudinal_gaussian_matched(general_params, RingAndRFSection_fake, my_beam, tau_0, unit='ns')

# Map
map_ = [my_accelerator_rf]

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
        plot_long_phase_space(my_beam, general_params, RingAndRFSection_fake, 0., 5., -150, 150, xunit='ns', separatrix_plot = True)                   




