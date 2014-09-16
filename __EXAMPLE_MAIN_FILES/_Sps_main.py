'''
SPS test case
'''

# General imports
from __future__ import division
import numpy as np
import time
import matplotlib.pyplot as plt
import os


# PyHEADTAIL imports
from input_parameters.general_parameters import GeneralParameters 
from input_parameters.rf_parameters import RFSectionParameters
from beams.beams import Beam
from beams.longitudinal_distributions import matched_from_distribution_density, \
                                             matched_from_line_density   
from trackers.longitudinal_tracker import RingAndRFSection, FullRingAndRF
from beams.slices import Slices
from impedances.longitudinal_impedance import TravelingWaveCavity, Resonators, \
                                              InductiveImpedance, InputTable, \
                                              InducedVoltageTime, InducedVoltageFreq, \
                                              TotalInducedVoltage

# Other imports
from scipy.constants import c


# Simulation parameters -------------------------------------------------------
# Simulation parameters
n_turns = 1500          # Number of turns to track
plot_step = 1           # Time steps between plots
output_step = 100

# General parameters
particle_type = 'proton'
circumference = 6911.56                         # Machine circumference [m]
gamma_transition = 1/np.sqrt(0.00192)           # Transition gamma Q26
# gamma_transition = 18.                        # Transition gamma Q20
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array
sync_momentum = 25.92e9                         # Synchronous momentum program [eV/c] Flat bottom
# sync_momentum = 450e9                         # Synchronous momentum program [eV/c] Flot top

# RF parameters
n_rf_systems = 2                # Number of rf systems
harmonic_numbers_1 = 4620       # Harmonic number first harmonic
# voltage_program_1 = 7.e6      # RF voltage [V] first harmonic
voltage_program_1 = 0.9e6       # RF voltage [V] first harmonic
phi_offset_1 = 0                # Phase offset first harmonic
harmonic_numbers_2 = 4620*4     # Harmonic number second harmonic
# voltage_program_2 = 0.65e6    # RF voltage [V] second harmonic
voltage_program_2 = 0.e6        # RF voltage [V] second harmonic
phi_offset_2 = np.pi            # Phase offset second harmonic

# Beam parameters
# intensity = 2.6e11        # Intensity
intensity = 8.e10           # Intensity
n_macroparticles = 5e5      # Macro-particles

# Slicing parameters
n_slices = 2**8
cut_left = 0.
cut_right = 2*np.pi / harmonic_numbers_1
frequency_step = 94.e3 # [Hz]

# Impedance parameters
resonators = np.loadtxt('Z_table.dat', comments='!')

Space_charge_Z_over_n = -1.5 # Ohms
Steps_Z_over_n = 0.5 # Ohms

    
# Simulation setup ------------------------------------------------------------
# General parameters                  
general_params = GeneralParameters(n_turns, circumference, momentum_compaction, 
                                   sync_momentum, particle_type)

# RF parameters
## Only one rf system
# rf_params = RFSectionParameters(general_params, 1, harmonic_numbers_1, 
#                                       voltage_program_1, phi_offset_1)
## Two rf systems
rf_params = RFSectionParameters(general_params, n_rf_systems, [harmonic_numbers_1, harmonic_numbers_2], 
                                [voltage_program_1, voltage_program_2], [phi_offset_1, phi_offset_2])

# RF tracker
longitudinal_tracker = RingAndRFSection(rf_params)
full_tracker = FullRingAndRF([longitudinal_tracker])

# Beam
SPS_beam = Beam(general_params, n_macroparticles, intensity)
 
# Slicing
slicing = Slices(SPS_beam, n_slices, cut_left = cut_left, cut_right = cut_right, 
                 cuts_coord = 'theta', slicing_coord = 'tau', mode = 'const_space_hist', fit_option = 'gaussian')
  
   
# Impedance sources
SPS_RES = Resonators(resonators[:,2]*1e6, resonators[:,0]*1e9, resonators[:,1])

Wake_list = [SPS_RES]
SPS_intensity_time = InducedVoltageTime(slicing, Wake_list)
    
SPS_inductive = InductiveImpedance(slicing, Space_charge_Z_over_n + Steps_Z_over_n, general_params.f_rev[0], deriv_mode = 'gradient')

SPS_longitudinal_intensity = TotalInducedVoltage(slicing, [SPS_intensity_time]) # , SPS_inductive

# Beam generation
# emittance = 0.4
# matched_from_distribution_density(SPS_beam, full_tracker, {'type':'parabolic_amplitude', 'emittance':emittance, 'density_variable':'density_from_H'}, main_harmonic_option = 'lowest_freq', TotalInducedVoltage = SPS_longitudinal_intensity, n_iterations_input = 50)
bunch_length = 2.e-9 / (SPS_beam.ring_radius / (SPS_beam.beta_r * c))
matched_from_line_density(SPS_beam, full_tracker, {'type':'parabolic_amplitude', 'bunch_length':bunch_length}, main_harmonic_option = 'lowest_freq', TotalInducedVoltage = SPS_longitudinal_intensity)
         
# Total simulation map
sim_map = [full_tracker] + [slicing] + [SPS_longitudinal_intensity]
         
# Tracking ---------------------------------------------------------------------

plt.ion()
plt.figure(3)
save_bl = np.zeros(n_turns)
save_bp = np.zeros(n_turns)
for i in range(n_turns):
                        
    if i % output_step == 0:
        print i
        t0 = time.clock()
                            
    # Track
    for m in sim_map:
        m.track(SPS_beam)
#         SPS_beam.longit_statistics()
                           
    if i % output_step == 0:
        t1 = time.clock()
#         print 4*SPS_beam.sigma_tau, SPS_beam.mean_tau
        print t1-t0
                                            
    if i % plot_step == 0:
#         plt.plot(SPS_beam.theta[0:10000], SPS_beam.dE[0:10000],'.')
#         plt.xlim((0,2*np.pi/harmonic_numbers_1))
#         plt.ylim((-1.50e8,1.50e8))
#         plt.pause(0.0001)
#         plt.clf()
#         plt.show()
        plt.plot(SPS_longitudinal_intensity.time_array, SPS_longitudinal_intensity.induced_voltage)
        plt.plot(SPS_longitudinal_intensity.time_array, slicing.n_macroparticles / np.max(slicing.n_macroparticles) *0.25e6)#* np.max(SPS_longitudinal_intensity.induced_voltage))
        plt.ylim((-0.5e6, 0.5e6))
#         plt.plot(slicing.bins_centers, slicing.n_macroparticles)
#         plt.ylim((0,6e3))
#         plt.plot(slicing.bins_centers, gauss(slicing.bins_centers, *slicing.pfit_gauss))
#         plt.plot(slicing.edges[:-1]-slicing.edges[1:])
        plt.pause(0.0001)
        plt.clf()
#         plt.show()
#         plot_long_phase_space(SPS_beam, general_params_dummy, rf_params_dummy, 
#                               0., 5., -150, 150, xunit='ns', separatrix_plot = 'True', sampling = 1e1)
       
    save_bl[i] = SPS_beam.bl_gauss_tau
    save_bp[i] = SPS_beam.bp_gauss_tau
#     save_bl[i] = 4*np.std(SPS_beam.tau)
#     save_bp[i] = np.mean(SPS_beam.tau)
       
    
plt.ioff()
     
plt.figure(1)
plt.plot(save_bl)
plt.figure(2)
plt.plot(save_bp)
plt.show()
