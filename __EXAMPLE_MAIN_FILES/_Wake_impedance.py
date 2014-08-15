
from __future__ import division
import numpy as np
import math
from scipy.constants import c, e, m_p
import time, sys
import matplotlib.pyplot as plt

from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.longitudinal_tracker import *
from beams.beams import *
from beams.longitudinal_distributions import *
from monitors.monitors import *
from beams.slices import *
from impedances.longitudinal_impedance import *
from longitudinal_plots.plot_beams import *
from longitudinal_plots.plot_impedance import *
from longitudinal_plots.plot_slices import *


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = int(1e11)          
n_macroparticles = int(5e5)
sigma_tau = 180e-9 / 4 # [s]     
sigma_delta = .5e-4 # [1]          
kin_beam_energy = 1.4e9 # [eV]

# Machine and RF parameters
radius = 25
gamma_transition = 4.4  # [1]
C = 2 * np.pi * radius  # [m]       
      
# Tracking details
n_turns = 2          
n_turns_between_two_plots = 1          

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy =  E_0 + kin_beam_energy # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2) # [eV / c]
gamma = tot_beam_energy / E_0  # [1]        
beta = np.sqrt(1 - 1 / gamma**2)  # [1]
sigma_theta = beta * c / radius * sigma_tau # [rad]     
sigma_dE = beta**2 * tot_beam_energy * sigma_delta # [eV]
momentum_compaction = 1 / gamma_transition**2 # [1]       

# Cavities parameters
n_rf_systems = 1                                     
harmonic_numbers = 1                         
voltage_program = 8.e3
phi_offset = 0

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor('beam', n_turns+1, statistics = "Longitudinal")

# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction, sync_momentum, 
                                   particle_type, number_of_sections = 1)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems, harmonic_numbers, 
                          voltage_program, phi_offset)

ring_RF_section = RingAndRFSection(RF_sct_par)

# DEFINE BEAM------------------------------------------------------------------

my_beam = Beam(general_params, n_macroparticles, n_particles)

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, sigma_theta, sigma_dE)


number_slices = 100
slice_beam = Slices(my_beam, number_slices, cut_left = - 5.72984173562e-07 / 2, 
                    cut_right = 5.72984173562e-07 / 2, mode = 'const_space_hist')

temp = np.loadtxt('new_HQ_table.dat', comments = '!')
R_shunt = temp[:, 2] * 10**6 
f_res = temp[:, 0] * 10**9
Q_factor = temp[:, 1]

resonator = Resonators(R_shunt, f_res, Q_factor)
ind_volt_time = InducedVoltageTime(slice_beam, [resonator])
ind_volt_freq = InducedVoltageFreq(slice_beam, [resonator], 2e5)
tot_vol = TotalInducedVoltage(slice_beam, [ind_volt_time])


# ACCELERATION MAP-------------------------------------------------------------

map_ = [slice_beam] + [tot_vol] + [ring_RF_section] + [bunchmonitor]


# TRACKING + PLOTS-------------------------------------------------------------

for i in range(n_turns):
    
    print i+1
    for m in map_:
        m.track(my_beam)
    

    # Plots
    if ((i+1) % n_turns_between_two_plots) == 0:
        
        plot_induced_voltage_vs_bins_centers(i+1, general_params, tot_vol, style = '-')
        
        
print "Done!"

bunchmonitor.h5file.close()
