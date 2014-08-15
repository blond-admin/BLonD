# Example script to take into account intensity effects from impedance tables

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

general_params = GeneralParameters(n_turns, C, [momentum_compaction, 1], sync_momentum, 
                                   particle_type, number_of_sections = 1)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems, harmonic_numbers, 
                          voltage_program, phi_offset)

ring_RF_section = RingAndRFSection(RF_sct_par)

# DEFINE BEAM------------------------------------------------------------------

my_beam = Beam(general_params, n_macroparticles, n_particles)

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, sigma_theta, sigma_dE)


# DEFINE SLICES----------------------------------------------------------------

number_slices = 100
slice_beam = Slices(my_beam, number_slices, cut_left = - 5.72984173562e-07 / 2, 
                    cut_right = 5.72984173562e-07 / 2, mode = 'const_space_hist')


# LOAD IMPEDANCE TABLES--------------------------------------------------------

var = str(kin_beam_energy / 1e9)

# ejection kicker
Ekicker = np.loadtxt('Ekicker_' + var + 'GeV.txt'
        , skiprows = 1, dtype=complex, converters = dict(zip((0, 1), (lambda s: 
        complex(s.replace('i', 'j')), lambda s: complex(s.replace('i', 'j'))))))

Ekicker_table = InputTable(Ekicker[:,0].real, Ekicker[:,1].real, Ekicker[:,1].imag)


# Finemet cavity
F_C = np.loadtxt('Finemet.txt', dtype = float, skiprows = 1)

F_C[:, 3], F_C[:, 5], F_C[:, 7] = np.pi * F_C[:, 3] / 180, np.pi * F_C[:, 5] / 180, np.pi * F_C[:, 7] / 180

option = "closed loop"

if option == "open loop":
    Re_Z = F_C[:, 4] * np.cos(F_C[:, 3])
    Im_Z = F_C[:, 4] * np.sin(F_C[:, 3])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
elif option == "closed loop":
    Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
    Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
elif option == "shorted":
    Re_Z = F_C[:, 6] * np.cos(F_C[:, 7])
    Im_Z = F_C[:, 6] * np.sin(F_C[:, 7])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
else:
    pass

# steps
steps = InductiveImpedance(slice_beam, 34.6669349520904 / 10e9 * general_params.f_rev[0], 2e5) 

# direct space charge
dir_space_charge = InductiveImpedance(slice_beam, -376.730313462   
                     / (general_params.beta_r[0,0] *
                     general_params.gamma_r[0,0]**2), 2e5)


# INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

imp_list = [Ekicker_table, F_C_table, steps, dir_space_charge]

ind_volt_freq = InducedVoltageFreq(slice_beam, imp_list, 2e5)

total_induced_voltage = TotalInducedVoltage(slice_beam, [ind_volt_freq])


# ACCELERATION MAP-------------------------------------------------------------

map_ = [slice_beam] + [total_induced_voltage] + [ring_RF_section] + [bunchmonitor]


# TRACKING + PLOTS-------------------------------------------------------------

for i in range(n_turns):
    
    print i+1
    t0 = time.clock()
    for m in map_:
        m.track(my_beam)
    t1 = time.clock()
    print t1 - t0
    # Plots
    if ((i+1) % n_turns_between_two_plots) == 0:
        
        plot_long_phase_space(my_beam, general_params, RF_sct_par, 
          - 5.72984173562e-07 / 2 * 1e9, 5.72984173562e-07 / 2 * 1e9, 
          - my_beam.sigma_dE * 4 * 1e-6, my_beam.sigma_dE * 4 * 1e-6, xunit = 'ns')
         
        plot_impedance_vs_frequency(i+1, general_params, ind_volt_freq, 
          option1 = "single", style = '-', option3 = "freq_table", option2 = "spectrum")
         
        plot_induced_voltage_vs_bins_centers(i+1, general_params, total_induced_voltage, style = '-')
         
        plot_beam_profile(i+1, general_params, slice_beam)
         
        plot_bunch_length_evol(my_beam, 'beam', general_params, n_turns)
        
        plot_position_evol(i+1, my_beam, 'beam', general_params, unit = None, dirname = 'fig')


print "Done!"

bunchmonitor.h5file.close()

