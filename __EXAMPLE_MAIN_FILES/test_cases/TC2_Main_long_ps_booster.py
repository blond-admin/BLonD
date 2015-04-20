
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects from impedance tables
'''

from __future__ import division
import numpy as np
import math
from scipy.constants import c, e, m_p
import time, sys
import matplotlib.pyplot as plt

from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from beams.beams import *
from beams.distributions import *
from monitors.monitors import *
from beams.slices import *
from impedances.impedance import *
from plots.plot_beams import *
from plots.plot_impedance import *
from plots.plot_slices import *

# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = 1e11
n_macroparticles = 5e5
sigma_tau = 180 / 4 # [ns]     
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

momentum_compaction = 1 / gamma_transition**2 # [1]       

# Cavities parameters
n_rf_systems = 1                                     
harmonic_numbers = 1                         
voltage_program = 8.e3
phi_offset = 0


# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction, sync_momentum, 
                                   particle_type, number_of_sections = 1)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems, harmonic_numbers, 
                          voltage_program, phi_offset)

ring_RF_section = RingAndRFSection(RF_sct_par)

# DEFINE BEAM------------------------------------------------------------------

my_beam = Beam(general_params, n_macroparticles, n_particles)

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, sigma_tau, seed=1, xunit='ns')


# DEFINE SLICES----------------------------------------------------------------

number_slices = 100
slice_beam = Slices(my_beam, number_slices, cut_left = - 5.72984173562e-07, 
                    cut_right = 5.72984173562e-07)
slice_beam.track(my_beam)

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor('../output_files/TC2_output_data', n_turns+1)
bunchmonitor.track(my_beam)

# LOAD IMPEDANCE TABLES--------------------------------------------------------

var = str(kin_beam_energy / 1e9)

# ejection kicker
Ekicker = np.loadtxt('../input_files/TC2_Ekicker_1.4GeV.txt'
        , skiprows = 1, dtype=complex, converters = dict(zip((0, 1), (lambda s: 
        complex(s.replace('i', 'j')), lambda s: complex(s.replace('i', 'j'))))))

Ekicker_table = InputTable(Ekicker[:,0].real, Ekicker[:,1].real, Ekicker[:,1].imag)


# Finemet cavity
F_C = np.loadtxt('../input_files/TC2_Finemet.txt', dtype = float, skiprows = 1)

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

steps = InductiveImpedance(slice_beam, [34.6669349520904 / 10e9 * general_params.f_rev], general_params.f_rev, RF_sct_par.counter, deriv_mode='diff') 

# direct space charge

dir_space_charge = InductiveImpedance(slice_beam, [-376.730313462   
                     / (general_params.beta_r[0] *
                     general_params.gamma_r[0]**2)], general_params.f_rev, RF_sct_par.counter)

# INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

imp_list = [Ekicker_table, F_C_table]

ind_volt_freq = InducedVoltageFreq(slice_beam, imp_list, 2e5)

total_induced_voltage = TotalInducedVoltage(slice_beam, [ind_volt_freq, steps, dir_space_charge])


# ACCELERATION MAP-------------------------------------------------------------

map_ = [total_induced_voltage] + [ring_RF_section] + [slice_beam] + [bunchmonitor]

plot_long_phase_space(my_beam, general_params, RF_sct_par, 
          - 5.72984173562e2, 5.72984173562e2, 
          - my_beam.sigma_dE * 4 * 1e-6, my_beam.sigma_dE * 4 * 1e-6, xunit = 'ns', dirname = '../output_files/TC2_fig')
         
plot_impedance_vs_frequency(0, general_params, ind_volt_freq, 
  option1 = "single", style = '.', option3 = "freq_table", option2 = "spectrum", dirname = '../output_files/TC2_fig')
 

plot_beam_profile(0, general_params, slice_beam, dirname = '../output_files/TC2_fig', style = '.')
 
plot_bunch_length_evol('../output_files/TC2_output_data', general_params, 0, dirname = '../output_files/TC2_fig')

plot_position_evol('../output_files/TC2_output_data', general_params, 0, unit = None, dirname = '../output_files/TC2_fig')

# TRACKING + PLOTS-------------------------------------------------------------

for i in range(1, n_turns+1):
    
    print i
    t0 = time.clock()
    for m in map_:
        m.track(my_beam)
    t1 = time.clock()
    print t1 - t0
    # Plots
    if (i% n_turns_between_two_plots) == 0:
        
        plot_long_phase_space(my_beam, general_params, RF_sct_par, 
          - 5.72984173562e2, 5.72984173562e2, 
          - my_beam.sigma_dE * 4 * 1e-6, my_beam.sigma_dE * 4 * 1e-6, xunit = 'ns', dirname = '../output_files/TC2_fig', separatrix_plot = True)
         
        plot_impedance_vs_frequency(i, general_params, ind_volt_freq, 
          option1 = "single", style = '.', option3 = "freq_table", option2 = "spectrum", dirname = '../output_files/TC2_fig')
         
        plot_induced_voltage_vs_bins_centers(i, general_params, total_induced_voltage, style = '.', dirname = '../output_files/TC2_fig')
         
        plot_beam_profile(i, general_params, slice_beam, dirname = '../output_files/TC2_fig', style = '.')
         
        plot_bunch_length_evol('../output_files/TC2_output_data', general_params, i, dirname = '../output_files/TC2_fig')
        
        plot_position_evol('../output_files/TC2_output_data', general_params, i, unit = None, dirname = '../output_files/TC2_fig')

print "Done!"

bunchmonitor.h5file.close()

