
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
SPS simulation with intensity effects in time and frequency domains using
a table of resonators. The input beam has been cloned to show that the two methods
are equivalent (compare the two figure folders). Note that to create an exact 
clone of the beam, the option seed=0 in the generation has been used. This 
script shows also an example of how to use the class SliceMonitor (check the
corresponding h5 files).
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
from plots.plot import *


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = 1e10        
n_macroparticles = 5000000
tau_0 = 2e-9 # [s]

# Machine and RF parameters
gamma_transition = 1/np.sqrt(0.00192)   # [1]
C = 6911.56  # [m]
      
# Tracking details
n_turns = 2          
dt_plt = 1          

# Derived parameters
sync_momentum = 25.92e9 # [eV / c]
momentum_compaction = 1 / gamma_transition**2 # [1]       

# Cavities parameters
n_rf_systems = 1                                     
harmonic_number = 4620                         
voltage_program = 0.9e6 # [V]
phi_offset = 0



# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction, sync_momentum, 
                                   particle_type, number_of_sections = 1)
general_params_copy = GeneralParameters(n_turns, C, momentum_compaction, sync_momentum, 
                                   particle_type, number_of_sections = 1)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems, harmonic_number, 
                          voltage_program, phi_offset)
RF_sct_par_copy = RFSectionParameters(general_params_copy, n_rf_systems, harmonic_number, 
                          voltage_program, phi_offset)

my_beam = Beam(general_params, n_macroparticles, n_particles)
my_beam_copy = Beam(general_params_copy, n_macroparticles, n_particles)

ring_RF_section = RingAndRFSection(RF_sct_par, my_beam)
ring_RF_section_copy = RingAndRFSection(RF_sct_par_copy, my_beam_copy)

# DEFINE BEAM------------------------------------------------------------------

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, tau_0/4, 
                             seed=1)

longitudinal_bigaussian(general_params_copy, RF_sct_par_copy, my_beam_copy, tau_0/4, 
                              seed=1)

number_slices = 2**8
slice_beam = Slices(RF_sct_par, my_beam, number_slices, cut_left = 0, 
                    cut_right = 2 * np.pi, 
                    cuts_unit = 'rad', 
                    fit_option = 'gaussian')
slice_beam.track()
slice_beam_copy = Slices(RF_sct_par_copy, my_beam_copy, number_slices, cut_left = 0, 
                    cut_right = 2 * np.pi , 
                    cuts_unit = 'rad', 
                    fit_option = 'gaussian')
slice_beam_copy.track()

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor(general_params, my_beam, '../output_files/TC5_output_data', Slices=slice_beam, buffer_time = 1)

bunchmonitor_copy = BunchMonitor(general_params_copy, my_beam_copy, '../output_files/TC5_output_data_copy', Slices=slice_beam_copy, buffer_time = 1)


# LOAD IMPEDANCE TABLE--------------------------------------------------------

table = np.loadtxt('../input_files_for_test_cases/TC5_new_HQ_table.dat', comments = '!')

R_shunt = table[:, 2] * 10**6 
f_res = table[:, 0] * 10**9
Q_factor = table[:, 1]
resonator = Resonators(R_shunt, f_res, Q_factor)

ind_volt_time = InducedVoltageTime(slice_beam, [resonator])
ind_volt_freq = InducedVoltageFreq(slice_beam_copy, [resonator], 1e5)

tot_vol = TotalInducedVoltage(my_beam, slice_beam, [ind_volt_time])
tot_vol_copy = TotalInducedVoltage(my_beam_copy, slice_beam_copy, [ind_volt_freq])

# PLOTS

plots = Plot(general_params, RF_sct_par, my_beam, dt_plt, 0, 
             0.0014*harmonic_number, - 1.5e8, 1.5e8, xunit= 'rad',
             separatrix_plot= True, Slices = slice_beam, h5file = '../output_files/TC5_output_data', histograms_plot = True, sampling=50)
 
 
plots.set_format(dirname = '../output_files/TC5_fig/1', linestyle = '.')
plots.track()

plots_copy = Plot(general_params_copy, RF_sct_par_copy, my_beam_copy, dt_plt, 0, 
             0.0014*harmonic_number, - 1.5e8, 1.5e8, xunit= 'rad',
             separatrix_plot= True, Slices = slice_beam_copy, h5file = '../output_files/TC5_output_data_copy', histograms_plot = True, sampling=50)

plots_copy.set_format(dirname = '../output_files/TC5_fig/2', linestyle = '.')
plots_copy.track() 

# ACCELERATION MAP-------------------------------------------------------------

map_ = [tot_vol] + [ring_RF_section] + [slice_beam] + [bunchmonitor] + [plots]
map_copy = [tot_vol_copy] + [ring_RF_section_copy] + [slice_beam_copy] + [bunchmonitor_copy] + [plots_copy]

# TRACKING + PLOTS-------------------------------------------------------------

for i in np.arange(1, n_turns+1):
    
    print i
    for m in map_:
        m.track()
    for m in map_copy:
        m.track()
    
    # Plots
    if (i % dt_plt) == 0:
        
        plot_induced_voltage_vs_bin_centers(i, general_params, tot_vol, style = '.', dirname = '../output_files/TC5_fig/1')
        plot_induced_voltage_vs_bin_centers(i, general_params_copy, tot_vol_copy, style = '.', dirname = '../output_files/TC5_fig/2')
        
 
        
print "Done!"


