
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


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = 1e10        
n_macroparticles = 5000000
tau_0 = 2.0 # [ns]

# Machine and RF parameters
gamma_transition = 1/np.sqrt(0.00192)   # [1]
C = 6911.56  # [m]
      
# Tracking details
n_turns = 2          
n_turns_between_two_plots = 1          

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

ring_RF_section = RingAndRFSection(RF_sct_par)
ring_RF_section_copy = RingAndRFSection(RF_sct_par_copy)

# DEFINE BEAM------------------------------------------------------------------

my_beam = Beam(general_params, n_macroparticles, n_particles)

my_beam_copy = Beam(general_params_copy, n_macroparticles, n_particles)

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, tau_0/4, 
                              xunit='ns', seed=1)

longitudinal_bigaussian(general_params_copy, RF_sct_par_copy, my_beam_copy, tau_0/4, 
                              xunit='ns', seed=1)

number_slices = 2**8
slice_beam = Slices(my_beam, number_slices, cut_left = 0, 
                    cut_right = 2 * np.pi / harmonic_number, 
                    cuts_coord = 'theta', slicing_coord = 'tau', 
                    fit_option = 'gaussian')
slice_beam.track(my_beam)
slice_beam_copy = Slices(my_beam_copy, number_slices, cut_left = 0, 
                    cut_right = 2 * np.pi / harmonic_number, 
                    cuts_coord = 'theta', slicing_coord = 'tau', 
                    fit_option = 'gaussian')
slice_beam_copy.track(my_beam_copy)

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor('../output_files/TC5_bunch', n_turns+1, slice_beam)
slicesmonitor = SlicesMonitor('../output_files/TC5_slices', n_turns+1, slice_beam)
bunchmonitor.track(my_beam)
slicesmonitor.track(my_beam)

bunchmonitor_copy = BunchMonitor('../output_files/TC5_bunch_copy', n_turns+1, slice_beam_copy)
slicesmonitor_copy = SlicesMonitor('../output_files/TC5_slices_copy', n_turns+1, slice_beam_copy)
bunchmonitor_copy.track(my_beam_copy)
slicesmonitor_copy.track(my_beam_copy)

# LOAD IMPEDANCE TABLE--------------------------------------------------------

table = np.loadtxt('../input_files/TC5_new_HQ_table.dat', comments = '!')
R_shunt = table[:, 2] * 10**6 
f_res = table[:, 0] * 10**9
Q_factor = table[:, 1]
resonator = Resonators(R_shunt, f_res, Q_factor)

ind_volt_time = InducedVoltageTime(slice_beam, [resonator])
ind_volt_freq = InducedVoltageFreq(slice_beam_copy, [resonator], 1e5)

tot_vol = TotalInducedVoltage(slice_beam, [ind_volt_time])
tot_vol_copy = TotalInducedVoltage(slice_beam_copy, [ind_volt_freq])

tot_vol.track(my_beam)
tot_vol_copy.track(my_beam_copy)


# ACCELERATION MAP-------------------------------------------------------------

map_ = [tot_vol] + [ring_RF_section] + [slice_beam] + [bunchmonitor] + [slicesmonitor]
map_copy = [tot_vol_copy] + [ring_RF_section_copy] + [slice_beam_copy] + [bunchmonitor_copy] + [slicesmonitor_copy]



plot_long_phase_space(my_beam, general_params, RF_sct_par, 
  0, 0.0014, - 1.5e2, 1.5e2, sampling=50, dirname = '../output_files/TC5_fig/1')
plot_long_phase_space(my_beam_copy, general_params_copy, RF_sct_par_copy, 
  0, 0.0014, - 1.5e2, 1.5e2, sampling=50, dirname = '../output_files/TC5_fig/2')

# TRACKING + PLOTS-------------------------------------------------------------

for i in range(1, n_turns+1):
    
    print i
    for m in map_:
        m.track(my_beam)
    for m in map_copy:
        m.track(my_beam_copy)
    
    # Plots
    if (i % n_turns_between_two_plots) == 0:
        
        plot_induced_voltage_vs_bins_centers(i, general_params, tot_vol, style = '.', dirname = '../output_files/TC5_fig/1')
        plot_induced_voltage_vs_bins_centers(i, general_params_copy, tot_vol_copy, style = '.', dirname = '../output_files/TC5_fig/2')
        
        plot_long_phase_space(my_beam, general_params, RF_sct_par, 
          0, 0.0014, - 1.5e2, 1.5e2, sampling=50, dirname = '../output_files/TC5_fig/1')
        plot_long_phase_space(my_beam_copy, general_params_copy, RF_sct_par_copy, 
          0, 0.0014, - 1.5e2, 1.5e2, sampling=50, dirname = '../output_files/TC5_fig/2')
        
print "Done!"

bunchmonitor.h5file.close()
