
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import division, print_function
from builtins import range
import numpy as np
from input_parameters.preprocess import *
from input_parameters.general_parameters import *
from beams.beams import *
from input_parameters.rf_parameters import *
from plots.plot_beams import *
from plots.plot_impedance import *
from plots.plot_slices import *
from plots.plot import *
from plots.plot_parameters import *
from beams.slices import *
from monitors.monitors import *
from trackers.tracker import *
import time 
from beams.distributions import *
from llrf.phase_loop import *

# Beam parameters
particle_type = 'proton'
n_macroparticles = 100000
n_particles = 0


# Machine and RF parameters
radius = 25 # [m]
gamma_transition = 4.076750841
alpha = 1 / gamma_transition**2
C = 2*np.pi*radius  # [m]     

n_turns = 500

general_params = GeneralParameters(n_turns, C, alpha, 310891054.809, 
                                   particle_type)

# Cavities parameters
n_rf_systems = 1                                     
harmonic_numbers_1 = 1
voltage_1 = 8000  # [V]  
phi_offset_1 = np.pi   # [rad]
rf_params = RFSectionParameters(general_params, n_rf_systems,
                                harmonic_numbers_1, voltage_1, phi_offset_1)

my_beam = Beam(general_params, n_macroparticles, n_particles)

slices_ring = Slices(rf_params, my_beam, 200, cut_left=0, cut_right=2.0*np.pi,
                     cuts_unit='rad')

#Phase loop
configuration = {'machine': 'PSB', 'PL_gain': 1./25.e-6*np.ones(n_turns+1),
                 'RL_gain': [0,0], 'PL_period': 10.e-6, 'RL_period': 7}
phase_loop = PhaseLoop(general_params, rf_params, slices_ring, configuration)


#Long tracker
long_tracker = RingAndRFSection(rf_params, my_beam, periodicity='Off',
                                PhaseLoop=phase_loop)

full_ring = FullRingAndRF([long_tracker])


distribution_type = 'gaussian'
bunch_length = 200.0e-9
distribution_variable = 'Action'

matched_from_distribution_function(my_beam, full_ring, 
                                   bunch_length=bunch_length,
                                   distribution_type=distribution_type, 
                                   distribution_variable=distribution_variable)
                                   
my_beam.dE += 90.0e3
slices_ring.track()

#Monitor
bunch_monitor = BunchMonitor(general_params, rf_params, my_beam,
                             '../output_files/TC8_output_data',
                             Slices=slices_ring, PhaseLoop=phase_loop)


#Plots
format_options = {'dirname': '../output_files/TC8_fig'}
plots = Plot(general_params, rf_params, my_beam, 50, n_turns, 0.0, 2*np.pi,
             -1e6, 1e6, xunit='rad', separatrix_plot=True, Slices=slices_ring,
             format_options=format_options,
             h5file='../output_files/TC8_output_data', PhaseLoop=phase_loop)


# Accelerator map
map_ = [full_ring] + [slices_ring] + [bunch_monitor] + [plots] 


for i in range(1, n_turns+1):
    print(i)
    t0 = time.clock()
    for m in map_:
        m.track()      
    print(time.clock()-t0) 
        
print('DONE')
