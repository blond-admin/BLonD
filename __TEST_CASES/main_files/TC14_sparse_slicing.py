# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


'''
Test case for the sparse_slicing routine.
Example for the FCC-ee at 175 GeV.

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division
from __future__ import print_function
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from input_parameters.general_parameters import GeneralParameters
from beams.beams import Beam
from beams.distributions import matched_from_distribution_function
from input_parameters.rf_parameters import RFSectionParameters
from beams.sparse_slices import SparseSlices
from trackers.tracker import RingAndRFSection, FullRingAndRF
from scipy.constants import c, e, m_e


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'electron'
n_particles = int(1.7e11)          
n_macroparticles = int(50e6)
sync_momentum = 175e9 # [eV]

distribution_type = 'gaussian'
emittance = 1.0
distribution_variable = 'Action'

# Machine and RF parameters
radius = 15915.49
gamma_transition = 377.96447
C = 2 * np.pi * radius  # [m]        
      
# Tracking details
n_turns = int(200)
n_turns_between_two_plots = 100
 
# Derived parameters
E_0 = m_e * c**2 / e    # [eV]
tot_beam_energy =  np.sqrt(sync_momentum**2 + E_0**2) # [eV]
momentum_compaction = 1 / gamma_transition**2  

# Cavities parameters
n_rf_systems = 1                                
harmonic_numbers = [133650]                        
voltage_program = [10e9]
phi_offset = [np.pi]

bucket_length = C / c / harmonic_numbers[0]
print(bucket_length)

# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction,
                                   sync_momentum, particle_type)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems,
                                 harmonic_numbers, voltage_program, phi_offset)

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)

# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = RingAndRFSection(RF_sct_par,beam)

full_tracker = FullRingAndRF([longitudinal_tracker])


# DEFINE SLICES----------------------------------------------------------------

n_slices = 500

n_bunches = 80
bunch_spacing = 2       # buckets
filling_pattern = np.zeros(bunch_spacing*n_bunches)
filling_pattern[::bunch_spacing] = 1


# BEAM GENERATION--------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker, emittance=emittance,
                                   distribution_type=distribution_type, 
                                   distribution_variable=distribution_variable)

indexes = np.arange(n_macroparticles)
#np.random.shuffle(indexes)

for i in np.arange(np.sum(filling_pattern)):
    beam.dt[indexes[i*len(beam.dt)//np.sum(filling_pattern)]: 
        indexes[(i+1)*len(beam.dt)//np.sum(filling_pattern)-1]] += (
        bucket_length * np.where(filling_pattern)[0][i])

import time


slice_beam = SparseSlices(RF_sct_par, beam, n_slices, filling_pattern)

t0 = time.time()
slice_beam.track()
print( 'Time for optimized C++ track ', time.time() - t0 )
plt.figure()
for i in range(int(np.sum(filling_pattern))):
    plt.plot(slice_beam.slices_array[i].bin_centers,
             slice_beam.slices_array[i].n_macroparticles)
plt.show()


for i in range(int(np.sum(filling_pattern))):
    slice_beam.slices_array[i].n_macroparticles *= 0


slice_beam = SparseSlices(RF_sct_par, beam, n_slices, filling_pattern,
                          tracker='onebyone')

t0 = time.time()
slice_beam.track()
print( 'Time for individual tracks ', time.time() - t0 )
plt.figure()
for i in range(int(np.sum(filling_pattern))):
    plt.plot(slice_beam.slices_array[i].bin_centers,
             slice_beam.slices_array[i].n_macroparticles)
plt.show()
