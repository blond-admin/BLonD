
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case for the bunch generation routines. Example for the LHC at 7 TeV.

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division
from __future__ import print_function
from builtins import str
import numpy as np
import pylab as plt
from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from trackers.tracker import RingAndRFSection, FullRingAndRF
from beams.beams import Beam
from beams.distributions import matched_from_distribution_function
from beams.distributions import matched_from_line_density
from beams.slices import Slices
from impedances.impedance import InducedVoltageFreq, TotalInducedVoltage, InducedVoltageTime
from impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = int(1e11)
n_macroparticles = int(1e6)
sync_momentum = 7e12 # [eV]

distribution_exponent = None
bunch_length_fit = 'full'
distribution_type = 'parabolic_line'
#distribution_type = 'parabolic_amplitude'
#distribution_type = 'binomial'
#distribution_exponent = 5
#bunch_length_fit = None
#distribution_type = 'gaussian'
bunch_length = 0.5e-9        # [s]
                        
# Machine and RF parameters
radius = 4242.89
gamma_transition = 55.76
C = 2 * np.pi * radius  # [m]
      
# Tracking details
n_turns = int(1e4)
n_turns_between_two_plots = 500

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy =  np.sqrt(sync_momentum**2 + E_0**2) # [eV]
momentum_compaction = 1 / gamma_transition**2

gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0-1.0/gamma**2.0)

momentum_compaction = 1 / gamma_transition**2

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = [35640.0]
voltage_program = [16e6]
phi_offset = [0]


# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction,
                                   sync_momentum, particle_type)

RF_sct_par = RFSectionParameters(general_params, n_rf_systems,
                                 harmonic_numbers, voltage_program, phi_offset)

beam = Beam(general_params, n_macroparticles, n_particles)
ring_RF_section = RingAndRFSection(RF_sct_par, beam)

full_tracker = FullRingAndRF([ring_RF_section])

fs = RF_sct_par.omega_s0[0]/2/np.pi    

bucket_length = 2.0 * np.pi / RF_sct_par.omega_RF[0,0]

# DEFINE SLICES ---------------------------------------------------------------

number_slices = 200
slice_beam = Slices(RF_sct_par, beam, number_slices, cut_left=0,
                    cut_right=bucket_length)
                
# LOAD IMPEDANCE TABLES -------------------------------------------------------

R_S = 2e4*1000
frequency_R = 10*RF_sct_par.omega_RF[0,0] / 2.0 / np.pi   /10
Q = 100

print('Im Z/n = '+str(R_S / (RF_sct_par.t_rev[0] * frequency_R * Q)))

resonator = Resonators(R_S, frequency_R, Q)


# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                                   frequency_resolution=5e5)

#ind_volt_freq = InducedVoltageTime(beam, slice_beam, imp_list)

total_ind_volt = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

# BEAM GENERATION -------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker,
                                  distribution_type=distribution_type,
                                  distribution_exponent=distribution_exponent,
                                  bunch_length=bunch_length,
                                  bunch_length_fit=bunch_length_fit,
                                  distribution_variable='Action')

plt.figure()
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2, 
         label='from distribution function')
         
matched_from_line_density(beam, full_tracker, bunch_length=bunch_length,
                          line_density_type=distribution_type,
                          line_density_exponent=distribution_exponent)

slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2, 
         label='from line density')

plt.legend(loc=0, fontsize='medium')
plt.title('Without intensity effects')

matched_from_distribution_function(beam, full_tracker,
                                  distribution_type=distribution_type,
                                  distribution_exponent=distribution_exponent,
                                  bunch_length_fit=bunch_length_fit,
                                  bunch_length=bunch_length, n_iterations=10,
                                  TotalInducedVoltage=total_ind_volt,
                                  distribution_variable='Action')

plt.figure()
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2, 
         label='from distribution function')
         
matched_from_line_density(beam, full_tracker, bunch_length=bunch_length,
                          line_density_type=distribution_type,
                          line_density_exponent=distribution_exponent,
                          TotalInducedVoltage=total_ind_volt)

slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2, 
         label='from line density')

plt.legend(loc=0, fontsize='medium')
plt.title('With intensity effects')