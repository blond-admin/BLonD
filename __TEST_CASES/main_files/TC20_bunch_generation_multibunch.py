
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case for the bunch generation routines for multi bunch.
Example for the LHC at 7 TeV.

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
from beams.distributions_multibunch \
                        import matched_from_distribution_density_multibunch,\
                               match_beam_from_distribution
from beams.distributions_multibunch import matched_from_line_density_multibunch
from beams.slices import Slices
from impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = int(3e11)
n_macroparticles = int(1.5e6)
sync_momentum = 7e12 # [eV]
                        
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

number_slices = 3000
slice_beam = Slices(RF_sct_par, beam, number_slices, cut_left=0,
                    cut_right=21*bucket_length)
                
# LOAD IMPEDANCE TABLES -------------------------------------------------------

R_S = 5e8
frequency_R = 2*RF_sct_par.omega_RF[0,0] / 2.0 / np.pi
Q = 10000

print('Im Z/n = '+str(R_S / (RF_sct_par.t_rev[0] * frequency_R * Q)))

resonator = Resonators(R_S, frequency_R, Q)


# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                                   frequency_resolution=5e4)

total_ind_volt = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

# BEAM GENERATION -------------------------------------------------------------
# --- from phase space distribution function
n_bunches = 3
bunch_spacing_buckets = 10
intensity_list = [1e11, 1e11, 1e11]
minimum_n_macroparticles = [5e5, 5e5, 5e5]
#distribution_options_list = {'bunch_length': 1e-9,
#                             'type': 'parabolic_amplitude',
#                             'density_variable': 'Hamiltonian'}
distribution_options = {'type': 'binomial', 'exponent':1.5,
                             'emittance':None, 'bunch_length':1e-9,
                             'bunch_length_fit':'FWHM', 
                             'density_variable': 'Hamiltonian'}

# No intensity
matched_from_distribution_density_multibunch(beam, general_params,
                             full_tracker, distribution_options,
                             n_bunches, bunch_spacing_buckets,
                             intensity_list=intensity_list,
                             minimum_n_macroparticles=minimum_n_macroparticles)

plt.figure('from distribution function')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='without intensity effects')

# Intensity
matched_from_distribution_density_multibunch(beam, general_params,
                             full_tracker, distribution_options,
                             n_bunches, bunch_spacing_buckets,
                             intensity_list=intensity_list,
                             minimum_n_macroparticles=minimum_n_macroparticles,
                             TotalInducedVoltage=total_ind_volt,
                             n_iterations_input=10)

slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='with intensity effects, bunch after bunch gen.')

match_beam_from_distribution(beam, full_tracker, general_params,
                             distribution_options, n_bunches,
                             bunch_spacing_buckets,
                             TotalInducedVoltage=total_ind_volt,
                             n_iterations=10, n_points_potential=1e3)

plt.figure('from distribution function')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2, label='with intensity effects, whole beam gen.')

plt.legend(loc='best', fontsize='medium', frameon=False)
plt.title('From distribution function')
   
# --- from line density

line_density_options_list = distribution_options_list
matched_from_line_density_multibunch(beam, general_params,
                        full_tracker, line_density_options_list, n_bunches,
                        bunch_spacing_buckets, intensity_list=intensity_list,
                        minimum_n_macroparticles=minimum_n_macroparticles)
                        
plt.figure('From line density')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='without intensity effects')

         
matched_from_line_density_multibunch(beam, general_params,
                        full_tracker, line_density_options_list, n_bunches,
                        bunch_spacing_buckets, intensity_list=intensity_list,
                        minimum_n_macroparticles=minimum_n_macroparticles,
                        TotalInducedVoltage=total_ind_volt)

plt.figure('From line density')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='with intensity effects')
plt.title('From line density')
plt.legend(loc='best', fontsize='medium', frameon=False)