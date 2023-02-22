
# Copyright 2014-2017 CERN. This software is distributed under the
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
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.distributions_multibunch \
                            import matched_from_distribution_density_multibunch
from blond.beam.distributions_multibunch import matched_from_line_density_multibunch
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p
import os
import matplotlib as mpl
mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_20_fig/', exist_ok=True)


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
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
harmonic_numbers = 35640.0
voltage_program = 16e6
phi_offset = 0


# DEFINE RING------------------------------------------------------------------

general_params = Ring(C, momentum_compaction,
                                   sync_momentum, Proton(), n_turns)

RF_sct_par = RFStation(general_params, [harmonic_numbers], [voltage_program],
                       [phi_offset], n_rf_systems)

beam = Beam(general_params, n_macroparticles, n_particles)
ring_RF_section = RingAndRFTracker(RF_sct_par, beam)

full_tracker = FullRingAndRF([ring_RF_section])

fs = RF_sct_par.omega_s0[0]/2/np.pi    

bucket_length = 2.0 * np.pi / RF_sct_par.omega_rf[0,0]

# DEFINE SLICES ---------------------------------------------------------------

number_slices = 3000
slice_beam = Profile(beam, CutOptions(cut_left=0, 
                    cut_right=21*bucket_length, n_slices=number_slices))
                
# LOAD IMPEDANCE TABLES -------------------------------------------------------

R_S = 5e8
frequency_R = 2*RF_sct_par.omega_rf[0,0] / 2.0 / np.pi
Q = 10000

print('Im Z/n = '+str(R_S / (RF_sct_par.t_rev[0] * frequency_R * Q)))

resonator = Resonators(R_S, frequency_R, Q)


# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                                   frequency_resolution=5e4)

total_ind_volt = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

# BEAM GENERATION -------------------------------------------------------------

n_bunches = 3
bunch_spacing_buckets = 10
intensity_list = [1e11, 1e11, 1e11]
minimum_n_macroparticles = [5e5, 5e5, 5e5]
distribution_options_list = {'bunch_length': 1e-9,
                             'type': 'parabolic_amplitude',
                             'density_variable': 'Hamiltonian'}

matched_from_distribution_density_multibunch(beam, general_params,
                             full_tracker, distribution_options_list,
                             n_bunches, bunch_spacing_buckets,
                             intensity_list=intensity_list,
                             minimum_n_macroparticles=minimum_n_macroparticles
                             , seed=31331)

plt.figure()
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='without intensity effects')
         
matched_from_distribution_density_multibunch(beam, general_params,
                             full_tracker, distribution_options_list,
                             n_bunches, bunch_spacing_buckets,
                             intensity_list=intensity_list,
                             minimum_n_macroparticles=minimum_n_macroparticles,
                             TotalInducedVoltage=total_ind_volt,
                             n_iterations_input=10, seed=7878)


slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='with intensity effects')
         
plt.legend(loc=0, fontsize='medium')
plt.title('From distribution function')
plt.savefig(this_directory + '../output_files/EX_20_fig/from_distr_funct.png')

line_density_options_list = distribution_options_list
matched_from_line_density_multibunch(beam, general_params,
                        full_tracker, line_density_options_list, n_bunches,
                        bunch_spacing_buckets, intensity_list=intensity_list,
                        minimum_n_macroparticles=minimum_n_macroparticles
                        , seed=86867676)
                        
plt.figure('From line density')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='without intensity effects')

         
matched_from_line_density_multibunch(beam, general_params,
                        full_tracker, line_density_options_list, n_bunches,
                        bunch_spacing_buckets, intensity_list=intensity_list,
                        minimum_n_macroparticles=minimum_n_macroparticles,
                        TotalInducedVoltage=total_ind_volt, seed=12)

plt.figure('From line density')
slice_beam.track()
plt.plot(slice_beam.bin_centers, slice_beam.n_macroparticles, lw=2,
         label='with intensity effects')
plt.title('From line density')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + '../output_files/EX_20_fig/from_line_density.png')

print("Done!")