
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


'''
Test case for the synchrotron frequency distribution routine in the utilities
of the tracker module.
Example for the LHC at 7 TeV. Single RF, double RF in BSM and BLM, and with
intensity effects. Comparison with analytical formula.

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import Resonators
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.trackers.utilities import synchrotron_frequency_distribution
from scipy.constants import m_p, e, c
from scipy.special import ellipk
import os
import matplotlib as mpl
mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


fig_directory = this_directory + '../output_files/EX_12_fig/'
os.makedirs(fig_directory, exist_ok=True)

# RING PARAMETERS
# Beam parameters
n_particles = int(20e11)
n_macroparticles = int(1e6)
sync_momentum = 7e12 # [eV]

distribution_type = 'gaussian'
emittance = 2.5
distribution_variable = 'Action'

# Machine and RF parameters
radius = 4242.89
gamma_transition = 55.76
C = 2 * np.pi * radius  # [m]

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy =  np.sqrt(sync_momentum**2 + E_0**2) # [eV]
momentum_compaction = 1 / gamma_transition**2

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 35640.0
voltage_program = 16e6
phi_offset = 0

# DEFINE RING------------------------------------------------------------------
n_turns = 1
general_params = Ring(C, momentum_compaction,
                                   sync_momentum, Proton(), n_turns)

RF_sct_par = RFStation(general_params, [harmonic_numbers], [voltage_program],
                       [phi_offset], n_rf_systems)

bucket_length = 2.0 * np.pi / RF_sct_par.omega_rf[0,0]

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)

# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = RingAndRFTracker(RF_sct_par,beam)
full_tracker = FullRingAndRF([longitudinal_tracker])


# DEFINE SLICES----------------------------------------------------------------

number_slices = 500
cut_options = CutOptions(cut_left= 0, cut_right=bucket_length, n_slices=number_slices)
slice_beam = Profile(beam, cut_options)


# Single RF -------------------------------------------------------------------
matched_from_distribution_function(beam, full_tracker, emittance=emittance, 
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable,
                                   main_harmonic_option='lowest_freq', seed=1256)

slice_beam.track()

[sync_freq_distribution_left, sync_freq_distribution_right], \
    [emittance_array_left, emittance_array_right], \
    [delta_time_left, delta_time_right], \
    particleDistributionFreq, synchronous_time = \
                         synchrotron_frequency_distribution(beam, full_tracker)

# Plot of the synchrotron frequency distribution
plt.figure('fs_distribution')
plt.plot(delta_time_left, sync_freq_distribution_left, lw=2, label='Left')
plt.plot(delta_time_right, sync_freq_distribution_right, 'r--', lw=2,
         label='Right')

## Analytical calculation of fs(phi)
gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0-1.0/gamma**2.0)
frev = beta * c / C
etta = 1/gamma**2.0- 1/gamma_transition**2.0
phi_s = np.pi
# Zero-amplitude synchrotron frequency
fs0 = frev * np.sqrt(harmonic_numbers * voltage_program * np.abs(etta * 
                              np.cos(phi_s))/(2*np.pi*beta**2*tot_beam_energy))
# Analytical synchrotron frequency distribution
phi = delta_time_left * 2.0 * np.pi / bucket_length
sync_freq_distribution_analytical = fs0 * np.pi / ( 2.0 *
                                                 ellipk(np.sin(phi/2.0)**2.0) )

plt.figure('fs_distribution')
plt.plot(delta_time_left, sync_freq_distribution_analytical, 'g-.', lw=2,
         label='Analytical')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Amplitude of particle oscillations [s]')
plt.ylabel('Synchrotron frequency [Hz]')
plt.title('Synchrotron frequency distribution')
plt.savefig(fig_directory+'fs_distribution.png')

# Particle distribution in synchrotron frequency
fs_dist, be = np.histogram(particleDistributionFreq,100)
plt.figure('distribution_in_fs')
plt.plot(0.5*(be[1:] + be[:-1]), fs_dist)
plt.xlabel('Synchrotron frequency [Hz]')
plt.title('Particle distribution in synchrotron frequency')
plt.savefig(fig_directory+'distribution_in_fs.png')

# Double RF BLM ---------------------------------------------------------------

# Plot single RF case for comparison
plt.figure('fs_distribution_DRF')
plt.plot(delta_time_left, sync_freq_distribution_left, lw=2, label='SRF')
plt.xlabel('Amplitude of particle oscillations [s]')
plt.ylabel('Synchrotron frequency [Hz]')
plt.title('Synchrotron frequency distribution')


# Cavities parameters
n_rf_systems = 2
harmonic_numbers = [35640.0, 35640.0*2.0]
voltage_program = [16e6, 8e6]
phi_offset = [0, 0]

RF_sct_par = RFStation(general_params, harmonic_numbers, voltage_program,
                       phi_offset, n_rf_systems)

longitudinal_tracker = RingAndRFTracker(RF_sct_par,beam)
full_tracker = FullRingAndRF([longitudinal_tracker])

beam_generation_output = matched_from_distribution_function(beam, full_tracker,
                                   distribution_type=distribution_type,
                                   emittance=emittance,
                                   distribution_variable=distribution_variable,
                                   main_harmonic_option='lowest_freq', seed=1256)
            
[sync_freq_distribution_left, sync_freq_distribution_right], \
    [emittance_array_left, emittance_array_right], \
    [delta_time_left, delta_time_right], \
    particleDistributionFreq, synchronous_time = \
                         synchrotron_frequency_distribution(beam, full_tracker)

# Plot of the synchrotron frequency distribution
plt.figure('fs_distribution_DRF')
plt.plot(delta_time_left, sync_freq_distribution_left, lw=2, label='DRF BLM')


# Double RF BSM ---------------------------------------------------------------
# Cavities parameters
harmonic_numbers = [35640.0, 35640.0*2.0]
voltage_program = [16e6, 8e6]
phi_offset = [0, np.pi]

RF_sct_par = RFStation(general_params, harmonic_numbers, voltage_program,
                       phi_offset, n_rf_systems)

longitudinal_tracker = RingAndRFTracker(RF_sct_par,beam)
full_tracker = FullRingAndRF([longitudinal_tracker])

[sync_freq_distribution_left, sync_freq_distribution_right], \
    [emittance_array_left, emittance_array_right], \
    [delta_time_left, delta_time_right], \
    particleDistributionFreq, synchronous_time = \
                         synchrotron_frequency_distribution(beam, full_tracker)

# Plot of the synchrotron frequency distribution
plt.figure('fs_distribution_DRF')
plt.plot(delta_time_left, sync_freq_distribution_left, 'r--', lw=2,
         label='DRF BSM')
plt.legend(loc=0, fontsize='medium')
# Value for the zero-amplitude synchrotron frequency in double RF with 
# second harmonic
plt.plot(0, fs0*np.sqrt(2), 'ko')
plt.savefig(fig_directory+'fs_distribution_DRF.png')

# With intensity effects ------------------------------------------------------

# DEFINE BEAM------------------------------------------------------------------

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = [35640.0]
voltage_program = [16e6]
phi_offset = [0]

RF_sct_par = RFStation(general_params, harmonic_numbers, voltage_program,
                       phi_offset, n_rf_systems)

longitudinal_tracker = RingAndRFTracker(RF_sct_par,beam)
full_tracker = FullRingAndRF([longitudinal_tracker])

# INDUCED VOLTAGE FROM IMPEDANCE-----------------------------------------------
R_S = 95e3
frequency_R = 10e9
Q = 1.0

frequency_resolution_input = 1e7

Zres = Resonators(R_S, frequency_R, Q)
ind_volt = InducedVoltageFreq(beam, slice_beam, [Zres],
                              frequency_resolution=frequency_resolution_input)
                     
total_induced_voltage = TotalInducedVoltage(beam, slice_beam, [ind_volt])

beam_generation_output = matched_from_distribution_function(beam, full_tracker,
                                   distribution_type=distribution_type,
                                   emittance=emittance,
                                   distribution_variable=distribution_variable,
                                   main_harmonic_option='lowest_freq',
                                   TotalInducedVoltage=total_induced_voltage,
                                   n_iterations=20, seed=1256)

[sync_freq_distribution_left, sync_freq_distribution_right], \
    [emittance_array_left, emittance_array_right], \
    [delta_time_left, delta_time_right], \
    particleDistributionFreq, synchronous_time = \
                         synchrotron_frequency_distribution(beam, full_tracker,
                                 TotalInducedVoltage=beam_generation_output[1])

# Plot of the synchrotron frequency distribution
plt.figure('fs_distribution_IE')
plt.plot(delta_time_left, sync_freq_distribution_left, lw=2, label='Left')
plt.plot(delta_time_right, sync_freq_distribution_right, 'r--', lw=2,
         label='Right')
phi = delta_time_left * 2.0 * np.pi / bucket_length
sync_freq_distribution_analytical = fs0 * np.pi / ( 2.0 *
                                                 ellipk(np.sin(phi/2.0)**2.0) )
plt.plot(delta_time_left, sync_freq_distribution_analytical, 'g-.', lw=2,
         label='Analytical no IE')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Amplitude of particle oscillations [s]')
plt.ylabel('Synchrotron frequency [Hz]')
plt.title('Synchrotron frequency distribution in single RF with intensity ' +
          'effects')
plt.savefig(fig_directory+'fs_distribution_IE.png')


plt.figure('fs_distribution_IE_J')
plt.plot(emittance_array_left, sync_freq_distribution_left, lw=2, label='Left')
plt.plot(emittance_array_right, sync_freq_distribution_right, 'r--', lw=2,
         label='Right')
plt.plot(emittance_array_left, sync_freq_distribution_analytical, 'g-.', lw=2,
         label='Analytical no IE')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Emittance [eVs]')
plt.ylabel('Synchrotron frequency [Hz]')
plt.title('Synchrotron frequency distribution in single RF with intensity ' +
          'effects')
plt.savefig(fig_directory+'fs_distribution_IE_J.png')

print("Done!")