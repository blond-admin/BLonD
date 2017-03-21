
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects with multi-turn wakes
Example for the PSB with a narrow-band resonator, to check Robinson instability

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import pylab as plt
from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from trackers.tracker import RingAndRFSection, FullRingAndRF
from beams.beams import Beam
from beams.distributions import matched_from_distribution_function
from beams.slices import Slices
from impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = 1e11
n_macroparticles = 1e5
kin_beam_energy = 1.4e9     # [eV]


distribution_type = 'parabolic_line'
bunch_length = 100e-9        # [s]


# Machine and RF parameters
radius = 25.0
gamma_transition = 4.4
C = 2 * np.pi * radius      # [m]

# Tracking details
n_turns = int(1e4)
n_turns_between_two_plots = 500

# Derived parameters
E_0 = m_p*c**2/e            # [eV]
tot_beam_energy =  E_0 + kin_beam_energy                # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)    # [eV/c]

gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0-1.0/gamma**2.0)

momentum_compaction = 1 / gamma_transition**2

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 1
voltage_program = 8e3   #[V]
phi_offset = -np.pi


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

number_slices = 100
slice_beam = Slices(RF_sct_par, beam, number_slices, cut_left=0,
                    cut_right=bucket_length)
                
# LOAD IMPEDANCE TABLES -------------------------------------------------------

R_S = 1e5
# - unstable, + stable
frequency_R = RF_sct_par.omega_RF[0,0] / 2.0 / np.pi - fs
Q = 1000

resonator = Resonators(R_S, frequency_R, Q)

# Robinson instability growth rate
Rp = R_S / (1 + 1j * Q * ((RF_sct_par.omega_RF[0,0] / 2.0 / np.pi+fs) / \
     frequency_R - frequency_R / (RF_sct_par.omega_RF[0,0] / 2.0 / np.pi+fs)))
Rm = R_S / (1 + 1j * Q * ((RF_sct_par.omega_RF[0,0] / 2.0 / np.pi-fs) / \
     frequency_R - frequency_R / (RF_sct_par.omega_RF[0,0] / 2.0 / np.pi-fs)))

etta = 1.0/gamma_transition**2 - 1.0/gamma**2

tau_RS = m_p * c**2.0 * 2.0 * gamma * \
         RF_sct_par.t_rev[0]**2.0 * RF_sct_par.omega_s0[0] / \
         (e**2.0 * n_particles * etta * RF_sct_par.omega_RF[0,0] * \
         np.real(Rp - Rm))

print('Robinson instability growth rate = {0:1.3f} turns'.format(tau_RS / \
                                                          RF_sct_par.t_rev[0]))


# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                    RFParams=RF_sct_par, frequency_resolution=5e2,
                    multi_turn_wake=True, mtw_mode='time')

total_ind_volt = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

f_rf = RF_sct_par.omega_RF[0,0] / 2.0 / np.pi
plt.figure()
plt.plot(ind_volt_freq.freq*1e-6, np.abs(ind_volt_freq.total_impedance * \
                                         slice_beam.bin_size), lw=2)
plt.plot([f_rf*1e-6]*2, plt.ylim(), 'k', lw=2)
plt.plot([f_rf*1e-6 + fs*1e-6]*2, plt.ylim(), 'k--', lw=2)
plt.plot([f_rf*1e-6 - fs*1e-6]*2, plt.ylim(), 'k--', lw=2)
plt.xlim(1.74,1.76)
plt.legend(('Impedance','RF frequency','Synchrotron sidebands'), loc=0, 
           fontsize='medium')
plt.xlabel('Frequency [MHz]')
plt.ylabel(r'Impedance [$\Omega$]')
plt.savefig('../output_files/TC18_fig/impedance.png')
plt.close()


# BEAM GENERATION -------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker,
                                  distribution_type=distribution_type,
                                  bunch_length=bunch_length, n_iterations=20,
                                  TotalInducedVoltage=total_ind_volt)

# ACCELERATION MAP ------------------------------------------------------------

map_ = [slice_beam] + [total_ind_volt] + [ring_RF_section]

import time
t0 = time.time()

bunch_center = np.zeros(n_turns)
bunch_std = np.zeros(n_turns)


# TRACKING --------------------------------------------------------------------
for i in range(n_turns):
    for m in map_:
        m.track()
    
    bunch_center[i] = beam.dt.mean()
    bunch_std[i] = beam.dt.std()
    
    if i % n_turns_between_two_plots == 0:
        plt.figure()
        plt.plot(beam.dt*1e9,beam.dE*1e-6,'.')
        plt.xlabel('Time [ns]')
        plt.ylabel('Energy [MeV]')
        plt.savefig('../output_files/TC18_fig/phase_space_{0:d}.png'.format(i))
        plt.close()

print(time.time() - t0)

plt.figure()
plt.plot(bunch_center*1e9)
plt.xlabel('Turns')
plt.ylabel('Bunch center [ns]')
plt.savefig('../output_files/TC18_fig/bunch_center.png')
plt.close()
plt.figure()
plt.plot(bunch_std*1e9)
plt.xlabel('Turns')
plt.ylabel('Bunch length [ns]')
plt.savefig('../output_files/TC18_fig/bunch_length.png')
plt.close()

print("Done!")
