
# Copyright 2014-2017 CERN. This software is distributed under the
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
import os
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pylab as plt
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p
from blond.utils import bmath as bm
from blond.utils.mpi_config import worker, mpiprint
bm.use_mpi()
print = mpiprint

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../mpi_output_files/EX_18_fig', exist_ok=True)


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
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
n_turns = int(10000)
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

number_slices = 100
slice_beam = Profile(beam, CutOptions(cut_left=0, 
                    cut_right=bucket_length, n_slices=number_slices)) 
                
# LOAD IMPEDANCE TABLES -------------------------------------------------------

R_S = 1e5
# - unstable, + stable
frequency_R = RF_sct_par.omega_rf[0,0] / 2.0 / np.pi - fs
Q = 1000

resonator = Resonators(R_S, frequency_R, Q)

# Robinson instability growth rate
Rp = R_S / (1 + 1j * Q * ((RF_sct_par.omega_rf[0,0] / 2.0 / np.pi+fs) / \
     frequency_R - frequency_R / (RF_sct_par.omega_rf[0,0] / 2.0 / np.pi+fs)))
Rm = R_S / (1 + 1j * Q * ((RF_sct_par.omega_rf[0,0] / 2.0 / np.pi-fs) / \
     frequency_R - frequency_R / (RF_sct_par.omega_rf[0,0] / 2.0 / np.pi-fs)))

etta = 1.0/gamma_transition**2 - 1.0/gamma**2

tau_RS = m_p * c**2.0 * 2.0 * gamma * \
         RF_sct_par.t_rev[0]**2.0 * RF_sct_par.omega_s0[0] / \
         (e**2.0 * n_particles * etta * RF_sct_par.omega_rf[0,0] * \
         np.real(Rp - Rm))

print('Robinson instability growth rate = {0:1.3f} turns'.format(tau_RS / \
                                                          RF_sct_par.t_rev[0]))


# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                    RFParams=RF_sct_par, frequency_resolution=5e2,
                    multi_turn_wake=True, mtw_mode='time')

total_ind_volt = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

f_rf = RF_sct_par.omega_rf[0,0] / 2.0 / np.pi
if worker.isMaster:
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
    plt.savefig(this_directory + '../mpi_output_files/EX_18_fig/impedance.png')
    plt.close()


# BEAM GENERATION -------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker,
                                  distribution_type=distribution_type,
                                  bunch_length=bunch_length, n_iterations=20,
                                  TotalInducedVoltage=total_ind_volt, seed=10)

# ACCELERATION MAP ------------------------------------------------------------

map_ = [slice_beam] + [total_ind_volt] + [ring_RF_section]

if worker.isMaster:
    # For testing purposes
    test_string = ''
    test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
        'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
    test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
        np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))


import time
t0 = time.time()

bunch_center = np.zeros(n_turns)
bunch_std = np.zeros(n_turns)

beam.split()
# TRACKING --------------------------------------------------------------------
for i in range(n_turns):
    
    print(i)
    for m in map_:
        m.track()

    beam.statistics()
    beam.gather_statistics()
    bunch_center[i] = beam.mean_dt
    
    # Not supported!
    bunch_std[i] = beam.dt.std()
    
    if i % n_turns_between_two_plots == 0:
        beam.gather()
        if worker.isMaster:
            plt.figure()
            plt.plot(beam.dt*1e9,beam.dE*1e-6,'.')
            plt.xlabel('Time [ns]')
            plt.ylabel('Energy [MeV]')
            plt.savefig(this_directory + '../mpi_output_files/EX_18_fig/phase_space_{0:d}.png'.format(i))
            plt.close()
        beam.split()

beam.gather()
worker.finalize()
print(time.time() - t0)

plt.figure()
plt.plot(bunch_center*1e9)
plt.xlabel('Turns')
plt.ylabel('Bunch center [ns]')
plt.savefig(this_directory + '../mpi_output_files/EX_18_fig/bunch_center.png')
plt.close()
plt.figure()
plt.plot(bunch_std*1e9)
plt.xlabel('Turns')
plt.ylabel('Bunch length [ns]')
plt.savefig(this_directory + '../mpi_output_files/EX_18_fig/bunch_length.png')
plt.close()

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
with open(this_directory + '../mpi_output_files/EX_16_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
