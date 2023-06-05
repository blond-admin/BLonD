
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects with multi-turn wakes
Example for the PSB with a narrow-band resonator, both in frequency and time
domain.

:Authors: **Juan F. Esteban Mueller**
'''


from __future__ import division, print_function
from blond.trackers.tracker import RingAndRFTracker
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.impedances.impedance_sources import Resonators
from blond.impedances.impedance import (InducedVoltageFreq, InducedVoltageTime,
                                        TotalInducedVoltage)
from blond.beam.profile import CutOptions, Profile
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
import blond.utils.bmath as bm
from scipy.constants import c, e, m_p
import os

from builtins import range

import matplotlib as mpl
import numpy as np
import pylab as plt

mpl.use('Agg')


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

USE_GPU = os.environ.get('USE_GPU', '0')
if len(USE_GPU) and int(USE_GPU):
    USE_GPU = True
else:
    USE_GPU = False

os.makedirs(this_directory + '../gpu_output_files/EX_17_fig', exist_ok=True)

# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
n_particles = 1e11
n_macroparticles = 5e5
sigma_dt = 180e-9 / 4  # [s]
kin_beam_energy = 1.4e9  # [eV]

# Machine and RF parameters
radius = 25.0
gamma_transition = 4.4
C = 2 * np.pi * radius  # [m]

# Tracking details
n_turns = 5
n_turns_between_two_plots = 1

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy = E_0 + kin_beam_energy  # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV/c]

gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0 - 1.0 / gamma**2.0)

momentum_compaction = 1 / gamma_transition**2

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 1
voltage_program = 8e3  # [V]
phi_offset = np.pi


# DEFINE RING------------------------------------------------------------------

general_params = Ring(C, momentum_compaction,
                      sync_momentum, Proton(), n_turns)

RF_sct_par = RFStation(general_params, [harmonic_numbers], [voltage_program],
                       [phi_offset], n_rf_systems)

beam = Beam(general_params, n_macroparticles, n_particles)
ring_RF_section = RingAndRFTracker(RF_sct_par, beam)

bucket_length = 2.0 * np.pi / RF_sct_par.omega_rf[0, 0]

# DEFINE BEAM------------------------------------------------------------------

bigaussian(general_params, RF_sct_par, beam, sigma_dt, seed=1)


# DEFINE SLICES----------------------------------------------------------------

number_slices = 200
slice_beam = Profile(beam, CutOptions(cut_left=0,
                                      cut_right=bucket_length, n_slices=number_slices))


# Overwriting the slices by a Gaussian profile (no slicing noise)
slice_beam.n_macroparticles = (n_macroparticles * slice_beam.bin_size /
                               (sigma_dt * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 *
                                                                          (slice_beam.bin_centers - bucket_length / 2.0)**2.0 / sigma_dt**2.0))

# LOAD IMPEDANCE TABLES--------------------------------------------------------

R_S = 5e3
frequency_R = 10e6
Q = 10

resonator = Resonators(R_S, frequency_R, Q)

# INDUCED VOLTAGE FROM IMPEDANCE-----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                                   RFParams=RF_sct_par, frequency_resolution=1e3,
                                   multi_turn_wake=True, mtw_mode='time')

ind_volt_time = InducedVoltageTime(beam, slice_beam, imp_list,
                                   RFParams=RF_sct_par, wake_length=n_turns * bucket_length,
                                   multi_turn_wake=True)

ind_volt_freq_periodic = InducedVoltageFreq(beam, slice_beam, imp_list)

total_ind_volt_freq = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

total_ind_volt_time = TotalInducedVoltage(beam, slice_beam, [ind_volt_time])

total_ind_volt_freq_periodic = TotalInducedVoltage(beam, slice_beam,
                                                   [ind_volt_freq_periodic])


# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(),
    beam.dt.mean(), beam.dt.std())

# ACCELERATION MAP-------------------------------------------------------------

map_ = [total_ind_volt_freq] + [total_ind_volt_time]
# map_ = [] + [total_ind_volt_time]


total_ind_volt_freq_periodic.track()


if USE_GPU:
    bm.use_gpu()
    RF_sct_par.to_gpu()
    slice_beam.to_gpu()
    ring_RF_section.to_gpu()
    total_ind_volt_freq.to_gpu()
    total_ind_volt_time.to_gpu()


# FIRST COMPARISON: CONSTANT REVOLUTION FREQUENCY -----------------------------
for i in range(n_turns):

    for m in map_:
        m.track()


if USE_GPU:
    bm.use_cpu()
    RF_sct_par.to_cpu()
    slice_beam.to_cpu()
    ring_RF_section.to_cpu()
    total_ind_volt_freq.to_cpu()
    total_ind_volt_time.to_cpu()

plt.figure('comparison', figsize=[6, 4.5])
plt.plot(slice_beam.bin_centers * 1e9, total_ind_volt_freq.induced_voltage, lw=2,
         label='Z in freq. MTW in time')
plt.plot(slice_beam.bin_centers * 1e9, total_ind_volt_time.induced_voltage, lw=2,
         label='Z in time MTW in freq', alpha=.75)
plt.plot(slice_beam.bin_centers * 1e9,
         total_ind_volt_freq_periodic.induced_voltage,
         label='Z in freq. MTW from periodicity', lw=2, alpha=.75)

# Multi-turn wake calculated using a convolution in time
time_array = np.arange(-np.sum(RF_sct_par.t_rev[1:]), bucket_length,
                       slice_beam.bin_size)
profiles = np.zeros(time_array.shape)

for i in range(1, n_turns + 1):
    profiles += n_macroparticles * slice_beam.bin_size / (sigma_dt *
                                                          np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (time_array - bucket_length / 2.0 +
                                                                                                 np.sum(RF_sct_par.t_rev[i:-1]))**2.0 / sigma_dt**2.0)

ind_volt = - beam.Particle.charge * e * beam.ratio * \
    np.convolve(profiles, ind_volt_time.total_wake)

plt.plot(time_array * 1e9, ind_volt[:time_array.shape[0]], lw=2, alpha=0.75,
         label='"Manual" convolution')
plt.xlim(0, bucket_length * 1e9)
plt.xlabel('Time [ns]')
plt.ylabel('Induced voltage [V]')
plt.title('Constant revolution frequency')
plt.legend(loc=2, fontsize='x-small')
plt.savefig(this_directory + '../gpu_output_files/EX_17_fig/const_rev_f.png')

# SECOND COMPARISON: DIFFERENT REVOLUTION FREQUENCIES -------------------------

# Modify revolution period array
RF_sct_par.t_rev *= 1 - np.arange(n_turns + 1) * 10 / 100


if USE_GPU:
    bm.use_gpu()
    RF_sct_par.to_gpu()
    ring_RF_section.to_gpu()
    slice_beam.to_gpu()
    total_ind_volt_freq.to_gpu()
    total_ind_volt_time.to_gpu()

for i in range(n_turns):

    for m in map_:
        m.track()

    # Increasing turn counter manually because tracker is not called
    RF_sct_par.counter[0] += 1


if USE_GPU:
    bm.use_cpu()
    RF_sct_par.to_cpu()
    ring_RF_section.to_cpu()
    slice_beam.to_cpu()
    total_ind_volt_freq.to_cpu()
    total_ind_volt_time.to_cpu()

plt.figure('comparison2', figsize=[6, 4.5])
plt.plot(slice_beam.bin_centers * 1e9, total_ind_volt_freq.induced_voltage, lw=2,
         label='Z in freq. MTW in time')
plt.plot(slice_beam.bin_centers * 1e9, total_ind_volt_time.induced_voltage, lw=2,
         label='Z in time MTW in freq', alpha=.75)

# Multi-turn wake calculated using a convolution in time
time_array = np.arange(-np.sum(RF_sct_par.t_rev[1:]), bucket_length,
                       slice_beam.bin_size)
profiles = np.zeros(time_array.shape)
for i in range(1, n_turns + 1):
    profiles += n_macroparticles * slice_beam.bin_size / (sigma_dt *
                                                          np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (time_array - bucket_length / 2.0 +
                                                                                                 np.sum(RF_sct_par.t_rev[i:-1]))**2.0 / sigma_dt**2.0)

ind_volt = -(beam.Particle.charge * e * beam.ratio *
             np.convolve(profiles, ind_volt_time.total_wake))

plt.plot(time_array * 1e9, ind_volt[:time_array.shape[0]], lw=2, alpha=0.75,
         label='"Manual" convolution')
plt.xlim(0, bucket_length * 1e9)
plt.xlabel('Time [ns]')
plt.ylabel('Induced voltage [V]')
plt.title('Different revolution frequencies')
plt.legend(loc=2, fontsize='medium')
plt.savefig(this_directory + '../gpu_output_files/EX_17_fig/diff_rev_f.png')

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(),
    beam.dt.mean(), beam.dt.std())
with open(this_directory + '../gpu_output_files/EX_17_test_data.txt', 'w') as f:
    f.write(test_string)
print(test_string)
print("Done!")
