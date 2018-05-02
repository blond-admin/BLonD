
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects without multi-turn wakes
Example for the PSB with a narrow-band resonator, both in frequency and time
domain, and with an inductive impedance.

:Authors: **Juan F. Esteban Mueller**
'''


from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import pylab as plt
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from trackers.tracker import RingAndRFTracker
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from impedances.impedance import InducedVoltageFreq, InducedVoltageTime
from impedances.impedance import InductiveImpedance, TotalInducedVoltage
from impedances.impedance_sources import Resonators
from scipy.constants import c, e, m_p
import os

try:
    os.mkdir('../output_files')
except:
    pass
try:
    os.mkdir('../output_files/EX_16_fig')
except:
    pass


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
n_particles = 1e11
n_macroparticles = 1e6
sigma_dt = 180e-9 / 4 # [s]
kin_beam_energy = 1.4e9 # [eV]

# Machine and RF parameters
radius = 25.0
gamma_transition = 4.4
C = 2 * np.pi * radius  # [m]

# Tracking details
n_turns = 1
n_turns_between_two_plots = 1

# Derived parameters
E_0 = m_p*c**2/e    # [eV]
tot_beam_energy =  E_0 + kin_beam_energy # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2) # [eV / c]

gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0-1.0/gamma**2.0)

momentum_compaction = 1 / gamma_transition**2

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 1
voltage_program = 8e3 #[V]
phi_offset = np.pi


# DEFINE RING------------------------------------------------------------------

general_params = Ring(C, momentum_compaction,
                                   sync_momentum, Proton(), n_turns)

RF_sct_par = RFStation(general_params, [harmonic_numbers], [voltage_program],
                       [phi_offset], n_rf_systems)

beam = Beam(general_params, n_macroparticles, n_particles)
ring_RF_section = RingAndRFTracker(RF_sct_par, beam)

bucket_length = 2.0 * np.pi / RF_sct_par.omega_rf[0,0]

# DEFINE BEAM------------------------------------------------------------------
bigaussian(general_params, RF_sct_par, beam, sigma_dt, seed=1)


# DEFINE SLICES----------------------------------------------------------------

number_slices = int(100*2.5)

slice_beam = Profile(beam, CutOptions(cut_left=0, 
                    cut_right=bucket_length, n_slices=number_slices)) 

# LOAD IMPEDANCE TABLES--------------------------------------------------------
                     
ZoN = InductiveImpedance(beam, slice_beam, [100]*n_turns, RF_sct_par) 

R_S = 5e4
frequency_R = 10e6
Q = 1e2

resonator = Resonators(R_S, frequency_R, Q)

# INDUCED VOLTAGE FROM IMPEDANCE-----------------------------------------------

imp_list = [resonator]

ind_volt_freq = InducedVoltageFreq(beam, slice_beam, imp_list,
                    frequency_resolution=1e4)

ind_volt_time = InducedVoltageTime(beam, slice_beam, imp_list)

total_ind_volt_freq = TotalInducedVoltage(beam, slice_beam, [ind_volt_freq])

total_ind_volt_time = TotalInducedVoltage(beam, slice_beam, [ind_volt_time])

total_ind_volt_ZoN = TotalInducedVoltage(beam, slice_beam, [ZoN])

# PLOTS 

# ACCELERATION MAP-------------------------------------------------------------

map_ = [slice_beam] + [total_ind_volt_freq] + [total_ind_volt_time] + \
       [total_ind_volt_ZoN] + [ring_RF_section]

# TRACKING + PLOTS-------------------------------------------------------------

for i in range(n_turns):
    
    print(i)
    for m in map_:
        m.track()


plt.figure()
plt.plot(slice_beam.bin_centers*1e9, total_ind_volt_freq.induced_voltage, 
         lw=2, label='Resonator freq. domain')
plt.plot(slice_beam.bin_centers*1e9, total_ind_volt_time.induced_voltage,
         lw=2, alpha=0.75, label='Resonator time domain')
plt.plot(slice_beam.bin_centers*1e9, total_ind_volt_ZoN.induced_voltage,
         lw=2, alpha=0.75, label=r'Z/n = 100 $\Omega$')
plt.xlabel('Time [ns]')
plt.ylabel('Induced voltage [V]')
plt.legend(loc=2, fontsize='medium')

plt.savefig('../output_files/EX_16_fig/fig.png')

print("Done!")
