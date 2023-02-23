
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Calculation of the induced voltage for a gaussian bunch and a resonator.
Four different methods: time domain with convolution, frequency domain with
FFT, time domain with MuSiC, time domain with analytical formula.

:Authors: **Danilo Quartullo**
'''

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import blond.input_parameters.ring as genparClass
import blond.beam.beam as beamClass
import blond.input_parameters.rf_parameters as rfparClass
import blond.beam.profile as slicesClass
import blond.impedances.impedance as impClass
import blond.impedances.impedance_sources as impSClass
import blond.impedances.induced_voltage_analytical as indVoltAn
import blond.impedances.music as musClass
from scipy.constants import m_p, e, c
import os
import matplotlib as mpl
mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

fig_directory = this_directory + '../output_files/EX_11_fig/'
os.makedirs(fig_directory, exist_ok=True)


# RING PARAMETERS
n_turns = 1
radius = 25 
C = 2*np.pi*radius   
gamma_transition = 4.076750841  
alpha = 1/gamma_transition**2 
tot_energy = 13e9
mass_rest = m_p*c**2/e
momentum = np.sqrt(tot_energy**2 - mass_rest**2)
n_particles = 1e12
n_rf_systems = 1
h_1 = 1
V_1 = 24e3
phi_1 = 0

# RESONATOR PARAMETERS
R_S = 1e7
frequency_R = 1e8
Q = 1
mode = impSClass.Resonators(R_S, frequency_R, Q)

# DEFINE MAIN CLASSES
general_params = genparClass.Ring(C, alpha, momentum,
                                               beamClass.Proton(), n_turns)

rf_params = rfparClass.RFStation(general_params, [h_1], [V_1],
                                 [phi_1], n_rf_systems)

# DEFINE FIRST BEAM TO BE USED WITH SLICES (t AND f DOMAINS), AND VOLTAGE CALCULATION
n_macroparticles = 10000000
my_beam = beamClass.Beam(general_params, n_macroparticles, n_particles)
np.random.seed(1000)
sigma_gaussian = 3e-8
my_beam.dt = sigma_gaussian*np.random.randn(n_macroparticles) + general_params.t_rev[0]/2
my_beam.dE = sigma_gaussian*np.random.randn(n_macroparticles)
n_slices = 10000
cut_options = slicesClass.CutOptions(cut_left= 0, cut_right=general_params.t_rev[0], n_slices=n_slices)
slices_ring = slicesClass.Profile(my_beam, cut_options)
slices_ring.track()
ind_volt = impClass.InducedVoltageTime(my_beam, slices_ring, [mode])
total_induced_voltage = impClass.TotalInducedVoltage(my_beam, slices_ring, [ind_volt])
total_induced_voltage.track()
ind_volt2 = impClass.InducedVoltageFreq(my_beam, slices_ring, [mode], None)
total_induced_voltage2 = impClass.TotalInducedVoltage(my_beam, slices_ring, [ind_volt2])
total_induced_voltage2.track() 

# DEFINE SECOND BEAM TO BE USED WITH MUSIC, AND VOLTAGE CALCULATION
n_macroparticles2 = n_macroparticles
if n_macroparticles2 == n_macroparticles: 
    music = musClass.Music(my_beam, [R_S, 2*np.pi*frequency_R, Q], n_macroparticles, n_particles, rf_params.t_rev[0])
else:
    my_beam2 = beamClass.Beam(general_params, n_macroparticles2, n_particles)
    np.random.seed(1000)
    my_beam2.dt = sigma_gaussian*np.random.randn(n_macroparticles2) + general_params.t_rev[0]/2
    my_beam2.dE = sigma_gaussian*np.random.randn(n_macroparticles2)
    music = musClass.Music(my_beam2, [R_S, 2*np.pi*frequency_R, Q], n_macroparticles2, n_particles, rf_params.t_rev[0])
music.track_cpp_multi_turn()

# ANALYTICAL VOLTAGE CALCULATION
time_array = np.linspace(0, general_params.t_rev[0], 1000000)
induced_voltage_analytical = indVoltAn.analytical_gaussian_resonator(sigma_gaussian, Q, R_S, 2*np.pi*frequency_R, time_array-general_params.t_rev[0]/2, n_particles)

# PLOTS
if n_macroparticles2 == n_macroparticles: 
    plt.plot(my_beam.dt*1e9, music.induced_voltage, label='MuSiC')
else:
    plt.plot(my_beam2.dt*1e9, music.induced_voltage, label='MuSiC')
plt.plot(slices_ring.bin_centers*1e9, total_induced_voltage.induced_voltage, label='convolution')

plt.plot(slices_ring.bin_centers*1e9, total_induced_voltage2.induced_voltage, label='FFT')
plt.plot(time_array*1e9, induced_voltage_analytical, label='analytical')
plt.legend(loc='upper left')
plt.savefig(fig_directory+'output.png')    

print("Done!")
