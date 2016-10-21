from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import input_parameters.preprocess as prepClass
import input_parameters.general_parameters as genparClass
import llrf.rf_noise as rfnoise
import beams.beams as beamClass
import input_parameters.rf_parameters as rfparClass
import beams.slices as slicesClass
import trackers.tracker as traClass
import trackers.utilities as trautClass
import impedances.impedance as impClass
import beams.distributions as beamdistr
import impedances.music as musClass
from scipy.constants import m_p, e, c
from scipy.integrate import cumtrapz
from scipy.signal import *
from scipy.optimize import fsolve
import scipy.optimize as optimize
import sys, time, os, subprocess, warnings, math, ctypes
from setup_cpp import libblond
from shutil import copy2 
import time


n_turns = 1000000
C = 628.32
alpha = 0.0268745
tot_energy = 13e9
mass_rest = m_p*c**2/e
momentum = np.sqrt(tot_energy**2 - mass_rest**2)
n_particles = 6.4e-7/e
n_macroparticles = 10000000
n_rf_systems = 1
h_1 = 1
V_1 = 165e3
phi_1 = 0


general_params = genparClass.GeneralParameters(n_turns, C, alpha, momentum, 
                                   'proton')

rf_params = rfparClass.RFSectionParameters(general_params, n_rf_systems, 
                                        h_1, V_1, phi_1)

my_beam = beamClass.Beam(general_params, n_macroparticles, n_particles)
np.random.seed(1000)
my_beam.dt = 1e-9*np.random.randn(n_macroparticles) + general_params.t_rev[0]/2
my_beam.dE = 1e-9*np.random.randn(n_macroparticles) 



R_S = 4e4
frequency_R = 2.4e9
Q = 1

n_slices = 1000000
slices_ring = slicesClass.Slices(rf_params, my_beam, n_slices, cut_left = general_params.t_rev[0]/2 - 5e-9, cut_right = general_params.t_rev[0]/2 + 5e-9)
slices_ring.track()

mode = impClass.Resonators(R_S, frequency_R, Q)
ind_volt = impClass.InducedVoltageTime(slices_ring, [mode])
total_induced_voltage = impClass.TotalInducedVoltage(my_beam, slices_ring, [ind_volt])
t0 = time.clock()
total_induced_voltage.track()
# print time.clock()-t0

ind_volt2 = impClass.InducedVoltageFreq(slices_ring, [mode], None)
total_induced_voltage2 = impClass.TotalInducedVoltage(my_beam, slices_ring, [ind_volt2])
t0 = time.clock()
total_induced_voltage2.track()
# print time.clock()-t0

print "passage"

n_macroparticles2 = 1000000
my_beam2 = beamClass.Beam(general_params, n_macroparticles2, n_particles)
my_beam2.dt = 1e-9*np.random.randn(n_macroparticles2) + general_params.t_rev[0]/2
my_beam2.dE = 1e-9*np.random.randn(n_macroparticles2) 
music = musClass.Music(my_beam2, [R_S, 2*np.pi*frequency_R, Q], n_macroparticles2, n_particles)
music2 = musClass.Music(my_beam2, [R_S, 2*np.pi*frequency_R, Q], n_macroparticles2, n_particles)
t0 = time.clock()
music.track()
print time.clock()-t0

# t0 = time.clock()
# music2.track_classic()
# print time.clock()-t0


print 'h'

plt.plot(my_beam2.dt*1e9, music.induced_voltage, '-')
plt.plot(slices_ring.bin_centers*1e9, total_induced_voltage.induced_voltage)
plt.plot(slices_ring.bin_centers*1e9, total_induced_voltage2.induced_voltage)
# plt.plot(my_beam2.dt*1e9, music2.induced_voltage, 'o')
plt.grid()
plt.show()    
    
