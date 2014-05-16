from __future__ import division
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants
from scipy.optimize import brentq

from beams.slices import *
from beams.bunch import *
from impedances.longitudinal_impedance import *
from trackers.longitudinal_tracker import *
from plots import *



## Simulation setup

charge = e
mass = m_p
n_particles = 8.e10 
gamma_t = 1/np.sqrt(0.00192)
C = 6911. # [m]
energy = 26e9 # total [eV]
n_turns = 500
nsigmaz = 3
n_macroparticles = 500000
n_slices = 100
R_frequency = 1.0e9 # [Hz]
Q = 1.
R_shunt = 0.23e6 # [Ohm/m]
RF_voltage = 0.9e6 # [V]
harmonic_number = 4620


## Longitudinal distribution from file

temp = np.loadtxt('distribution.crd')
z = temp[:,0]
dp = temp[:,1]
n_macroparticles = len(z)

## Initialize the bunch

bunch = Bunch(n_macroparticles, n_particles, charge, energy, mass)
bunch.z = z
bunch.dp = dp

## First slicing

z_cut_tail = - C / (2 * harmonic_number)
z_cut_head = C / (2 * harmonic_number)
slices = Slices(300, z_cut_tail = - C / (2 * harmonic_number), z_cut_head = C / (2 * harmonic_number), mode = 'const_space')
slices.track(bunch)

## Resonator wakefields
  
temp = np.loadtxt('Z_table.dat', comments = '!')
R_shunt = temp[:,2]*10**6
frequency = temp[:,0]*10**9
Q = temp[:,1]
 
## Calculate the z kick
  
wake = long_wake_analytical(R_shunt, frequency, Q, slices, bunch)
induced_voltage = np.dot(slices.n_macroparticles, wake.wake_matrix) * n_particles / n_macroparticles * e * (-1)
fact = 2 * np.pi * harmonic_number / C
def V_RF(z):
    return RF_voltage * np.sin(fact * z)
V_RF_array = np.zeros(slices.n_slices)
for i in range(slices.n_slices):
    V_RF_array[i] = V_RF(slices.z_centers[i])
V_effect = induced_voltage + V_RF_array
V_effect_interp = interp1d(slices.z_centers, V_effect)
z_kick = brentq(V_effect_interp, 0, slices.z_centers[-1])
print z_kick
# Add the following lines if you want to plot the various functions
plt.plot(slices.z_centers, V_effect, color = 'green')
plt.plot(slices.z_centers, V_RF_array, color = 'blue')
plt.plot(slices.z_centers, induced_voltage, color = 'yellow')
plt.plot(slices.z_centers, slices.n_macroparticles / np.max(slices.n_macroparticles) * np.max(V_RF_array), color = 'red')
plt.axhline(0, color = 'black')
plt.axvline(0, color = 'black')
plt.pause(5)
  
## Apply the z kick
  
bunch.z +=  z_kick
  
## Synchrotron motion
  
cavity = RFCavity(C, C, gamma_t, harmonic_number, RF_voltage, 0)
 
## Accelerator map
  
plt.ion()

for i in range(n_turns):
    t0 = time.clock() 
    cavity.track(bunch)
    slices.track(bunch)
    wake.slices = slices
    wake.track(bunch)
    bunch.longit_mean_and_std()
    print '{0:4d} \t {1:+3e} \t {2:3f} \t {3:3f} \t {4:3f} \t {5:3f} \t {6:3s}'.format \
         (i+1, bunch.mean_z, bunch.sigma_z, bunch.mean_dp, bunch.sigma_dp, bunch.epsn_z_dp, str(time.clock() - t0))
    ## Add the following lines if you want to plot on (z, dp) phase space
    if np.mod(i, 5) == 0:
        plt.figure(1)
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(z_cut_tail, z_cut_head)
#         ax.set_ylim(-3e-3, 3e-3)
        ax.plot(bunch.z, bunch.dp, marker='.', lw=0)
        plt.draw()
        plt.pause(1)
# 

