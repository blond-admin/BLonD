from __future__ import division
import cProfile, itertools, ipdb, time, timeit
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants
from scipy.optimize import brentq

from beams.bunch import *
from beams import slices
from monitors.monitors import *
from aperture.aperture import *
from impedances.wake_fields  import *
from trackers.transverse_tracker import *
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


## Synchrotron motion

cavity = RFCavity(C, C, gamma_t, harmonic_number, RF_voltage, 0, integrator='rk4')


## Longitudinal distribution from file

temp = np.loadtxt('distribution.crd')
dz = temp[:,0]
dp = temp[:,1]

## Initialize the bunch

n_macroparticles = len(dz)
zeroarray = np.zeros(n_macroparticles)
bunch = Bunch(zeroarray, zeroarray, zeroarray, zeroarray, dz, dp)
bunch.set_beam_physics(n_macroparticles, charge, energy, mass)
bunch.set_beam_numerics()
bunch.set_slices(Slices(n_slices, nsigmaz, 'cspace'))
bunch.update_slices()

## Resonator wakefields

temp = np.loadtxt('Z_table.dat', comments = '!')
R_shunt = temp[:,2]*10**6
R_frequency = temp[:,0]*10**9
Q = temp[:,1]
wakes = BB_Resonator_longitudinal(R_shunt=R_shunt, frequency=R_frequency, Q=Q)

## Calculate the z kick

wakekick = BB_Resonator_longitudinal(R_shunt=R_shunt, frequency=R_frequency, Q=Q)
wake = wakekick.wake_longitudinal 
dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] - np.transpose([bunch.slices.dz_centers[1:-2]])
wakekick.longitudinal_kick = np.zeros(bunch.slices.n_slices)
wakekick.longitudinal_kick = np.dot(bunch.slices.n_macroparticles[1:-3], wake(bunch, dz_to_target_slice)) * n_particles / n_macroparticles * e * (-1)
fact = 2 * np.pi * harmonic_number / C
def V_RF(z):
    return RF_voltage * np.sin(fact * z)
V_RF_array = np.zeros(bunch.slices.n_slices)
for i in range(bunch.slices.n_slices):
    V_RF_array[i] = V_RF(bunch.slices.dz_centers[i+1])
V_effect = wakekick.longitudinal_kick + V_RF_array
V_effect_interp = interp1d(bunch.slices.dz_centers[1:-2], V_effect)
z_kick = brentq(V_effect_interp, bunch.slices.dz_centers[1], bunch.slices.dz_centers[-3])
print z_kick
## Add the following lines if you want to plot the various functions
# plt.plot(bunch.slices.dz_centers[1:-2], V_effect)
# plt.plot(bunch.slices.dz_centers[1:-2], V_RF_array)
# plt.plot(bunch.slices.dz_centers[1:-2], wakekick.longitudinal_kick)
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.pause(100000)

## Apply the z kick

bunch.dz +=  z_kick

## Accelerator map
map_ = [wakes] + [cavity] 
for i in range(n_turns):
    t0 = time.clock() 
    for m in map_:
        m.track(bunch)
    print '{0:4d} \t {1:+3e} \t {2:3f} \t {3:3f} \t {4:3f} \t {5:4e} \t {6:3s}'.format(i, bunch.slices.mean_dz[-2], bunch.slices.epsn_z[-2], bunch.slices.sigma_dz[-2], bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2] / bunch.n_macroparticles * bunch.n_particles, str(time.clock() - t0))



