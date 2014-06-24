# Example input for stationary longitudinal simulation
# No intensity effects

import numpy as np
import time 
from scipy.constants import c, e, m_p

from trackers.ring_and_RFstation import *
from trackers.longitudinal_tracker import *
from beams.beams import *
from beams.longitudinal_distributions import *
from longitudinal_plots.longitudinal_plots import *


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 100000           # Macro-particles
tau_0 = 2.       # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
h = []
V_rf = []
dphi = []
gamma_t = 1/np.sqrt(0.00192)  # Transition gamma
C = 6911.56            # Machine circumference [m]
p_s = 25.92e9         # Synchronous momentum [eV]
h.append(4620)      # Harmonic number
V_rf.append(0.9e6)    # RF voltage [eV]
dphi.append(0)      # Phase modulation/offset

# Tracking details
N_t = 2000            # Number of turns to track
dt_out = 20          # Time steps between output
dt_plt = 200          # Time steps between plots

# Derived parameters
#m_p *= c**2/e
E_s = np.sqrt(p_s**2 + m_p**2 * c**4 / e**2)  # Sychronous energy [eV]
gamma = E_s/(m_p*c**2/e)          # Relativistic gamma
beta = np.sqrt(1. - 1./gamma**2)  # Relativistic beta
T0 = C / beta / c                 # Turn period
sigma_theta = 2 * np.pi * tau_0 / T0      # R.m.s. theta
alpha = []
alpha.append(1./gamma_t/gamma_t)        # First order mom. comp. factor




# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Define Ring and RF Station
ring = Ring_and_RFstation(C, p_s * np.ones(N_t + 1), alpha, C, h, V_rf, dphi)

# Define Beam
beam = Beam(ring, m_p, N_p, e, N_b)
print "Momentumi %.6e eV" %beam.ring.p0_i()
print "Momentumf %.6e eV" %beam.ring.p0_f()
print "Gammai %3.3f" %beam.ring.gamma_i(beam)
print "Gammaf %3.3f" %beam.ring.gamma_f(beam)
print "Betai %.6e" %beam.ring.beta_i(beam)
print "Betaf %.6e" %beam.ring.beta_f(beam)
print "Energyi %.6e eV" %beam.ring.energy_i(beam)
print "Energyf %.6e eV" %beam.ring.energy_f(beam)
print ""

# Choose Tracker 
long_tracker = Longitudinal_tracker(ring)
print "RF station set"

# Choose Distribution
#distribution = longitudinal_bigaussian(beam, sigma_theta*10, sigma_dE*10)
distribution = longitudinal_gaussian_matched(beam, tau_0, unit='ns')
print "Initial distribution set"


beam.longit_statistics()
print "Sigma dE %.2f MeV" %(beam.sigma_dE*1e-6)
print "Sigma theta %.4e rad" %beam.sigma_theta
print "RMS emittance %.4f eVs" %beam.epsn_rms_l


# Accelerator map
map_ = [long_tracker] # No intensity effects, no aperture limitations
print "Map set"
print ""

# print beam.dE
# print beam.theta
# print ""
# print beam.delta
# print beam.z
#plot_long_phase_space(beam, ring, 0, -1.5, 1.5, -1.e-3, 1.e-3, unit='ns')

# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()
    
    # Track
    for m in map_:
        m.track(beam)
    
#     print "Momentumi %.6e eV" %beam.p0_i()
#     print "Particle energy, theta %.6e %.6e" %(beam.dE[0], beam.theta[0])
    # Output data
    #if (i % dt_out) == 0:
    
#     print '{0:5d} \t {1:3e} \t {2:3e} \t {3:3e} \t {4:3e} \t {5:3e} \t {6:5d} \t {7:3s}'.format(i, bunch.slices.mean_dz[-2], bunch.slices.epsn_z[-2],
#                 bunch.slices.sigma_dz[-2], bunch.bl_gauss, 
#                 bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2], 
#                 str(time.clock() - t0))

    

    # Plot
    if (i % dt_plt) == 0:
        #plot_long_phase_space(ring, beam, i, -0.75, 0, -1.e-3, 1.e-3, xunit='m', yunit='1')
        #plot_long_phase_space(ring, beam, i, 0, 2.5, -.5e3, .5e3, xunit='ns', yunit='MeV')
        plot_long_phase_space(beam, i, 0, 5., -150, 150, xunit='ns')
#        plot_bunch_length_evol(bunch, 'bunch', i, unit='ns')
#        plot_bunch_length_evol_gaussian(bunch, 'bunch', i, unit='ns')



print "Done!"
print ""


