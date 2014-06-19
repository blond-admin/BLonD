# Example input for stationary longitudinal simulation for transverse use
# No intensity effects

import numpy as np
import time 
from scipy.constants import c, e, m_p

from trackers.ring_and_RFstation import *
from trackers.longitudinal_tracker import *
from beams.beams import *
from beams.transverse_distributions import *
from longitudinal_plots.longitudinal_plots import *


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles

# Machine and RF parameters
gamma_t = 55.759505  # Transition gamma
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]

# Tracking details
N_t = 2000            # Number of turns to track
dt_out = 20          # Time steps between output
dt_plt = 200          # Time steps between plots

# Derived parameters
#m_p *= c**2/e
alpha = []
alpha.append(1./gamma_t/gamma_t)        # First order mom. comp. factor
R = C/2/np.pi

# Transverse parameters (dummy)
epsn_x = 2.          # Horizontal emittance [um]
epsn_y = 2.          # Vertical emittance [um]
beta_x = 50.         # Horizontal betatron amplitude [m]
beta_y = 50.         # Vertical betatron amplitude [m]
alpha_x = 0.5        # Betatron amplitude fct. [1], dummy
alpha_y = 0.5        # Betatron amplitude fct. [1], dummy
Qx = 64.28           # Horizontal tune at injection
Qy = 59.31           # Vertical tune at injection
Qp_x = 0. 
Qp_y = 0. 


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Define beam w/ minimal ring properties
ring = Ring_and_RFstation(C, p_s*np.ones(N_t+1), alpha)
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
Qs = 4.9053e-03
long_tracker = LinearMap(ring, Qs)
print "RF station set"

# Choose Distribution
eta = 1 / gamma_t**2 - 1 / ring.gamma_i(beam)**2
distribution = as_bunch(beam, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, R*eta/Qs, 0.1)
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

    # Output data
    #if (i % dt_out) == 0:
    
#     print '{0:5d} \t {1:3e} \t {2:3e} \t {3:3e} \t {4:3e} \t {5:3e} \t {6:5d} \t {7:3s}'.format(i, bunch.slices.mean_dz[-2], bunch.slices.epsn_z[-2],
#                 bunch.slices.sigma_dz[-2], bunch.bl_gauss, 
#                 bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2], 
#                 str(time.clock() - t0))

    # Plot
    if (i % dt_plt) == 0:
        #plot_long_phase_space(beam, i, -0.75, 0, -1.e-3, 1.e-3, xunit='m', yunit='1')
        #plot_long_phase_space(beam, i, 0, 2.5, -.5e3, .5e3, xunit='ns', yunit='MeV')
        plot_long_phase_space(beam, i, 0, 0.0001763, -450, 450, separatrix='off')
#        plot_bunch_length_evol(bunch, 'bunch', i, unit='ns')
#        plot_bunch_length_evol_gaussian(bunch, 'bunch', i, unit='ns')



print "Done!"
print ""


