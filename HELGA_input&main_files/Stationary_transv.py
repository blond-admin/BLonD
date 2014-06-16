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
N_p = 10001            # Macro-particles
tau_0 = 0.4e-9       # Initial bunch length, 4 sigma [s]
sd = .5e-4           # Initial r.m.s. momentum spread

# Machine and RF parameters
h = []
V_rf = []
dphi = []
gamma_t = 55.759505  # Transition gamma
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h.append(35640)      # Harmonic number
V_rf.append(6.e6)    # RF voltage [eV]
dphi.append(0.)      # Phase modulation/offset

# Tracking details
N_t = 200            # Number of turns to track
dt_out = 20          # Time steps between output
dt_plt = 20          # Time steps between plots

# Derived parameters
#m_p *= c**2/e
E_s = np.sqrt(p_s**2 + m_p**2 * c**4 / e**2)  # Sychronous energy [eV]
gamma = E_s/(m_p*c**2/e)          # Relativistic gamma
beta = np.sqrt(1. - 1./gamma**2)  # Relativistic beta
T0 = C / beta / c                 # Turn period
sigma_theta = 2 * np.pi * tau_0 / T0      # R.m.s. theta
sigma_dE = sd * beta**2 * E_s           # R.m.s. dE
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


# Define beam w/o defining the ring
beam = Beam(R, m_p, N_p, e, N_b, p_s*np.ones(N_t+1))
print "Momentumi %.6e eV" %beam.p0_i
print "Momentumf %.6e eV" %beam.p0_f
print "Gammai %3.3f" %beam.gamma_i()
print "Gammaf %3.3f" %beam.gamma_f()
print "Betai %.6e" %beam.beta_i()
print "Betaf %.6e" %beam.beta_f()
print "Energyi %.6e eV" %beam.energy_i()
print "Energyf %.6e eV" %beam.energy_f()
print ""

# Choose Tracker 
Qs = 4.9053e-03
long_tracker = LinearMap(C, 1/gamma_t**2, Qs)
print "RF station set"

# Choose Distribution
eta = 1 / gamma_t**2 - 1 / beam.gamma_i()**2
distribution = as_bunch(beam, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, R*eta/Qs, 0.1)
print "Initial distribution set"


beam.longit_statistics()
print "Sigma dE %.2f MeV" %(beam.sigma_dE*1e-6)
print "Sigma theta %.4e rad" %beam.sigma_theta
print "RMS emittance %.4f eVs" %beam.eps_rms_l


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
        #plot_long_phase_space(ring, beam, i, -0.75, 0, -1.e-3, 1.e-3, xunit='m', yunit='1')
        #plot_long_phase_space(ring, beam, i, 0, 2.5, -.5e3, .5e3, xunit='ns', yunit='MeV')
        plot_long_phase_space(beam, i, 0, 0.0001763, -450, 450)
#        plot_bunch_length_evol(bunch, 'bunch', i, unit='ns')
#        plot_bunch_length_evol_gaussian(bunch, 'bunch', i, unit='ns')



print "Done!"
print ""


