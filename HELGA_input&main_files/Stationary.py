# Test of PyHEADTAIL
# Stationary bucket

import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
import time 

from scipy.constants import c, e, m_p
from beams.beams import *
from beams import slices
from monitors.monitors import *
from trackers.longitudinal_tracker import *
from trackers.transverse_tracker import *
from longitudinal_plots.longitudinal_plots import *
from beams.transverse_distributions import *
from beams.longitudinal_distributions import *


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10000          # Macro-particles
tau_0 = 0.4e-9       # Initial bunch length, 4 sigma [s]
sd = .5e-4           # Initial r.m.s. momentum spread
N_sl = 100           # Number of slices
N_sz = 5.            # Bunch extension in +- z direction

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

# Machine and RF parameters
h = []
V_rf1 = []
V_rf2 = []
dphi = []
gamma_t = 55.759505  # Transition gamma
C = 26658.883        # Machine circumference [m]
#h = 35640            # Harmonic number
h.append(35640)      # Harmonic number
p_s = 450.e9         # Synchronous momentum [eV]
#V_rf = 6.e6          # RF voltage [eV]
V_rf1.append(6.e6/2)    # RF voltage [eV]
V_rf2.append(6.e6)
#dphi = 0.            # Phase modulation/offset
dphi.append(0.)      # Phase modulation/offset

# Tracking details
N_t = 200            # Number of turns to track
dt_out = 20          # Time steps between output
dt_plt = 50          # Time steps between plots

# Derived parameters
#m_p *= c**2/e
E_s = np.sqrt(p_s**2 + m_p**2 * c**4 / e**2)  # Sychronous energy [eV]
gamma = E_s/(m_p*c**2/e)          # Relativistic gamma
beta = np.sqrt(1. - 1./gamma**2)  # Relativistic beta
sz = tau_0*beta*c/4.              # R.m.s. bunch length in m
alpha = []
alpha.append(1./gamma_t/gamma_t)        # First order mom. comp. factor

print gamma
print beta
print alpha

# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Synchrotron motion
station0 = Ring_and_RFstation(C, p_s*np.ones(N_t+2), alpha, C, h, V_rf2, dphi)
station1 = Ring_and_RFstation(C, p_s*np.ones(N_t+2), alpha, C*0.3, h, V_rf1, dphi)
tracker1 = Longitudinal_tracker(station1)

station2 = Ring_and_RFstation(C, p_s*np.ones(N_t+2), alpha, C*0.7, h, V_rf1, dphi)
tracker2 = Longitudinal_tracker(station2)

#ring2 = Ring_and_RF(C, C*0.7, h, V_rf, dphi, alpha, p_s*np.ones(N_t+1))
#cavity2 = RFSystems(ring2)
#cavity = RFCavity(C, C, gamma_t, h, V_rf, 0.)#, integrator='euler-chromer')
#cavity = RFCavity(C, C, gamma_t, h, V_rf, np.pi)#, integrator='euler-chromer')
#cavity = RFCavityArray(C, gamma_t, h, V_rf, 0.)
print "Cavity set"

# Betatron motion (dummy)
#N_seg = 1
#s = np.arange(1, N_seg + 1) * C / N_seg
#linear_map = TransverseTracker.from_copy(s, [0], [beta_x], [0], [0], [beta_y], 
#                                         [0], Qx, Qp_x, 0, Qy, Qp_y, 0)
#print "Betatron motion set"

# Bunch initialisation
#print gamma/e*m_p*c**2
#print E_s
#bunch = bunch_matched_and_sliced(N_p, N_b, e, E_s, m_p, epsn_x, epsn_y, 
#                                 linear_map[0], sz, bucket=cavity, 
#                                 matching='simple', n_slices=N_sl, 
#                                 nsigmaz=N_sz, slicemode='cspace')
#print "Bunch initialised"
#ParticleMonitor('initial_distribution').dump(bunch)
#print "Initial distribution set"


beam = Beam(station0, m_p, N_p, e, N_b)
distribution = longitudinal_gaussian_matched(beam, 1., unit='ns')
#beam2 = Beam(ring1, m_p, N_p, e, N_b)
#bunch2 = as_cloud(ring1, beam2, 1, 1, 1, 1)
                      
print "Initial distribution set"


# Accelerator map
map_ = [tracker1] + [tracker2] # No intensity effects, no aperture limitations
print "Map set"
print ""

# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()
    
    # Track
    for m in map_:
        #print beam.theta
        #print beam.dE
#         print beam2.theta
#         print beam2.dE
#         print ""
        m.track(beam)

    
    

    # Output data
    #if (i % dt_out) == 0:
    
#     print '{0:5d} \t {1:3e} \t {2:3e} \t {3:3e} \t {4:3e} \t {5:3e} \t {6:5d} \t {7:3s}'.format(i, bunch.slices.mean_dz[-2], bunch.slices.epsn_z[-2],
#                 bunch.slices.sigma_dz[-2], bunch.bl_gauss, 
#                 bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2], 
#                 str(time.clock() - t0))

    # Plot
    if (i % dt_plt) == 0:
        plot_long_phase_space(beam, i, 0, 0.0001763, -450, 450)
    #    plot_long_phase_space(bunch, cavity, i, -1.5, 1.5, -1.e-3, 1.e-3, 
    #                           unit='ns')
#        plot_bunch_length_evol(bunch, 'bunch', i, unit='ns')
#        plot_bunch_length_evol_gaussian(bunch, 'bunch', i, unit='ns')



print "Done!"
print ""



