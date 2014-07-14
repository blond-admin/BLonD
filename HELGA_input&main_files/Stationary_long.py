# Example input for stationary longitudinal simulation
# No intensity effects

import numpy as np
import time 
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt

from trackers.ring_and_RFstation import *
from trackers.longitudinal_tracker import *
from beams.beams import *
from beams.slices import *
from beams.longitudinal_distributions import *
from longitudinal_plots.longitudinal_plots import *
from FFT.RF_noise import Phase_noise
from monitors.monitors import *

# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001            # Macro-particles
tau_0 = 0.4       # Initial bunch length, 4 sigma [ns]
# sd = .5e-4           # Initial r.m.s. momentum spread

# Machine and RF parameters
h = []
V_rf = []
dphi = []
gamma_t = 55.759505  # Transition gamma
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h.append(35640)      # Harmonic number
V_rf.append(6.e6)    # RF voltage [eV]
dphi.append(0)      # Phase modulation/offset

# Tracking details
N_t = 2000           # Number of turns to track
dt_out = 200          # Time steps between output
dt_plt = 200          # Time steps between plots

# Derived parameters
#m_p *= c**2/e
# E_s = np.sqrt(p_s**2 + m_p**2 * c**4 / e**2)  # Sychronous energy [eV]
# gamma = E_s/(m_p*c**2/e)          # Relativistic gamma
# beta = np.sqrt(1. - 1./gamma**2)  # Relativistic beta
# T0 = C / beta / c                 # Turn period
# sigma_theta = 2 * np.pi * tau_0 / T0      # R.m.s. theta
# sigma_dE = sd * beta**2 * E_s           # R.m.s. dE
alpha = []
alpha.append(1./gamma_t/gamma_t)        # First order mom. comp. factor



# Pre-processing: RF phase noise -----------------------------------------------
f = np.arange(0, 5.6227612455e+03, 1.12455000e-02)
spectrum = np.concatenate((1.11100000e-07 * np.ones(4980), np.zeros(495021)))
noise_t, noise_dphi = Phase_noise(f, spectrum).spectrum_to_phase_noise()

# Hermitian vs complex FFT (gives the same result)
# plot_noise_spectrum(f, spectrum, sampling=100)
# plot_phase_noise(noise_t, noise_dphi, sampling=100)
# print "Sigma of noise 1 is %.4e" %np.std(noise_dphi)
# print "Time step of noise 1 is %.4e" %noise_t[1]
# f2 = np.arange(0, 2*5.62275e+03, 1.12455000e-02)
# spectrum2 = np.concatenate(( 1.11100000e-07 * np.ones(4980), np.zeros(990040), 1.11100000e-07 * np.ones(4980) ))
# noise_t2, noise_dphi2 = Phase_noise(f2, spectrum2).spectrum_to_phase_noise(transform='c')
# os.rename('fig/noise_spectrum.png', 'fig/noise_spectrum_r.png')
# os.rename('fig/phase_noise.png', 'fig/phase_noise_r.png')
# plot_noise_spectrum(f2, spectrum2, sampling=100)
# plot_phase_noise(noise_t2, noise_dphi2, sampling=100)
# print "Sigma of noise 2 is %.4e" %np.std(noise_dphi)
# print "Time step of noise 2 is %.4e" %noise_t[1]
# os.rename('fig/noise_spectrum.png', 'fig/noise_spectrum_c.png')
# os.rename('fig/phase_noise.png', 'fig/phase_noise_c.png')

plot_noise_spectrum(f, spectrum, sampling=100)
plot_phase_noise(noise_t, noise_dphi, sampling=100)
print "   Sigma of RF noise is %.4e" %np.std(noise_dphi)
print "   Time step of RF noise is %.4e" %noise_t[1]
print ""


# Simulation setup -------------------------------------------------------------
print "   Setting up the simulation..."
print ""


# Define Ring and RF Station
p_f = 500.005e9
#np.arange(p_s, p_f, 0.25e7)
ring = Ring_and_RFstation(C, p_s*np.ones(N_t+2), alpha, C, h, V_rf, dphi)

# Define Beam
beam = Beam(ring, m_p, N_p, e, N_b)
slices = Slices(100)
slices.set_longitudinal_cuts(beam)
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
print "RMS emittance %.4f eVs" %beam.eps_rms_l


# Accelerator map
map_ = [long_tracker] # No intensity effects, no aperture limitations
print "Map set"
print ""

# Outputs
output = BunchMonitor('output', N_t+1, statistics = "Longitudinal", long_gaussian_fit = "On")




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
        ring = Ring_and_RFstation(C, p_s*np.ones(N_t+2), alpha, C, h, V_rf, noise_dphi[i])
        m.track(beam)
    
#     print "Momentumi %.6e eV" %beam.p0_i()
#     print "Particle energy, theta %.6e %.6e" %(beam.dE[0], beam.theta[0])
    # Output data
    #if (i % dt_out) == 0:
    
#     print '{0:5d} \t {1:3e} \t {2:3e} \t {3:3e} \t {4:3e} \t {5:3e} \t {6:5d} \t {7:3s}'.format(i, bunch.slices.mean_dz[-2], bunch.slices.epsn_z[-2],
#                 bunch.slices.sigma_dz[-2], bunch.bl_gauss, 
#                 bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2], 
#                 str(time.clock() - t0))
    # Statistics
    slices.slice_constant_space_histogram(beam)
    #beam.longit_statistics(gaussian_fit='On', slices=slices)
    #print "4 sigma theta %.4e, Gaussian fit %.4e" %(4*beam.sigma_theta, beam.bl_gauss)
    output.dump(beam, slices)
    

    # Plot
    if (i % dt_plt) == 0:
        print "   Outputting at time step %6d" %i
        #plot_long_phase_space(ring, beam, i, -0.75, 0, -1.e-3, 1.e-3, xunit='m', yunit='1')
        plot_long_phase_space(beam, i, 0, 2.5, -.5e3, .5e3, xunit='ns')
        #plot_long_phase_space(beam, i, 0, 0.0001763, -450, 450)
        plot_bunch_length_evol(beam, 'output', i, unit='ns')
        plot_bunch_length_evol_gaussian(beam, 'output', i, unit='ns')



print "Done!"
print ""


