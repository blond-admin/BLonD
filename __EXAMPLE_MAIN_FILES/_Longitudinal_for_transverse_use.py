# Example input for transverse use of longitudinal packages
# No intensity effects

import time 

from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from beams.beams import *
from beams.distributions import *
from beams.slices import *
from monitors.monitors import *
from plots.plot_beams import *
from plots.plot_impedance import *
from plots.plot_slices import *

# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles
s_theta = 0.5e-5     # Initial r.m.s. theta [RF-rad]
s_dE = 50e6          # Initial r.m.s. dE [eV]   

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V = 6.e6             # RF voltage [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor
Qs = 4.9053e-03      # Synchrotron tune

# Tracking details
N_t = 2001           # Number of turns to track
dt_plt = 200         # Time steps between plots



# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, p_s, 'proton')

# Define longitudinal tracker for transverse use
long_tracker = LinearMap(general_params, Qs)

print "General and RF parameters set..."


# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
beam.theta = s_theta*np.random.randn(N_p)
beam.dE = s_dE*np.random.randn(N_p)

print "Beam set and distribution generated..."


# Need slices for the Gaussian fit
slice_beam = Slices(beam, 100, fit_option = 'gaussian', slice_immediately = 'on')


# Define what to save in file
bunchmonitor = BunchMonitor('output_data', N_t+1, slice_beam)
bunchmonitor.track(beam)

print "Statistics set..."


# Accelerator map
map_ = [long_tracker] + [slice_beam] + [bunchmonitor]  # No intensity effects, no aperture limitations
print "Map set"
print ""



# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()
    
    # Track
    for m in map_:
        m.track(beam)

    # Plotting
    if (i % dt_plt) == 0:
        print "Outputting at time step %d..." %i
        print "   Beam momentum %.6e eV" %beam.momentum
        print "   Beam gamma %3.3f" %beam.gamma_r
        print "   Beam beta %3.3f" %beam.beta_r
        print "   Beam energy %.6e eV" %beam.energy
        print "   Four-times r.m.s. bunch length %.4e rad" %(4.*beam.sigma_theta)
        print "   Gaussian bunch length %.4e rad" %beam.bl_gauss
        print ""
        # In plots, you can choose following units: rad, ns, m  
        plot_bunch_length_evol(beam, 'output_data', general_params, i, unit='ns')
        plot_bunch_length_evol_gaussian(beam, 'output_data', general_params, slice_beam, i, unit='ns')


bunchmonitor.h5file.close()
print "Done!"
print ""


