
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation with RF noise
No intensity effects
'''

import time 
import numpy as np

from llrf.rf_noise import *
from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from beams.beams import *
from beams.distributions import *
from beams.slices import *
from monitors.monitors import *
from plots.plot_beams import *
from plots.plot_llrf import *
from plots.plot_slices import *
from plots.plot import *


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V = 6e6             # RF voltage [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 200         # Number of turns to track
dt_plt = 20        # Time steps between plots


# Pre-processing: RF phase noise -----------------------------------------------
f = np.arange(0, 5.6227612455e+03, 1.12455000e-02)
spectrum = np.concatenate((1.11100000e-07 * np.ones(4980), np.zeros(495021)))
RFnoise = PhaseNoise(f, spectrum, seed1=1234, seed2=7564)
RFnoise.spectrum_to_phase_noise()

# Hermitian vs complex FFT (gives the same result)
# plot_noise_spectrum(f, spectrum, sampling=100)
# plot_phase_noise(noise_t, noise_dphi, sampling=100)
# print "Sigma of noise 1 is %.4e" %np.std(noise_dphi)
# print "Time step of noise 1 is %.4e" %noise_t[1]
# f2 = np.arange(0, 2*5.62275e+03, 1.12455000e-02)
# spectrum2 = np.concatenate(( 1.11100000e-07 * np.ones(4980), np.zeros(990040), 1.11100000e-07 * np.ones(4980) ))
# noise_t2, noise_dphi2 = Phase_noise(f2, spectrum2).spectrum_to_phase_noise(transform='c')
# os.rename('temp/noise_spectrum.png', 'temp/noise_spectrum_r.png')
# os.rename('temp/phase_noise.png', 'temp/phase_noise_r.png')
# plot_noise_spectrum(f2, spectrum2, sampling=100)
# plot_phase_noise(noise_t2, noise_dphi2, sampling=100)
# print "Sigma of noise 2 is %.4e" %np.std(noise_dphi)
# print "Time step of noise 2 is %.4e" %noise_t[1]
# os.rename('temp/noise_spectrum.png', 'temp/noise_spectrum_c.png')
# os.rename('temp/phase_noise.png', 'temp/phase_noise_c.png')

plot_noise_spectrum(f, spectrum, sampling=100, dirname = '../output_files/TC3_fig')
plot_phase_noise(RFnoise.t, RFnoise.dphi, sampling=100, dirname = '../output_files/TC3_fig')
#plot_phase_noise(RFnoise.t[0:10000], RFnoise.dphi[0:10000], sampling=1, dirname = '../output_files/TC4_fig')
print "   Sigma of RF noise is %.4e" %np.std(RFnoise.dphi)
print "   Time step of RF noise is %.4e" %RFnoise.t[1]
print ""


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""

# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, p_s, 'proton')

# Define RF station parameters and corresponding tracker
rf_params = RFSectionParameters(general_params, 1, h, V, 0, RFnoise.dphi[0:N_t+1])
beam = Beam(general_params, N_p, N_b)
long_tracker = RingAndRFSection(rf_params, beam)

print "General and RF parameters set..."


# Define beam and distribution

# Generate new distribution
longitudinal_bigaussian(general_params, rf_params, beam, tau_0/4, 
                              reinsertion = 'on', seed=1)

print "Beam set and distribution generated..."


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Slices(rf_params, beam, 100, fit_option = 'gaussian', cuts_unit = 'rad')

# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params, beam, '../output_files/TC3_output_data', Slices=slice_beam)


# PLOTS

format_options = {'dirname': '../output_files/TC3_fig', 'linestyle': '.'}
plots = Plot(general_params, rf_params, beam, dt_plt, N_t, 0, 
             0.0001763*h, -450e6, 450e6, xunit= 'rad',
             separatrix_plot= True, Slices = slice_beam, h5file = '../output_files/TC3_output_data', 
             histograms_plot = True, format_options = format_options)

# Accelerator map
map_ = [long_tracker] + [slice_beam] + [bunchmonitor] + [plots]
print "Map set"
print ""



# Tracking ---------------------------------------------------------------------
for i in range(1,N_t+1):
    t0 = time.clock()
    
    
    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        
        print "Outputting at time step %d..." %i
        print "   Beam momentum %.6e eV" %beam.momentum
        print "   Beam gamma %3.3f" %beam.gamma
        print "   Beam beta %3.3f" %beam.beta
        print "   Beam energy %.6e eV" %beam.energy
        print "   Four-times r.m.s. bunch length %.4e [s]" %(4.*beam.sigma_dt)
        print "   Gaussian bunch length %.4e [s]" %slice_beam.bl_gauss
        print ""
        

    # Track
    for m in map_:
        m.track()
        
    

    
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params, beam)
    beam.losses_longitudinal_cut(0., 2.5e-9)



print "Done!"
print ""





