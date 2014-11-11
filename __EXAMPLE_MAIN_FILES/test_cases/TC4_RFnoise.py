
# Copyright 2014 CERN. This software is distributed under the
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

from llrf.RF_noise import *
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



# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles
tau_0 = 0.4          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V = 6.e6             # RF voltage [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 201           # Number of turns to track
dt_plt = 200         # Time steps between plots


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

plot_noise_spectrum(f, spectrum, sampling=100, dirname = '../output_files/TC4_fig')
plot_phase_noise(RFnoise.t, RFnoise.dphi, sampling=100, dirname = '../output_files/TC4_fig')
#plot_phase_noise(RFnoise.t[0:10000], RFnoise.dphi[0:10000], sampling=1, dirname = '../output_files/TC4_fig')
print "   Sigma of RF noise is %.4e" %np.std(RFnoise.dphi)
print "   Time step of RF noise is %.4e" %RFnoise.t[1]
print ""


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""

# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, p_s,'proton')

# Define RF station parameters and corresponding tracker
rf_params = RFSectionParameters(general_params, 1, h, V, RFnoise.dphi[0:N_t+1])
long_tracker = RingAndRFSection(rf_params)

print "General and RF parameters set..."


# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
# Generate new distribution
longitudinal_bigaussian(general_params, rf_params, beam, tau_0/4, 
                              xunit = 'ns', reinsertion = 'on')

print "Beam set and distribution generated..."


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Slices(beam, 100, slicing_coord = 'theta', fit_option = 'gaussian')
slice_beam.track(beam)

# Define what to save in file
bunchmonitor = BunchMonitor('../output_files/TC4_output_data', N_t+1, slice_beam)
bunchmonitor.track(beam)

print "Statistics set..."


# Accelerator map
map_ = [long_tracker] + [slice_beam] + [bunchmonitor] # No intensity effects, no aperture limitations
print "Map set"
print ""



# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()
    
   
    # Plot has to be done before tracking (at least for cases with separatrix)
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
        plot_long_phase_space(beam, general_params, rf_params, 0, 0.0001763, -450, 450, separatrix_plot = True, dirname = '../output_files/TC4_fig')
        plot_beam_profile(i, general_params, slice_beam, dirname = '../output_files/TC4_fig')

    # Track
    for m in map_:
        m.track(beam)
        
    # These plots have to be done after the tracking
    if (i % dt_plt) == 0:
        plot_bunch_length_evol(beam, '../output_files/TC4_output_data', general_params, i, unit='ns', dirname = '../output_files/TC4_fig')
        plot_bunch_length_evol_gaussian(beam, '../output_files/TC4_output_data', general_params, slice_beam, i, unit='ns', dirname = '../output_files/TC4_fig')

    
    # Define losses according to separatrix and/or longitudinal position
    #beam.losses_separatrix(general_params, rf_params)
    #beam.losses_longitudinal_cut(0.28e-4, 0.75e-4)


bunchmonitor.h5file.close()
print "Done!"
print ""





