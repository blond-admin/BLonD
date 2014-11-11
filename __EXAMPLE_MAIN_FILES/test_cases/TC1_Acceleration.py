
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects
'''

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
N_p = 50000         # Macro-particles
tau_0 = 0.4          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450.e9         # Synchronous momentum [eV]
p_f = 460.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6.e6         # RF voltage [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2001           # Number of turns to track
dt_plt = 200       # Time steps between plots



# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, np.linspace(p_i, p_f, 2002), 
                                   'proton')

# Define RF station parameters and corresponding tracker
rf_params = RFSectionParameters(general_params, 1, h, V, dphi)
long_tracker = RingAndRFSection(rf_params)

print "General and RF parameters set..."


# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
longitudinal_bigaussian(general_params, rf_params, beam, tau_0/4, 
                        xunit = 'ns', reinsertion = 'on')


print "Beam set and distribution generated..."


# Need slices for the Gaussian fit
slice_beam = Slices(beam, 100, fit_option = 'gaussian')
slice_beam.track(beam)

# Define what to save in file
bunchmonitor = BunchMonitor('../output_files/TC1_output_data', N_t+1, slice_beam)
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
        plot_long_phase_space(beam, general_params, rf_params, 0, 0.0001763, -400, 400, separatrix_plot = True, dirname = '../output_files/TC1_fig')

    # Track
    for m in map_:
        m.track(beam)

    # These plots have to be done after the tracking
    if (i % dt_plt) == 0:
        plot_bunch_length_evol(beam, '../output_files/TC1_output_data', general_params, i, unit='ns', dirname = '../output_files/TC1_fig')
        plot_bunch_length_evol_gaussian(beam, '../output_files/TC1_output_data', general_params, slice_beam, i, unit='ns', dirname = '../output_files/TC1_fig')
    
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params)
    beam.losses_cut(0.28e-4, 0.75e-4)


bunchmonitor.h5file.close()
print "Done!"
print ""


