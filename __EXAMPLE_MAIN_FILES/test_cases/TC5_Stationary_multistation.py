
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulating a ring with multiple RF stations
No intensity effects
'''

import time 

from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from trackers.utilities import *
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
tau_0 = 0.4          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V1 = 2.e6            # RF voltage, station 1 [eV]
V2 = 4.e6            # RF voltage, station 1 [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2001           # Number of turns to track
dt_plt = 200         # Time steps between plots



# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Define general parameters containing data for both RF stations
general_params = GeneralParameters(N_t, [0.3*C, 0.7*C], [[alpha], [alpha]], 
                                   [p_s*np.ones(N_t+1), p_s*np.ones(N_t+1)], 
                                   'proton', number_of_sections = 2)

# Define RF station parameters and corresponding tracker
rf_params_1 = RFSectionParameters(general_params, 1, h, V1, dphi, 1)
long_tracker_1 = RingAndRFSection(rf_params_1)

rf_params_2 = RFSectionParameters(general_params, 1, h, V2, dphi, 2)
long_tracker_2 = RingAndRFSection(rf_params_2)

# Define full voltage over one turn and a corresponding "overall" set of 
#parameters, which is used for the separatrix (in plotting and losses)
Vtot = total_voltage([rf_params_1, rf_params_2])
rf_params_tot = RFSectionParameters(general_params, 1, h, Vtot, dphi)
long_tracker_tot = RingAndRFSection(rf_params_tot)
beam_dummy = Beam(general_params, 0, N_b)
print "General and RF parameters set..."
print Vtot

# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
longitudinal_bigaussian(general_params, rf_params_tot, beam, tau_0/4, 
                              xunit = 'ns', reinsertion = 'on')
print "Beam set and distribution generated..."


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Slices(beam, 100, fit_option = 'gaussian')
slice_beam.track(beam)

# Define what to save in file
bunchmonitor = BunchMonitor('../output_files/TC5_output_data', N_t+1, slice_beam)
bunchmonitor.track(beam)

print "Statistics set..."


# Accelerator map
map_ = [long_tracker_1] + [long_tracker_2] + [slice_beam] + [bunchmonitor] # No intensity effects, no aperture limitations
print "Map set"
print ""



# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    
    t0 = time.clock()
       
    # Plot has to be done before tracking (at least for cases with separatrix)
    # Use the full voltage for plotting the separatrix
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
        plot_long_phase_space(beam, general_params, rf_params_tot, 0, 0.0001763, -450, 450, separatrix_plot = True, dirname = '../output_files/TC5_fig')

    # Track
    for m in map_:
        m.track(beam)
    # Update full RF counter
    long_tracker_tot.track(beam_dummy)

    # These plots have to be done after the tracking
    if (i % dt_plt) == 0:
        plot_bunch_length_evol(beam, '../output_files/TC5_output_data', general_params, i, unit='ns', dirname = '../output_files/TC5_fig')
        plot_bunch_length_evol_gaussian(beam, '../output_files/TC5_output_data', general_params, slice_beam, i, unit='ns', dirname = '../output_files/TC5_fig')
    
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params_tot)
    #beam.losses_longitudinal_cut(0.28e-4, 0.75e-4)


bunchmonitor.h5file.close()
print "Done!"
print ""


