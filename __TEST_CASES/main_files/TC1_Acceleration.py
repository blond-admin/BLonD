# Copyright 2016 CERN. This software is distributed under the
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
#  General Imports
from __future__ import division, print_function
from builtins import range
import numpy as np

#  BLonD Imports
from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from trackers.tracker import RingAndRFSection
from beams.beams import Beam
from beams.distributions import longitudinal_bigaussian
from beams.slices import Slices
from monitors.monitors import BunchMonitor
from plots.plot import Plot



# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 50000         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
p_f = 460.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track
dt_plt = 200         # Time steps between plots



# Simulation setup ------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, np.linspace(p_i, p_f, 2001), 
                                   'proton')

# Define beam and distribution
beam = Beam(general_params, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf_params = RFSectionParameters(general_params, 1, h, V, dphi)
long_tracker = RingAndRFSection(rf_params, beam)

longitudinal_bigaussian(general_params, rf_params, beam, tau_0/4, 
                        reinsertion = 'on', seed=1)


# Need slices for the Gaussian fit
slice_beam = Slices(rf_params, beam, 100, fit_option='gaussian')                     
                     
# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params, beam,
                          '../output_files/TC1_output_data', Slices=slice_beam)

format_options = {'dirname': '../output_files/TC1_fig'}
plots = Plot(general_params, rf_params, beam, dt_plt, N_t, 0, 0.0001763*h,
             -400e6, 400e6, xunit='rad', separatrix_plot=True, 
             Slices=slice_beam, h5file='../output_files/TC1_output_data', 
             format_options=format_options)

# Accelerator map
map_ = [long_tracker] + [slice_beam] + [bunchmonitor] + [plots]
print("Map set")
print("")



# Tracking --------------------------------------------------------------------
for i in range(1, N_t+1):
    
    
    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print("Outputting at time step %d..." %i)
        print("   Beam momentum %.6e eV" %beam.momentum)
        print("   Beam gamma %3.3f" %beam.gamma)
        print("   Beam beta %3.3f" %beam.beta)
        print("   Beam energy %.6e eV" %beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" %(4.*beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" %slice_beam.bl_gauss)
        print("")
        
    # Track
    for m in map_:
        m.track()
        
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params, beam)
    beam.losses_longitudinal_cut(0., 2.5e-9)
    
print("Done!")
print("")