
# Copyright 2016 CERN. This software is distributed under the
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

from __future__ import division, print_function
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
from plots.plot import *

# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V1 = 2e6           # RF voltage, station 1 [eV]
V2 = 4e6           # RF voltage, station 1 [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track
dt_plt = 200         # Time steps between plots



# Simulation setup ------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters containing data for both RF stations
general_params = GeneralParameters(N_t, [0.3*C, 0.7*C], [[alpha], [alpha]], 
                                   [p_s*np.ones(N_t+1), p_s*np.ones(N_t+1)], 
                                   'proton', number_of_sections = 2)


# Define RF station parameters and corresponding tracker
beam = Beam(general_params, N_p, N_b)
rf_params_1 = RFSectionParameters(general_params, 1, h, V1, dphi,
                                  section_index=1)
long_tracker_1 = RingAndRFSection(rf_params_1, beam)

rf_params_2 = RFSectionParameters(general_params, 1, h, V2, dphi,
                                  section_index=2)
long_tracker_2 = RingAndRFSection(rf_params_2, beam)

# Define full voltage over one turn and a corresponding "overall" set of 
#parameters, which is used for the separatrix (in plotting and losses)
Vtot = total_voltage([rf_params_1, rf_params_2])
rf_params_tot = RFSectionParameters(general_params, 1, h, Vtot, dphi)
beam_dummy = Beam(general_params, 1, N_b)
long_tracker_tot = RingAndRFSection(rf_params_tot, beam_dummy)

print("General and RF parameters set...")


# Define beam and distribution

longitudinal_bigaussian(general_params, rf_params_tot, beam, tau_0/4, 
                              reinsertion = 'on', seed=1)

print("Beam set and distribution generated...")


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Slices(rf_params_tot, beam, 100, fit_option='gaussian')

# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params_tot, beam,
                            '../output_files/TC4_output_data',
                            Slices=slice_beam, buffer_time=1)

# PLOTS
format_options = {'dirname': '../output_files/TC4_fig', 'linestyle': '.'}
plots = Plot(general_params, rf_params_tot, beam, dt_plt, dt_plt, 0, 
             0.0001763*h, -450e6, 450e6, xunit='rad',
             separatrix_plot=True, Slices=slice_beam,
             h5file='../output_files/TC4_output_data',
             histograms_plot=True, format_options=format_options)


# Accelerator map
map_ = [long_tracker_1] + [long_tracker_2] + [slice_beam] + [bunchmonitor] + \
       [plots]
print("Map set")
print("")

# Tracking --------------------------------------------------------------------
for i in np.arange(1,N_t+1):
    print(i)
    
    long_tracker_tot.track() 
       
    # Track
    for m in map_:
        m.track()
    
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params_tot, beam)
    beam.losses_longitudinal_cut(0., 2.5e-9)


print("Done!")
print("")
