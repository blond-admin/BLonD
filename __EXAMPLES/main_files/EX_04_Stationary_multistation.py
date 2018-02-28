# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulating a ring with multiple RF stations
No intensity effects

:Authors: **Helga Timko**
'''

from __future__ import division, print_function
import numpy as np
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from trackers.tracker import RingAndRFTracker
from trackers.utilities import total_voltage
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import CutOptions, Profile, FitOptions
from monitors.monitors import BunchMonitor
from plots.plot import Plot
import os

try:
    os.mkdir('../output_files')
except:
    pass
try:
    os.mkdir('../output_files/EX_04_fig')
except:
    pass

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
general_params = Ring([0.3*C, 0.7*C], [[alpha], [alpha]], 
                                   [p_s*np.ones(N_t+1), p_s*np.ones(N_t+1)], 
                                   Proton(), N_t, n_sections = 2)


# Define RF station parameters and corresponding tracker
beam = Beam(general_params, N_p, N_b)
rf_params_1 = RFStation(general_params, [h], [V1], [dphi],
                                  section_index=1)
long_tracker_1 = RingAndRFTracker(rf_params_1, beam)

rf_params_2 = RFStation(general_params, [h], [V2], [dphi],
                                  section_index=2)
long_tracker_2 = RingAndRFTracker(rf_params_2, beam)

# Define full voltage over one turn and a corresponding "overall" set of 
#parameters, which is used for the separatrix (in plotting and losses)
Vtot = total_voltage([rf_params_1, rf_params_2])
rf_params_tot = RFStation(general_params, [h], [Vtot], [dphi])
beam_dummy = Beam(general_params, 1, N_b)
long_tracker_tot = RingAndRFTracker(rf_params_tot, beam_dummy)

print("General and RF parameters set...")


# Define beam and distribution

bigaussian(general_params, rf_params_tot, beam, tau_0/4, 
                              reinsertion = 'on', seed=1)

print("Beam set and distribution generated...")


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Profile(beam, CutOptions(n_slices=100),
                 FitOptions(fit_option='gaussian'))       
# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params_tot, beam,
                            '../output_files/EX_04_output_data',
                            Profile=slice_beam, buffer_time=1)

# PLOTS
format_options = {'dirname': '../output_files/EX_04_fig', 'linestyle': '.'}
plots = Plot(general_params, rf_params_tot, beam, dt_plt, dt_plt, 0, 
             0.0001763*h, -450e6, 450e6, xunit='rad',
             separatrix_plot=True, Profile=slice_beam,
             h5file='../output_files/EX_04_output_data',
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
    beam.losses_separatrix(general_params, rf_params_tot)
    beam.losses_longitudinal_cut(0., 2.5e-9)


print("Done!")
