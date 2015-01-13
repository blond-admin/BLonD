
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulating the bunch splitting and rotation at PS flat top
Starting from tomoscope-reconstructed distribution
No intensity effects
25 ns LHC-type beam
'''

import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys

from plots.plot_settings import *
from input_parameters.preprocess import *
from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from beams.beams import *
from monitors.monitors import *
from plots.plot_beams import *


# Simulation parameters --------------------------------------------------------

# Bunch parameters
N_b = 1.e9                   # Intensity
N_p = 500000                 # Macro-particles

# Machine and RF parameters
C = 2.*np.pi*100.            # Machine circumference [m]
p_s = 25.92e9                # Synchronous momentum [eV]
h = [21, 42, 84, 168]        # Harmonic number
nh = 4                       # No. of harmonic RF systems
dphi = [np.pi, 0., 0., np.pi]# Phase offset between RF systems
gamma_tsq = 37.2             # Transition gamma
alpha = 1./gamma_tsq         # First order mom. comp. factor

# Tracking details
N_t = 128871                 # Number of turns to track

print "Setting up the simulation..."
print ""


# Simulation setup -------------------------------------------------------------

# Define general and RF parameters
general_params = GeneralParameters(N_t, C, alpha, p_s,'proton')
print "General parameters set..."

# Initialize plots
PlotSettings().set_plot_format()
plot_time = [2.190, 2.255, 2.320, 2.403, 2.455, 2.460, 2.4601, 2.46015, 2.46027]
plot_time = (plot_time - 2.190*np.ones(len(plot_time)))/general_params.t_rev[0] 
plot_time = [int(i) for i in plot_time]
print "Plotting at below time steps:"
print plot_time

# Interpolate voltage programmes
t1, v21 = loaddata('../input_files/TC8_PS_h21.dat', ignore=1)

# Alternative way to read data, e.g. if you want to group columns
#data = loaddata('../input_files/TC8_PS_h21.dat', ignore=1)
#t1 = data[0]
#v21 = data[1]

# Using "loaddata" makes sure that the arrays are c-contiguous and numpy arrays
print v21.flags['C_CONTIGUOUS']
print type(v21)
V1 = preprocess_rf_params(general_params, t1, v21, figname='Voltage_PS_h21', figdir='../output_files/TC8_fig')

t1, v42 = loaddata('../input_files/TC8_PS_h42.dat', ignore=1)
V2 = preprocess_rf_params(general_params, t1, v42, figname='Voltage_PS_h42', figdir='../output_files/TC8_fig')
print v42.flags['C_CONTIGUOUS']

t1, v84 = loaddata('../input_files/TC8_PS_h84.dat', ignore=1)
V3 = preprocess_rf_params(general_params, t1, v84, figname='Voltage_PS_h84', figdir='../output_files/TC8_fig')
print v84.flags['C_CONTIGUOUS']

t1, v168 = loaddata('../input_files/TC8_PS_h168.dat', ignore=1)
V4 = preprocess_rf_params(general_params, t1, v168, figname='Voltage_PS_h168', figdir='../output_files/TC8_fig')
V = [V1[0], V2[0], V3[0], V4[0]]
print "Voltage programme set..."

rf_params = RFSectionParameters(general_params, nh, h, V, dphi)
tracker = RingAndRFSection(rf_params)
print "RF parameters and tracker set..."

# Define beam and distribution: load tomoscope-reconstructed distribution
beam = Beam(general_params, N_p, N_b)
beam.theta, beam.dE = loaddata('../input_files/TC8_6_1_c45.dst', ignore=1)
beam.dE *= 1.e6 # to convert from MeV

# Plot initial distribution, set phi_s by hand
rf_params.phi_s = np.pi*np.ones(N_t+1) 
plot_long_phase_space(beam, general_params, rf_params, -0.15, 0.15, -50, 50, dirname = '../output_files/TC8_fig')
print "Bunch distribution loaded and plotted..."

# Define what to save in file
bunchmonitor = BunchMonitor('../output_files/TC8_output_data', N_t+1)
bunchmonitor.track(beam)
print "Statistics set..."

# Accelerator map
map_ = [tracker] + [bunchmonitor] # No intensity effects, no aperture limitations
print "Map set"
print ""

print "Phis = %.4e rad" %rf_params.phi_s[0]
print "Slippage = %.4e" %rf_params.eta_0[0]

print rf_params.voltage[0]
print rf_params.voltage[1]
print rf_params.voltage[2]
print rf_params.voltage[3]



# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    print i
    t0 = time.clock()
    
   
    # Define losses to extract one splitted bunch
    if i == N_t:   
        beam.losses_longitudinal_cut(0., 0.0748)

    # Plot has to be done before tracking (at least for cases with separatrix)
    if i in plot_time:
        print "Outputting at time step %d..." %i
        print "   Beam momentum %.6e eV" %beam.momentum
        print "   Beam gamma %3.3f" %beam.gamma_r
        print "   Beam beta %3.3f" %beam.beta_r
        print "   Beam energy %.6e eV" %beam.energy
        print "   RF voltage at h21: %.2e kV, h42: %.2e kV, h84: %.2e kV, h168: %.2e kV" \
            %(rf_params.voltage[0,i]*1.e-3, rf_params.voltage[1,i]*1.e-3, 
              rf_params.voltage[2,i]*1.e-3, rf_params.voltage[3,i]*1.e-3)
        print "   Four-times r.m.s. bunch length %.4e rad" %(4.*beam.sigma_theta)
        print ""
        # In plots, you can choose following units: rad, ns, m  
        plot_long_phase_space(beam, general_params, rf_params, -0.15, 0.15, -50, 50, dirname = '../output_files/TC8_fig')
        np.savetxt('../output_files/TC8_coords_' "%d" %rf_params.counter[0] + '.dat', 
                   np.c_[beam.theta, beam.dE, beam.id], fmt='%.10e')

    # Track
    for m in map_:
        m.track(beam)
        
# Final data processing
plot_bunch_length_evol('../output_files/TC8_output_data', general_params, i, unit='ns', dirname = '../output_files/TC8_fig')
plot_long_phase_space(beam, general_params, rf_params, -0.15, 0.15, -50, 50, dirname = '../output_files/TC8_fig')
plot_long_phase_space(beam, general_params, rf_params, 0., 0.08, -80, 80, dirname = '../output_files/TC8_fig')
np.savetxt('../output_files/TC8_coords_' "%d" %rf_params.counter[0] + '.dat', 
           np.c_[beam.theta, beam.dE, beam.id], fmt='%.10e')


bunchmonitor.h5file.close()

print "Done!"
print ""
