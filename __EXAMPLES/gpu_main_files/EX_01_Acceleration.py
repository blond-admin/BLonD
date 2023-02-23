
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects

:Authors: **Helga Timko**
'''
#  General Imports
from __future__ import division, print_function
import blond.utils.bmath as bm
from blond.plots.plot import Plot
from blond.monitors.monitors import BunchMonitor
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
from blond.trackers.tracker import RingAndRFTracker
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from builtins import range
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')


#  BLonD Imports


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

USE_GPU = os.environ.get('USE_GPU', '0')
if len(USE_GPU) and int(USE_GPU):
    USE_GPU = True
else:
    USE_GPU = False

os.makedirs(this_directory + '../gpu_output_files/EX_01_fig', exist_ok=True)


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
ring = Ring(C, alpha, np.linspace(p_i, p_f, 2001), Proton(), N_t)

# Define beam and distribution
beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])
long_tracker = RingAndRFTracker(rf, beam)


bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=1)


# Need slices for the Gaussian fit
profile = Profile(beam, CutOptions(n_slices=100),
                  FitOptions(fit_option='gaussian')
                  )

# Define what to save in file
bunchmonitor = BunchMonitor(ring, rf, beam,
                            this_directory + '../gpu_output_files/EX_01_output_data', 
                            Profile=profile)

format_options = {'dirname': this_directory + '../gpu_output_files/EX_01_fig'}
plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
             -400e6, 400e6, xunit='rad', separatrix_plot=True,
             Profile=profile, h5file=this_directory + '../gpu_output_files/EX_01_output_data',
             format_options=format_options)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(), beam.dt.mean(), beam.dt.std())


# For the GPU version we disable bunchmonitor and plots, since they
# make the simulation much slower

# Accelerator map
map_ = [long_tracker] + [profile] + [plots]
# + [bunchmonitor] + [plots]
print("Map set")
print("")

# This is the way to enable the GPU
if USE_GPU:
    bm.use_gpu()
    rf.to_gpu()
    beam.to_gpu()
    long_tracker.to_gpu()
    profile.to_gpu()


# Tracking --------------------------------------------------------------------
for i in range(1, N_t+1):


    # Plot has to be done before tracking (at least for cases with separatrix)
    if i % dt_plt == 0:
        print("Outputting at time step %d..." % i)
        print("   Beam momentum %.6e eV" % beam.momentum)
        print("   Beam gamma %3.3f" % beam.gamma)
        print("   Beam beta %3.3f" % beam.beta)
        print("   Beam energy %.6e eV" % beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" % profile.bunchLength)
        print("")

    long_tracker.track()
    profile.track()

    if i % dt_plt == 0:
        if USE_GPU:
            # Copy to CPU all needed data
            bm.use_cpu()
            beam.to_cpu()
            profile.to_cpu()
            rf.to_cpu()

        plots.track()
        
        if USE_GPU:
            # copy back to GPU
            bm.use_gpu()
            beam.to_gpu()
            profile.to_gpu()
            rf.to_gpu()
    
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(ring, rf)
    beam.losses_longitudinal_cut(0., 2.5e-9)

if USE_GPU:
    bm.use_cpu()
    rf.to_cpu()
    beam.to_cpu()
    long_tracker.to_cpu()
    profile.to_cpu()

print('dE mean: ', beam.dE.mean())
print('dE std: ', beam.dE.std())
print('profile mean: ', profile.n_macroparticles.mean())
print('profile std: ', profile.n_macroparticles.std())


# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(), beam.dt.mean(), beam.dt.std())
with open(this_directory + '../gpu_output_files/EX_01_test_data.txt', 'w') as f:
    f.write(test_string)

print("Done!")
