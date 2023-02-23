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
import blond.utils.bmath as bm
from blond.utils import input_parser
from blond.plots.plot import Plot
from blond.monitors.monitors import BunchMonitor
from blond.beam.profile import CutOptions, Profile, FitOptions
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Proton
from blond.trackers.utilities import total_voltage
from blond.trackers.tracker import RingAndRFTracker
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

USE_GPU = os.environ.get('USE_GPU', '0')
if len(USE_GPU) and int(USE_GPU):
    USE_GPU = True
else:
    USE_GPU = False

os.makedirs(this_directory + '../gpu_output_files/EX_04_fig', exist_ok=True)


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
                      Proton(), N_t, n_sections=2)


# Define RF station parameters and corresponding tracker
beam = Beam(general_params, N_p, N_b)
rf_params_1 = RFStation(general_params, [h], [V1], [dphi],
                        section_index=1)
long_tracker_1 = RingAndRFTracker(rf_params_1, beam)

rf_params_2 = RFStation(general_params, [h], [V2], [dphi],
                        section_index=2)
long_tracker_2 = RingAndRFTracker(rf_params_2, beam)

# Define full voltage over one turn and a corresponding "overall" set of
# parameters, which is used for the separatrix (in plotting and losses)
Vtot = total_voltage([rf_params_1, rf_params_2])
rf_params_tot = RFStation(general_params, [h], [Vtot], [dphi])
beam_dummy = Beam(general_params, 1, N_b)
long_tracker_tot = RingAndRFTracker(rf_params_tot, beam_dummy)

print("General and RF parameters set...")


# Define beam and distribution

bigaussian(general_params, rf_params_tot, beam, tau_0/4,
           reinsertion='on', seed=1)

print("Beam set and distribution generated...")


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Profile(beam, CutOptions(n_slices=100),
                     FitOptions(fit_option='gaussian'))
# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params_tot, beam,
                            this_directory + '../gpu_output_files/EX_04_output_data',
                            Profile=slice_beam, buffer_time=1)

# PLOTS
format_options = {'dirname': this_directory +
                  '../gpu_output_files/EX_04_fig', 'linestyle': '.'}
plots = Plot(general_params, rf_params_tot, beam, dt_plt, dt_plt, 0,
             0.0001763*h, -450e6, 450e6, xunit='rad',
             separatrix_plot=True, Profile=slice_beam,
             h5file=this_directory + '../gpu_output_files/EX_04_output_data',
             histograms_plot=True, format_options=format_options)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(), beam.dt.mean(), beam.dt.std())

# Accelerator map
map_ = [long_tracker_1] + [long_tracker_2] + \
    [slice_beam]  # + [bunchmonitor] + \
# [plots]
print("Map set")
print("")

if USE_GPU:
    bm.use_gpu()
    beam.to_gpu()
    long_tracker_1.to_gpu()
    long_tracker_2.to_gpu()
    long_tracker_tot.to_gpu()
    rf_params_1.to_gpu()
    rf_params_2.to_gpu()
    rf_params_tot.to_gpu()
    slice_beam.to_gpu()
    beam_dummy.to_gpu()


# Tracking --------------------------------------------------------------------
for i in np.arange(1, N_t+1):
    # print(i)

    if (i % dt_plt) == 0:

        if USE_GPU:
            # Copy to CPU all needed data
            bm.use_cpu()
            beam.to_cpu()
            long_tracker_1.to_cpu()
            long_tracker_2.to_cpu()
            long_tracker_tot.to_cpu()
            rf_params_1.to_cpu()
            rf_params_2.to_cpu()
            rf_params_tot.to_cpu()
            slice_beam.to_cpu()
            beam_dummy.to_cpu()

        plots.track()

        if USE_GPU:
            # copy back to GPU
            bm.use_gpu()
            beam.to_gpu()
            long_tracker_1.to_gpu()
            long_tracker_2.to_gpu()
            long_tracker_tot.to_gpu()
            rf_params_1.to_gpu()
            rf_params_2.to_gpu()
            rf_params_tot.to_gpu()
            slice_beam.to_gpu()
            beam_dummy.to_gpu()

    long_tracker_tot.track()

    # Track
    for m in map_:
        m.track()

    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params_tot)
    beam.losses_longitudinal_cut(0., 2.5e-9)


if USE_GPU:
    bm.use_cpu()
    beam.to_cpu()
    long_tracker_1.to_cpu()
    long_tracker_2.to_cpu()
    long_tracker_tot.to_cpu()
    rf_params_1.to_cpu()
    rf_params_2.to_cpu()
    rf_params_tot.to_cpu()
    slice_beam.to_cpu()
    beam_dummy.to_cpu()

print('dE mean: ', beam.dE.mean())
print('dE std: ', beam.dE.std())
print('profile mean: ', slice_beam.n_macroparticles.mean())
print('profile std: ', slice_beam.n_macroparticles.std())

# # For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    beam.dE.mean(), beam.dE.std(), beam.dt.mean(), beam.dt.std())
with open(this_directory + '../gpu_output_files/EX_04_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
