
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case to show how to use radial loop (CERN PS Booster context).

:Authors: **Danilo Quartullo**
'''

from __future__ import division, print_function

import os

import matplotlib as mpl
import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.beam_feedback import BeamFeedback
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_9_fig', exist_ok=True)


# Beam parameters
n_macroparticles = 100000
n_particles = 0


# Machine and RF parameters
radius = 25  # [m]
gamma_transition = 4.076750841
alpha = 1 / gamma_transition**2
C = 2 * np.pi * radius  # [m]

n_turns = 2000

general_params = Ring(C, alpha, 310891054.809,
                      Proton(), n_turns)

# Cavities parameters
n_rf_systems = 1
harmonic_numbers_1 = 1
voltage_1 = 8000  # [V]
phi_offset_1 = np.pi   # [rad]
rf_params = RFStation(general_params, [harmonic_numbers_1], [voltage_1],
                      [phi_offset_1], n_rf_systems)

my_beam = Beam(general_params, n_macroparticles, n_particles)


cut_options = CutOptions(cut_left=0, cut_right=2.0 * 0.9e-6, n_slices=200)
slices_ring = Profile(my_beam, cut_options)

# Phase loop
# configuration = {'machine': 'PSB', 'PL_gain': 0., 'RL_gain': [34.8,16391],
#                 'PL_period': 10.e-6, 'RL_period': 7}
configuration = {'machine': 'PSB', 'PL_gain': 0,
                 'RL_gain': [1.e7, 1.e11], 'period': 10.e-6}
phase_loop = BeamFeedback(general_params, rf_params, slices_ring, configuration)


# Long tracker
long_tracker = RingAndRFTracker(rf_params, my_beam,
                                BeamFeedback=phase_loop)

full_ring = FullRingAndRF([long_tracker])


distribution_type = 'gaussian'
bunch_length = 200.0e-9
distribution_variable = 'Action'

matched_from_distribution_function(my_beam, full_ring,
                                   bunch_length=bunch_length,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1223)

slices_ring.track()

# Monitor
bunch_monitor = BunchMonitor(general_params, rf_params, my_beam,
                             this_directory + '../output_files/EX_09_output_data',
                             Profile=slices_ring, PhaseLoop=phase_loop)


# Plots
format_options = {'dirname': this_directory + '../output_files/EX_09_fig'}
plots = Plot(general_params, rf_params, my_beam, 100, n_turns, 0.0, 2.0 * 0.9e-6,
             -1.e6, 1.e6, separatrix_plot=True, Profile=slices_ring,
             format_options=format_options,
             h5file=this_directory + '../output_files/EX_09_output_data', PhaseLoop=phase_loop)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))

# Accelerator map
map_ = [long_tracker] + [slices_ring] + [bunch_monitor] + [plots]

for i in range(1, n_turns + 1):

    if i == 100:
        phase_loop.reference += 0.00001  # /1.6e-6

    for m in map_:
        m.track()
    slices_ring.cut_options.track_cuts(my_beam)
    slices_ring.set_slices_parameters()

    if (i % 50 == 0):
        print("Time step %d" % i)
        print("    Radial error %.4e" % (phase_loop.drho))
        print("    Radial loop frequency correction %.4e 1/s"
              % (phase_loop.domega_rf))
        print("    RF phase %.4f rad" % (rf_params.phi_rf[0, i]))
        print("    RF frequency %.6e 1/s" % (rf_params.omega_rf[0, i]))

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))
with open(this_directory + '../output_files/EX_09_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
