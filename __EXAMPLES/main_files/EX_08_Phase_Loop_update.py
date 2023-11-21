
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case to show how to use phase loop (CERN PS Booster context).

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
from blond.llrf.beam_feedback_update_subclasses import BeamFeedback_PSB
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_08_fig_updt', exist_ok=True)


# Beam parameters
n_macroparticles = 100000
n_particles = 0

# Machine and RF parameters
radius = 25  # [m]
gamma_transition = 4.076750841
alpha = 1 / gamma_transition**2
C = 2 * np.pi * radius  # [m]
n_turns = 500
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


cut_options = CutOptions(cut_left=0, cut_right=2 * np.pi, n_slices=200,
                         RFSectionParameters=rf_params, cuts_unit='rad')
slices_ring = Profile(my_beam, cut_options)

# Phase loop
phase_loop = BeamFeedback_PSB(general_params, rf_params, slices_ring, RL_gain=[0,0])

# Long tracker
long_tracker = RingAndRFTracker(rf_params, my_beam, periodicity=False,
                                BeamFeedback=phase_loop
                                )

full_ring = FullRingAndRF([long_tracker])


distribution_type = 'gaussian'
bunch_length = 200.0e-9
distribution_variable = 'Action'

matched_from_distribution_function(my_beam, full_ring,
                                   bunch_length=bunch_length,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1222)

my_beam.dE += 90.0e3
slices_ring.track()

# Monitor
bunch_monitor = BunchMonitor(general_params, rf_params, my_beam,
                             this_directory + '../output_files/EX_08_updt_output_data',
                             Profile=slices_ring, PhaseLoop=phase_loop)


# Plots
format_options = {'dirname': this_directory + '../output_files/EX_08_updt_fig'}
plots = Plot(general_params, rf_params, my_beam, 50, n_turns, 0.0, 2 * np.pi,
             -1e6, 1e6, xunit='rad', separatrix_plot=True, Profile=slices_ring,
             format_options=format_options,
             h5file=this_directory + '../output_files/EX_08_updt_output_data', PhaseLoop=phase_loop)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))


# Accelerator map
map_ = [full_ring] + [slices_ring] + [bunch_monitor] + [plots]


for i in range(1, n_turns + 1):
    print(i)

    for m in map_:
        m.track()

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))
with open(this_directory + '../output_files/EX_08_updt_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
