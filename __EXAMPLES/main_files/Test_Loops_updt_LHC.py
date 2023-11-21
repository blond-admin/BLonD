
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case to show how to use phase loop (CERN SPS context).

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

os.makedirs(this_directory + '../output_files/Test_Loops_updt_fig', exist_ok=True)


# Bunch parameters
N_b = 1e9           # Intensity
N_p = 50000         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
p_f = 450.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
n_turns = 500       # Number of turns to track


# Simulation setup ------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, n_turns+1), Proton(), n_turns)

# Define beam and distribution
beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf_params = RFStation(ring, [h], [V], [dphi])
cut_options = CutOptions(cut_left=0, cut_right=2*np.pi, n_slices=200,
                         RFSectionParameters=rf_params, cuts_unit='rad')
slices_ring = Profile(beam, cut_options)

# Phase loop
configuration = {'machine': 'LHC', 'PL_gain': 0.01, 'RL_gain': 0.01}
phase_loop = BeamFeedback(ring, rf_params, slices_ring, configuration)


# Long tracker
long_tracker = RingAndRFTracker(rf_params, beam, periodicity=False,
                                BeamFeedback=phase_loop
                                )

full_ring = FullRingAndRF([long_tracker])


distribution_type = 'gaussian'
bunch_length = 0.4e-9
distribution_variable = 'Action'

matched_from_distribution_function(beam, full_ring,
                                   bunch_length=bunch_length,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1222)

# my_beam.dE += 5e6
slices_ring.track()

# Monitor
bunch_monitor = BunchMonitor(ring, rf_params, beam,
                             this_directory + '../output_files/Test_Loops_updt_output_data',
                             Profile=slices_ring, PhaseLoop=phase_loop)


# Plots
format_options = {'dirname': this_directory + '../output_files/Test_Loops_updt_fig'}
plots = Plot(ring, rf_params, beam, 50, n_turns, 0.0, 2 * np.pi,
             -400e6, 400e6, xunit='rad', separatrix_plot=True, Profile=slices_ring,
             format_options=format_options,
             h5file=this_directory + '../output_files/Test_Loops_updt_output_data', PhaseLoop=phase_loop)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))


# Accelerator map
map_ = [full_ring] + [slices_ring] + [bunch_monitor] + [plots]


for i in range(1, n_turns + 1):
    print(i)

    for m in map_:
        m.track()
    print("domega from loop:", phase_loop.domega_rf)

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
with open(this_directory + '../output_files/Test_Loops_updt_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
