
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

import os
from builtins import range

import matplotlib as mpl
import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian, parabolic
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import RingAndRFTracker
from blond.utils.assembler import Assembler

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_01_fig', exist_ok=True)


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
alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor

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


bigaussian(ring, rf, beam, tau_0 / 4, reinsertion=True, seed=1)
# parabolic(ring, rf, beam, tau_0, seed=1)


# Need slices for the Gaussian fit
profile = Profile(beam, CutOptions(n_slices=100),
                  FitOptions(fit_option='gaussian'))

# Define what to save in file
bunchmonitor = BunchMonitor(ring, rf, beam,
                            this_directory + '../output_files/EX_01_output_data', Profile=profile)

format_options = {'dirname': this_directory + '../output_files/EX_01_fig'}
plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763 * h,
             -400e6, 400e6, xunit='rad', separatrix_plot=True,
             Profile=profile, h5file=this_directory + '../output_files/EX_01_output_data',
             format_options=format_options)

class BeamLosses:
    def __init__(self, beam, ring, rf):
        self.beam = beam
        self.ring = ring
        self.rf = rf
    
    def track(self):
        # Define losses according to separatrix and/or longitudinal position
        self.beam.losses_separatrix(self.ring, self.rf)
        self.beam.losses_longitudinal_cut(0., 2.5e-9)

class OutputReporting:
    def __init__(self, beam, profile, dt_plt):
        self.beam = beam
        self.profile = profile
        self.track_period = dt_plt
        self.track_priority = -1
        self.i = 0

    def track(self):
        # Plot has to be done before tracking (at least for cases with separatrix)
        print("Outputting at time step %d..." % self.i)
        print("   Beam momentum %.6e eV" % self.beam.momentum)
        print("   Beam gamma %3.3f" % self.beam.gamma)
        print("   Beam beta %3.3f" % self.beam.beta)
        print("   Beam energy %.6e eV" % self.beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4. * self.beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" % self.profile.bunchLength)
        print("")
        self.i += self.track_period

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))

assembler = Assembler(element_list=[ring, beam, rf, long_tracker, profile,
                                    bunchmonitor, plots, 
                                    BeamLosses(beam, ring, rf),
                                    OutputReporting(beam, profile, dt_plt)])
assembler.build_pipeline()
print("Map set\n")

assembler.track(N_t, with_timing=True)
assembler.report_timing()

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
with open(this_directory + '../output_files/assembler_EX_01_test_data.txt', 'w') as f:
    f.write(test_string)

print("Done!")
