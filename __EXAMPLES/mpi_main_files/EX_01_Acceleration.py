
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
from builtins import range
import numpy as np

#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
import os
import matplotlib as mpl
mpl.use('Agg')

from blond.utils import bmath as bm
from blond.utils.mpi_config import worker, mpiprint
bm.use_mpi()

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../mpi_output_files/EX_01_fig', exist_ok=True)


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
mpiprint("Setting up the simulation...\n")


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
                  FitOptions(fit_option='gaussian'))


# Accelerator map
map_ = [long_tracker] + [profile]

# Define what to save in file
if worker.isMaster:
    bunchmonitor = BunchMonitor(ring, rf, beam,
                                this_directory + '../mpi_output_files/EX_01_output_data',
                                Profile=profile)
    format_options = {'dirname': this_directory +
                      '../mpi_output_files/EX_01_fig'}
    plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
                 -400e6, 400e6, xunit='rad', separatrix_plot=True,
                 Profile=profile, h5file=this_directory + '../mpi_output_files/EX_01_output_data',
                 format_options=format_options)

    map_ += [bunchmonitor] + [plots]

    # For testing purposes
    test_string = ''
    test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
        'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
    test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
        np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))

mpiprint("Map set\n")
beam.split()

# Tracking --------------------------------------------------------------------
for i in range(1, N_t+1):

    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        mpiprint("Outputting at time step %d..." % i)
        mpiprint("   Beam momentum %.6e eV" % beam.momentum)
        mpiprint("   Beam gamma %3.3f" % beam.gamma)
        mpiprint("   Beam beta %3.3f" % beam.beta)
        mpiprint("   Beam energy %.6e eV" % beam.energy)
        mpiprint("   Four-times r.m.s. bunch length %.4e s" %
                 (4.*beam.sigma_dt))
        mpiprint("   Gaussian bunch length %.4e s" % profile.bunchLength)

    # Track
    for m in map_:
        m.track()

    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(ring, rf)
    beam.losses_longitudinal_cut(0., 2.5e-9)
    beam.gather_losses()

beam.gather()
worker.finalize()

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
with open(this_directory + '../mpi_output_files/EX_01_test_data.txt', 'w') as f:
    f.write(test_string)

print("Done!")
