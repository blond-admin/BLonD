# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of ion dynamics
No intensity effects

:Authors: **Alexandre Lasheen**
'''

from __future__ import division, print_function
from blond.utils.mpi_config import mpiprint, WORKER
from blond.utils import bmath as bm
from blond.trackers.tracker import RingAndRFTracker
from blond.plots.plot import Plot
from blond.monitors.monitors import BunchMonitor
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import CutOptions, Profile
from blond.beam.distributions import bigaussian
from blond.beam.beam import Beam, Particle
import numpy as np
import matplotlib as mpl
import os

from builtins import range

from scipy.constants import physical_constants

# Atomic Mass Unit [eV]
u = physical_constants['atomic mass unit-electron volt relationship'][0]


mpl.use('Agg')


bm.use_mpi()
print = mpiprint

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../mpi_output_files/EX_07_fig', exist_ok=True)


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 5.0e11                 # Design Intensity in SIS100
N_p = 50000                  # Macro-particles
tau_0 = 100.0e-9             # Initial bunch length, 4 sigma [s]
Z = 28.                      # Charge state of Uranium
m_p = 238.05078826 * u         # Isotope mass of U-238

# Machine and RF parameters
C = 1083.6                   # Machine circumference [m]
p_i = 153.37e9               # Synchronous momentum [eV/c]
p_f = 535.62e9               # Synchronous momentum, final 535.62e9
h = 10                       # Harmonic number
V = 280.e3                   # RF voltage [V]
dphi = np.pi                 # Phase modulation/offset
gamma_t = 15.59              # Transition gamma
alpha = 1. / gamma_t / gamma_t   # First order mom. comp. factor

# Tracking details
N_t = 45500                 # Number of turns to track
dt_plt = 5000                # Time steps between plots


# Simulation setup -------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters


general_params = Ring(C, alpha, np.linspace(p_i, p_f, N_t + 1),
                      Particle(m_p, Z), n_turns=N_t)

# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
print("Particle mass is %.3e eV" % general_params.Particle.mass)
print("Particle charge is %d e" % general_params.Particle.charge)

linspace_test = np.linspace(p_i, p_f, N_t + 1)
momentum_test = general_params.momentum
beta_test = general_params.beta
gamma_test = general_params.gamma
energy_test = general_params.energy
mass_test = general_params.Particle.mass  # [eV]
charge_test = general_params.Particle.charge  # e*Z

# Define RF station parameters and corresponding tracker
rf_params = RFStation(general_params, [h], [V], [dphi])
print("Initial bucket length is %.3e s" % (2. * np.pi / rf_params.omega_rf[0, 0]))
print("Final bucket length is %.3e s" % (2. * np.pi / rf_params.omega_rf[0, N_t]))

phi_s_test = rf_params.phi_s  # : *Synchronous phase
omega_RF_d_test = rf_params.omega_rf_d  # : *Design RF frequency of the RF systems in the station [GHz]*
omega_RF_test = rf_params.omega_rf  #: *Initial, actual RF frequency of the RF systems in the station [GHz]*
phi_RF_test = rf_params.omega_rf  # : *Initial, actual RF phase of each harmonic system*
E_increment_test = rf_params.delta_E  # Energy increment (acceleration/deceleration) between two turns,


long_tracker = RingAndRFTracker(rf_params, beam)

eta_0_test = rf_params.eta_0  # : *Slippage factor (0th order) for the given RF section*
eta_1_test = rf_params.eta_1  # : *Slippage factor (1st order) for the given RF section*
eta_2_test = rf_params.eta_2  # : *Slippage factor (2nd order) for the given RF section*
alpha_order_test = rf_params.alpha_order

bigaussian(general_params, rf_params, beam, tau_0 / 4,
           reinsertion='on', seed=1)


# Need slices for the Gaussian fit
slice_beam = Profile(beam, CutOptions(n_slices=100))

# Accelerator map
map_ = [long_tracker] + [slice_beam]

if WORKER.is_master:

    # Define what to save in file
    bunchmonitor = BunchMonitor(general_params, rf_params, beam,
                                this_directory + '../mpi_output_files/EX_07_output_data',
                                Profile=slice_beam)

    format_options = {'dirname': this_directory + '../mpi_output_files/EX_07_fig'}
    plots = Plot(general_params, rf_params, beam, dt_plt, N_t, 0, 8.e-7,
                 -400e6, 400e6, separatrix_plot=True, Profile=slice_beam,
                 h5file=this_directory + '../mpi_output_files/EX_07_output_data',
                 format_options=format_options)
    map_ += [bunchmonitor, plots]

    # For testing purposes
    test_string = ''
    test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
        'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
    test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
        np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))

print("Map set")
print("")


beam.split()
# Tracking ---------------------------------------------------------------------
for i in range(1, N_t + 1):

    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print("Outputting at time step %d..." % i)
        print("   Beam momentum %.6e eV" % beam.momentum)
        print("   Beam gamma %3.3f" % beam.gamma)
        print("   Beam beta %3.3f" % beam.beta)
        print("   Beam energy %.6e eV" % beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4. * beam.sigma_dt))
        print("")

    # Track
    for m in map_:
        m.track()

    # Define losses according to separatrix
    beam.losses_separatrix(general_params, rf_params)
    beam.gather_losses()

beam.gather()
WORKER.finalize()

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
with open(this_directory + '../mpi_output_files/EX_07_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
