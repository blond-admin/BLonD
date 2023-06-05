
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case for the synchrotron radiation routine.
Example for the FCC-ee at 175 GeV.

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division
from scipy.optimize import curve_fit

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_e

from blond.beam.beam import Beam, Electron
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.synchrotron_radiation.synchrotron_radiation import \
    SynchrotronRadiation
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_13_fig/', exist_ok=True)


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = Electron()
n_particles = int(1.7e11)
n_macroparticles = int(1e5)
sync_momentum = 175e9  # [eV]


distribution_type = 'gaussian'
emittance = 1.0
distribution_variable = 'Action'

# Machine and RF parameters
radius = 15915.49
gamma_transition = 377.96447
C = 2 * np.pi * radius  # [m]

# Tracking details
n_turns = int(200)
n_turns_between_two_plots = 100

# Derived parameters
E_0 = m_e * c**2 / e    # [eV]
tot_beam_energy = np.sqrt(sync_momentum**2 + E_0**2)  # [eV]
momentum_compaction = 1 / gamma_transition**2  # [1]

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 133650
voltage_program = 10e9
phi_offset = np.pi

bucket_length = C / c / harmonic_numbers

# DEFINE RING------------------------------------------------------------------

n_sections = 2
general_params = Ring(np.ones(n_sections) * C / n_sections,
                      np.tile(momentum_compaction, (1, n_sections)).T,
                      np.tile(sync_momentum, (n_sections, n_turns + 1)),
                      particle_type, n_turns, n_sections=n_sections)

RF_sct_par = []
for i in np.arange(n_sections) + 1:
    RF_sct_par.append(RFStation(general_params,
                                [harmonic_numbers], [voltage_program / n_sections],
                                [phi_offset], n_rf_systems, section_index=i))

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)

# DEFINE SLICES----------------------------------------------------------------

number_slices = 500

cut_options = CutOptions(cut_left=0., cut_right=bucket_length, n_slices=number_slices)
slice_beam = Profile(beam, CutOptions=cut_options)

# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = []
for i in range(n_sections):
    longitudinal_tracker.append(RingAndRFTracker(RF_sct_par[i], beam, Profile=slice_beam))

full_tracker = FullRingAndRF(longitudinal_tracker)


# BEAM GENERATION--------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker, emittance=emittance,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1000)

slice_beam.track()

# Synchrotron radiation objects without quantum excitation
rho = 11e3
SR = []
for i in range(n_sections):
    SR.append(SynchrotronRadiation(general_params, RF_sct_par[i], beam, rho,
                                   quantum_excitation=False, python=True))

SR[0].print_SR_params()

# ACCELERATION MAP-------------------------------------------------------------

map_ = []
for i in range(n_sections):
    map_ += [longitudinal_tracker[i]] + [SR[i]]
map_ += [slice_beam]

# TRACKING + PLOTS-------------------------------------------------------------

avg_dt = np.zeros(n_turns)
std_dt = np.zeros(n_turns)

for i in range(n_turns):
    for m in map_:
        m.track()

    avg_dt[i] = np.mean(beam.dt)
    std_dt[i] = np.std(beam.dt)

# Fitting routines for synchrotron radiation damping


def sine_exp_fit(x, y, **keywords):
    try:
        init_values = keywords['init_values']
        offset = init_values[-1]
        init_values[-1] = 0
    except Exception:
        offset = np.mean(y)
        # omega estimation using FFT
        npoints = 12
        y_fft = np.fft.fft(y - offset, 2**npoints)
        omega_osc = (2.0 * np.pi * np.abs(y_fft[:2**(npoints - 1)]).argmax() /
                     len(y_fft) / (x[1] - x[0]))
        init_amp = (y.max() - y.min()) / 2.0
        init_omega = omega_osc
        init_values = [init_omega, 0, init_amp, 0, 0]

    popt, pcov = curve_fit(sine_exp_f, x, y - offset, p0=init_values)

    popt[0] = np.abs(popt[0])
    popt[2] = np.abs(popt[2])
    popt[3] += offset
    if np.isinf(pcov).any():
        pcov = np.zeros([5, 5])

    return popt, pcov


def sine_exp_f(x, omega, phi, amp, offset, tau):
    return offset + np.abs(amp) * np.sin(omega * x + phi) * np.exp(tau * x)


def exp_f(x, amp, offset, tau):
    return offset + np.abs(amp) * np.exp(-np.abs(tau) * x)


# Fit of the bunch length
plt.figure(figsize=[6, 4.5])
plt.plot(1e12 * 4.0 * std_dt, lw=2)
a, b = popt, pcov = curve_fit(exp_f, np.arange(len(std_dt)), 4.0 * std_dt)
amp, offset, tau = a[0], a[1], a[2]
plt.plot(np.arange(len(std_dt)), 1e12 * exp_f(np.arange(len(std_dt)), amp,
                                              offset, np.abs(tau)), 'r--', lw=2, alpha=0.75)
plt.ylim(0, plt.ylim()[1])
plt.xlabel('Turns')
plt.ylabel('Bunch length [ps]')
plt.legend(('Simulation', 'Damping time: {0:1.1f} turns (fit)'.format(1 /
           np.abs(tau))), loc=0, fontsize='medium')
plt.savefig(this_directory + '../output_files/EX_13_fig/bl_fit.png')
plt.close()


# Fit of the bunch position
a, b = sine_exp_fit(np.arange(len(avg_dt)), avg_dt)
omega, phi, amp, offset, tau = a[0], a[1], a[2], a[3], a[4]

plt.figure(figsize=[6, 4.5])
plt.plot(avg_dt * 1e9, lw=2)
plt.plot(np.arange(len(avg_dt)), sine_exp_f(np.arange(len(avg_dt)), omega, phi,
                                            amp, offset, tau) * 1e9, 'r--', lw=2, alpha=0.75)
plt.xlabel('Turns')
plt.ylabel('Bunch position [ns]')
plt.legend(('Simulation', 'Damping time: {0:1.1f} turns (fit)'.format(1 /
            np.abs(tau))), loc=0, fontsize='medium')
plt.savefig(this_directory + '../output_files/EX_13_fig/pos_fit')
plt.close()

# WITH QUANTUM EXCITATION
n_turns = 200
# DEFINE RING------------------------------------------------------------------

n_sections = 10
general_params = Ring(np.ones(n_sections) * C / n_sections,
                      np.tile(momentum_compaction, (1, n_sections)).T,
                      np.tile(sync_momentum, (n_sections, n_turns + 1)),
                      particle_type, n_turns, n_sections=n_sections)

RF_sct_par = []
for i in np.arange(n_sections) + 1:
    RF_sct_par.append(RFStation(general_params,
                                [harmonic_numbers], [voltage_program / n_sections],
                                [phi_offset], n_rf_systems, section_index=i))

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)


# DEFINE SLICES----------------------------------------------------------------

cut_options = CutOptions(cut_left=0., cut_right=bucket_length, n_slices=number_slices)
slice_beam = Profile(beam, CutOptions=cut_options)

# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = []
for i in range(n_sections):
    longitudinal_tracker.append(RingAndRFTracker(RF_sct_par[i], beam, Profile=slice_beam))

full_tracker = FullRingAndRF(longitudinal_tracker)


# BEAM GENERATION--------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker, emittance=emittance,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1000)

slice_beam.track()

# Redefine Synchrotron radiation objects with quantum excitation
SR = []
for i in range(n_sections):
    SR.append(SynchrotronRadiation(general_params, RF_sct_par[i], beam, rho, python=False, seed=7))

# ACCELERATION MAP-------------------------------------------------------------
map_ = []
for i in range(n_sections):
    map_ += [longitudinal_tracker[i]] + [SR[i]]
map_ += [slice_beam]

# TRACKING + PLOTS-------------------------------------------------------------

std_dt = np.zeros(n_turns)
std_dE = np.zeros(n_turns)

for i in range(n_turns):
    for m in map_:
        m.track()

    std_dt[i] = np.std(beam.dt)
    std_dE[i] = np.std(beam.dE)

plt.figure(figsize=[6, 4.5])
plt.plot(1e-6 * std_dE, lw=2)
plt.plot(np.arange(len(std_dE)), [1e-6 * SR[0].sigma_dE * sync_momentum] *
         len(std_dE), 'r--', lw=2)
print('Equilibrium energy spread = {0:1.3f} [MeV]'.format(1e-6 *
                                                          std_dE[-10:].mean()))
plt.xlabel('Turns')
plt.ylabel('Energy spread [MeV]')
plt.savefig(this_directory + '../output_files/EX_13_fig/std_dE_QE.png')
plt.close()

plt.figure(figsize=[6, 4.5])
plt.plot(1e12 * 4.0 * std_dt, lw=2)
print('Equilibrium bunch length = {0:1.3f} [ps]'.format(4e12 *
                                                        std_dt[-10:].mean()))
plt.xlabel('Turns')
plt.ylabel('Bunch length [ps]')
plt.savefig(this_directory + '../output_files/EX_13_fig/bl_QE.png')
plt.close()

print("Done!")
