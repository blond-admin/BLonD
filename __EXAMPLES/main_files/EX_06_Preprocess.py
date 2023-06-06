# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test case to show how to use preprocess_ramp and preprocess_rf_params in the
main file (CERN PS Booster context).

:Authors: **Danilo Quartullo**
'''

from __future__ import division

import os

import matplotlib as mpl
import numpy as np

from blond.beam.beam import Proton
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.rf_parameters_options import RFStationOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.ring_options import RingOptions

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_06_fig', exist_ok=True)

# Beam parameters
n_particles = 3e12


# Machine and RF parameters
radius = 25  # [m]
gamma_transition = 4.076750841
alpha = 1 / gamma_transition**2
C = 2 * np.pi * radius  # [m]

# Initial and final simulation times
initial_time = 0.277  # [s]
final_time = 0.700  # [s]

momentum_program = np.loadtxt(this_directory + '../input_files/EX_06_Source_TOF_P.csv',
                              delimiter=',')
time_array = momentum_program[:, 0] * 1e-3  # [s]
momentum = momentum_program[:, 1] * 1e9  # [eV/c]

particle_type = Proton()
ring_opt = RingOptions(interpolation='linear', plot=True,
                       figdir=this_directory + '../output_files/EX_06_fig',
                       t_start=initial_time, t_end=final_time)

general_params = Ring(C, alpha, (time_array, momentum), particle_type,
                      RingOptions=ring_opt)

# Cavities parameters
n_rf_systems = 3
harmonic_numbers_1 = 1
harmonic_numbers_2 = 2
harmonic_numbers_3 = 16
phi_rf_1 = 0  # [rad]
phi_rf_2 = np.pi  # [rad]
phi_rf_3 = np.pi / 6  # [rad]

voltage_program_C02 = np.loadtxt(
    this_directory + '../input_files/EX_06_voltage_program_LHC25_c02.txt')
voltage_program_C04 = np.loadtxt(
    this_directory + '../input_files/EX_06_voltage_program_LHC25_c04.txt')
voltage_program_C16 = np.loadtxt(
    this_directory + '../input_files/EX_06_voltage_program_LHC25_c16.txt')
time_C02 = voltage_program_C02[:, 0] * 1e-3  # [s]
voltage_C02 = voltage_program_C02[:, 1] * 1e3  # [V]
time_C04 = voltage_program_C04[:, 0] * 1e-3  # [s]
voltage_C04 = voltage_program_C04[:, 1] * 1e3  # [V]
time_C16 = voltage_program_C16[:, 0] * 1e-3  # [s]
voltage_C16 = voltage_program_C16[:, 1] * 1e3  # [V]

rf_station_options = RFStationOptions(
    interpolation='linear', smoothing=0,
    plot=True, figdir=this_directory + '../output_files/EX_06_fig',
    figname=['voltage_C02 [V]', 'voltage_C04 [V]', 'voltage_C16 [V]'],
    sampling=1)

rf_params = RFStation(
    general_params,
    [harmonic_numbers_1, harmonic_numbers_2, harmonic_numbers_3],
    ((time_C02, voltage_C02),
     (time_C04, voltage_C04),
     (time_C16, voltage_C16)),
    [phi_rf_1, phi_rf_2, phi_rf_3], n_rf_systems,
    RFStationOptions=rf_station_options)

print("Done!")
