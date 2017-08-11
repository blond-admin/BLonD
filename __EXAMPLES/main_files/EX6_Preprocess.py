
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
'''
Test case to show how to use preprocess_ramp and preprocess_rf_params in the
main file (CERN PS Booster context).
'''

from __future__ import division
import numpy as np
from input_parameters.preprocess import *
from input_parameters.ring import *
from beam.beam import *
from input_parameters.rf_parameters import *
from beam.profile import *
from monitors.monitors import *
from trackers.tracker import *


# Beam parameters
particle_type = 'proton'
n_particles = 3e12


# Machine and RF parameters
radius = 25 # [m]
gamma_transition = 4.076750841
alpha = 1 / gamma_transition**2
C = 2*np.pi*radius  # [m]

# Initial and final simulation times
initial_time = 0.275 # [s]
final_time = 0.700 # [s]

momentum_program = np.loadtxt('../input_files/EX6_Source_TOF_P.csv',
                              delimiter=',')
time_array = momentum_program[:, 0]*1e-3  # [s]
momentum = momentum_program[:, 1]*1e9  # [eV/c]

initial_index = np.min(np.where(time_array>=initial_time)[0])
final_index = np.max(np.where(time_array<=final_time)[0])

time_cut = time_array[initial_index:(final_index+1)]
momentum_cut = momentum[initial_index:(final_index+1)]

momentum_interp = preprocess_ramp(particle_type, C, time_cut, momentum_cut,
                                  interpolation='linear',
                                  figdir='../output_files/EX6_fig')

n_turns = len(momentum_interp[0])-1

general_params = Ring(n_turns, C, alpha, momentum_interp, 
                                   particle_type)

# Cavities parameters
n_rf_systems = 2                                     
harmonic_numbers_1 = 1 
harmonic_numbers_2 = 2                    
phi_offset_1 = 0   # [rad]
phi_offset_2 = np.pi # [rad]

voltage_program_C02 = np.loadtxt('../input_files/EX6_voltage_program_LHC25_c02.txt')
voltage_program_C04 = np.loadtxt('../input_files/EX6_voltage_program_LHC25_c04.txt')
voltage_program_C16 = np.loadtxt('../input_files/EX6_voltage_program_LHC25_c16.txt')
time_C02 = voltage_program_C02[:, 0]*1e-3  # [s]
voltage_C02 = voltage_program_C02[:, 1]*1e3  # [V]
time_C04 = voltage_program_C04[:, 0]*1e-3  # [s]
voltage_C04 = voltage_program_C04[:, 1]*1e3  # [V]
time_C16 = voltage_program_C16[:, 0]*1e-3  # [s]
voltage_C16 = voltage_program_C16[:, 1]*1e3  # [V]

data_interp = preprocess_rf_params(general_params,
                                   [time_C02, time_C04, time_C16],
                                   [voltage_C02, voltage_C04, voltage_C16],
                                   interpolation='linear', smoothing=0,
                                   plot=True, figdir='../output_files/EX6_fig',
                                   figname=['voltage_C02 [V]',
                                   'voltage_C04 [V]', 'voltage_C16 [V]'],
                                   sampling=1)

rf_params = RFStation(general_params, 2,
                                [harmonic_numbers_1,harmonic_numbers_2],
                                [data_interp[0], data_interp[1]],
                                [phi_offset_1, phi_offset_2])
