# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example script to take into account intensity effects from impedance tables

:Authors: **Danilo Quartullo**
'''

from __future__ import division, print_function

import os
from builtins import bytes, range, str

import matplotlib as mpl
import numpy as np
from scipy.constants import c, e, m_p

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import (InducedVoltageFreq, InductiveImpedance,
                                        TotalInducedVoltage)
from blond.impedances.impedance_sources import InputTable
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.plots.plot_impedance import (plot_impedance_vs_frequency,
                                        plot_induced_voltage_vs_bin_centers)
from blond.trackers.tracker import RingAndRFTracker

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_02_fig', exist_ok=True)


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
n_particles = 1e11
n_macroparticles = 5e5
sigma_dt = 180e-9 / 4  # [s]
kin_beam_energy = 1.4e9  # [eV]

# Machine and RF parameters
radius = 25
gamma_transition = 4.4  # [1]
C = 2 * np.pi * radius  # [m]

# Tracking details
n_turns = 2
n_turns_between_two_plots = 1

# Derived parameters
E_0 = m_p * c**2 / e    # [eV]
tot_beam_energy = E_0 + kin_beam_energy  # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV / c]
momentum_compaction = 1 / gamma_transition**2  # [1]

# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 1
voltage_program = 8e3  # [V]
phi_offset = np.pi


# DEFINE RING------------------------------------------------------------------

ring = Ring(C, momentum_compaction, sync_momentum,
            Proton(), n_turns)

RF_sct_par = RFStation(ring, [harmonic_numbers],
                       [voltage_program], [phi_offset], n_rf_systems)

my_beam = Beam(ring, n_macroparticles, n_particles)

ring_RF_section = RingAndRFTracker(RF_sct_par, my_beam)

# DEFINE BEAM------------------------------------------------------------------
bigaussian(ring, RF_sct_par, my_beam, sigma_dt, seed=1)

# DEFINE SLICES----------------------------------------------------------------
slice_beam = Profile(my_beam, CutOptions(cut_left=-5.72984173562e-7,
                                         cut_right=5.72984173562e-7, n_slices=10000))

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor(ring, RF_sct_par, my_beam,
                            this_directory + '../output_files/EX_02_output_data', buffer_time=1)

# LOAD IMPEDANCE TABLES--------------------------------------------------------

var = str(kin_beam_energy / 1e9)

# ejection kicker
Ekicker = np.loadtxt(this_directory + '../input_files/EX_02_Ekicker_1.4GeV.txt', skiprows=1, dtype=complex, converters={0: lambda s:
                                                                                                                        complex(bytes(s).decode('UTF-8').replace('i', 'j')),
                                                                                                                        1: lambda s: complex(bytes(s).decode('UTF-8').replace('i', 'j'))})

Ekicker_table = InputTable(Ekicker[:, 0].real, Ekicker[:, 1].real, Ekicker[:, 1].imag)


# Finemet cavity
F_C = np.loadtxt(this_directory + '../input_files/EX_02_Finemet.txt', dtype=float, skiprows=1)

F_C[:, 3], F_C[:, 5], F_C[:, 7] = np.pi * F_C[:, 3] / 180, np.pi * F_C[:, 5] / 180, np.pi * F_C[:, 7] / 180

option = "closed loop"

if option == "open loop":
    Re_Z = F_C[:, 4] * np.cos(F_C[:, 3])
    Im_Z = F_C[:, 4] * np.sin(F_C[:, 3])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
elif option == "closed loop":
    Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
    Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
elif option == "shorted":
    Re_Z = F_C[:, 6] * np.cos(F_C[:, 7])
    Im_Z = F_C[:, 6] * np.sin(F_C[:, 7])
    F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)
else:
    pass

# steps
steps = InductiveImpedance(my_beam, slice_beam, 34.6669349520904 / 10e9 *
                           ring.f_rev, RF_sct_par, deriv_mode='diff')
# direct space charge
dir_space_charge = InductiveImpedance(my_beam, slice_beam, -376.730313462
                                      / (ring.beta[0] * ring.gamma[0] ** 2),
                                      RF_sct_par)


# INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

imp_list = [Ekicker_table, F_C_table]

ind_volt_freq = InducedVoltageFreq(my_beam, slice_beam, imp_list,
                                   frequency_resolution=2e5)


total_induced_voltage = TotalInducedVoltage(my_beam, slice_beam,
                                            [ind_volt_freq, steps, dir_space_charge])

# PLOTS

format_options = {'dirname': this_directory + '../output_files/EX_02_fig', 'linestyle': '.'}
plots = Plot(ring, RF_sct_par, my_beam, 1, n_turns, 0,
             5.72984173562e-7, - my_beam.sigma_dE * 4.2, my_beam.sigma_dE * 4.2, xunit='s',
             separatrix_plot=True, Profile=slice_beam, h5file=this_directory + '../output_files/EX_02_output_data',
             histograms_plot=True, format_options=format_options)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))


# ACCELERATION MAP-------------------------------------------------------------

map_ = [total_induced_voltage] + [ring_RF_section] + [slice_beam] + [bunchmonitor] + [plots]

# TRACKING + PLOTS-------------------------------------------------------------

for i in range(1, n_turns + 1):

    print(i)

    for m in map_:
        m.track()

    # Plots
    if (i % n_turns_between_two_plots) == 0:

        plot_impedance_vs_frequency(ind_volt_freq, figure_index=i, cut_up_down=(0,1000), cut_left_right=(0,3e9), show_plots=False,
                                    plot_total_impedance=False, style='-', plot_interpolated_impedances=False, plot_spectrum=False, dirname=this_directory + '../output_files/EX_02_fig')

        plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='.', dirname=this_directory + '../output_files/EX_02_fig', show_plots=False)

# For testing purposes
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(my_beam.dE), np.std(my_beam.dE), np.mean(my_beam.dt), np.std(my_beam.dt))
with open(this_directory + '../output_files/EX_02_test_data.txt', 'w') as f:
    f.write(test_string)


print("Done!")
