
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input to pre-process momentum and voltage machine data
'''

import numpy as np

from input_parameters.general_parameters import *
from input_parameters.preprocess import *
from plots.plot_settings import *


# Simulation parameters --------------------------------------------------------
# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor


# Setting up the simulation ----------------------------------------------------
# Import data defining the acceleration ramp
#time, momentum = np.loadtxt('../input_files/TC7_momentum.dat', unpack=True)
time, momentum = loaddata('../input_files/TC7_momentum.dat')
momentum *= 1.e9     # Make sure momentum is in eV!
print "Time and momentum loaded..."

# Set up plot formatting
PlotSettings().set_plot_format()

# Interpolate ramp data
ps = preprocess_ramp('proton', C, time, momentum, flat_bottom=10000, 
                     flat_top=20000, figname='Momentum', figdir='../output_files/TC7_fig')
print "Ramp data pre-processed..."

# Define general parameters based on momentum programme
# Momentum programme defines also the number of time steps
general_params = GeneralParameters((len(ps)-1), C, alpha, ps, 'proton')
print "General parameters set..."

# EXAMPLE 1: Interpolate similarly a single-harmonic voltage programme
#time1, voltage1 = np.loadtxt('../input_files/TC7_voltage_1.dat', unpack=True)
time1, voltage1 = loaddata('../input_files/TC7_voltage_1.dat')

preprocess_rf_params(general_params, time1, voltage1, flat_bottom=10000, 
                     figname='Voltage_ex1', figdir='../output_files/TC7_fig')
print "Voltage of example 1 pre-processed..."

# EXAMPLE 2: Interpolate multiple harmonics, e.g. by putting them in a list
voltage2 = 0.2*voltage1
preprocess_rf_params(general_params, time1, [voltage1, voltage2], 
                     flat_bottom=10000, figname='Voltage_ex2', figdir='../output_files/TC7_fig')
print "Voltages of example 2 pre-processed..."


print ""
print "Done!"
print ""


