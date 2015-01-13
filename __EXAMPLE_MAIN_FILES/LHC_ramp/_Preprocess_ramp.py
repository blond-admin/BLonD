# Example input pre-processing acceleration ramp data
# H. Timko

import numpy as np

from input_parameters.general_parameters import *
from input_parameters.preprocess import *
from plots.plot_settings import *


# Simulation parameters --------------------------------------------------------
# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 8700001           # Number of turns to track


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Set up plot formatting
PlotSettings().set_plot_format()


# Pre-process momentum and voltage for the acceleration ramp
input_t1, input_ps = np.loadtxt('momentum.dat', unpack=True)
input_t2, input_V = np.loadtxt('total_voltage.dat', unpack=True)
print "Data read in..."

ps = preprocess_ramp('proton', C, input_t1, 1.e9*input_ps, 
                     plot=True, figname='momentum', sampling=10, flat_top=40970)
print "Momentum pre-processed..."
print len(ps)
# Check that we obtain the same revolution period from GeneralParameters
general_params = GeneralParameters(N_t, C, alpha, ps, 'proton')
np.savetxt('check_gener.dat', np.c_[general_params.t_rev], fmt='%.10e')
print "General parameters set..."

V = preprocess_rf_params(general_params, input_t2, 1.e6*input_V,  
                         plot=True, figname='voltage', sampling=10)
print "Voltage pre-processed..."

np.savetxt('LHC_momentum_programme.dat', np.c_[ps], fmt='%.10e')
np.savetxt('LHC_voltage_programme.dat', np.c_[V], fmt='%.10e')
print "Saved to file..."


print "Done!"
print ""


