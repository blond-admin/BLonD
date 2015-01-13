# Generate phase noise for LHC controlled emittance blow-up during the 
# acceleration ramp, with a moving window in the spectrum
# No intensity effects
# H. Timko


import time 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from input_parameters.general_parameters import *
from input_parameters.preprocess import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from llrf.feedbacks import *
from llrf.RF_noise import *
from beams.beams import *
from beams.distributions import *
from beams.slices import *
from monitors.monitors import *
from plots.plot_settings import *
from plots.plot_beams import *
from plots.plot_llrf import *
from plots.plot_slices import *



# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.2e9          # Intensity
N_p = 50001          # Macro-particles
tau_0 = 1.2          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
h = 35640            # Harmonic number
dphi = 0.            # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 8700001        # Number of turns to track; full ramp: 8700001



# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Set up plot formatting
PlotSettings().set_plot_format()

# Import pre-processed momentum and voltage for the acceleration ramp
ps = np.loadtxt('LHC_momentum_programme.dat', unpack=True)
V = np.loadtxt('LHC_voltage_programme.dat', unpack=True)
print "Momentum and voltage loaded..."

# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, ps[0:N_t+1], 'proton')
print "General parameters set..."
 
# Define RF station parameters, phase loop, and corresponding tracker
RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], dphi)

# Generate the phase noise    
LHCnoise = LHCFlatSpectrum(general_params, RF_params, 1000000, corr_time = 10000, 
                           fmin = 0.8571, fmax = 1.1, initial_amplitude = 4.44e-5)
LHCnoise.generate()
 
# Save data
np.savetxt('LHC_noise_programme.dat', np.c_[LHCnoise.dphi], fmt='%.10e')
 
print "Done!"
print ""

