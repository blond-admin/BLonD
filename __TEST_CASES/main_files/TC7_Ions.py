
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of ion dynamics
No intensity effects
'''

from __future__ import division, print_function
from builtins import range
import numpy as np
from scipy.constants import physical_constants
# Atomic Mass Unit [eV]
u = physical_constants['atomic mass unit-electron volt relationship'][0] 

from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from beams.beams import *
from beams.distributions import *
from beams.slices import *
from monitors.monitors import *
from plots.plot_beams import *
from plots.plot_impedance import *
from plots.plot_slices import *
from plots.plot import *
from plots.plot_parameters import *


# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 5.0e11                 # Design Intensity in SIS100
N_p = 50000                  # Macro-particles
tau_0 = 100.0e-9             # Initial bunch length, 4 sigma [s]
Z = 28.                      # Charge state of Uranium
m_p = 238.05078826*u         # Isotope mass of U-238

# Machine and RF parameters
C = 1083.6                   # Machine circumference [m]
p_i = 153.37e9               # Synchronous momentum [eV/c]
p_f = 535.62e9               # Synchronous momentum, final 535.62e9
h = 10                       # Harmonic number
V = 280.e3                   # RF voltage [V]
dphi = 0.                    # Phase modulation/offset
gamma_t = 15.59              # Transition gamma
alpha = 1./gamma_t/gamma_t   # First order mom. comp. factor

# Tracking details
N_t = 45500                  # Number of turns to track
dt_plt = 5000                # Time steps between plots


# Simulation setup -------------------------------------------------------------
print("Setting up the simulation...")
print("")


# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, np.linspace(p_i, p_f, N_t+1), 
                                   'user_input', user_mass=m_p, user_charge=Z)

# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
print("Particle mass is %.3e eV" %general_params.mass)
print("Particle charge is %d e" %general_params.charge)

linspace_test = np.linspace(p_i, p_f, N_t+1)
momentum_test = general_params.momentum
beta_test = general_params.beta
gamma_test = general_params.gamma
energy_test = general_params.energy
mass_test = general_params.mass # [eV]
charge_test = general_params.charge # e*Z

# Define RF station parameters and corresponding tracker
rf_params = RFSectionParameters(general_params, 1, h, V, dphi)
print("Initial bucket length is %.3e s" %(2.*np.pi/rf_params.omega_RF[0,0]))
print("Final bucket length is %.3e s" %(2.*np.pi/rf_params.omega_RF[0,N_t]))

phi_s_test = rf_params.phi_s #: *Synchronous phase
omega_RF_d_test = rf_params.omega_RF_d #: *Design RF frequency of the RF systems in the station [GHz]*
omega_RF_test = rf_params.omega_RF  #: *Initial, actual RF frequency of the RF systems in the station [GHz]*
phi_RF_test = rf_params.omega_RF #: *Initial, actual RF phase of each harmonic system*
E_increment_test = rf_params.E_increment #Energy increment (acceleration/deceleration) between two turns,


long_tracker = RingAndRFSection(rf_params, beam)

eta_0_test = rf_params.eta_0 #: *Slippage factor (0th order) for the given RF section*
eta_1_test = rf_params.eta_1 #: *Slippage factor (1st order) for the given RF section*
eta_2_test = rf_params.eta_2 #: *Slippage factor (2nd order) for the given RF section*
alpha_order_test = rf_params.alpha_order

longitudinal_bigaussian(general_params, rf_params, beam, tau_0/4, 
                        reinsertion = 'on', seed=1)


# Need slices for the Gaussian fit
slice_beam = Slices(rf_params, beam, 100)

# Define what to save in file
bunchmonitor = BunchMonitor(general_params, rf_params, beam,
                            '../output_files/TC7_output_data',
                            Slices=slice_beam)

format_options = {'dirname': '../output_files/TC7_fig'}
plots = Plot(general_params, rf_params, beam, dt_plt, N_t, -4.e-7, 4.e-7,
             -400e6, 400e6, separatrix_plot=True, Slices=slice_beam,
             h5file='../output_files/TC7_output_data', 
             format_options=format_options)

# Accelerator map
map_ = [long_tracker] + [slice_beam] + [bunchmonitor] + [plots]
print("Map set")
print("")



# Tracking ---------------------------------------------------------------------
for i in range(1, N_t+1):
    
    
    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print("Outputting at time step %d..." %i)
        print("   Beam momentum %.6e eV" %beam.momentum)
        print("   Beam gamma %3.3f" %beam.gamma)
        print("   Beam beta %3.3f" %beam.beta)
        print("   Beam energy %.6e eV" %beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" %(4.*beam.sigma_dt))
        #print "   Gaussian bunch length %.4e s" %slice_beam.bl_gauss
        print("")
        
    # Track
    for m in map_:
        m.track()
        
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(general_params, rf_params, beam)
    #beam.losses_longitudinal_cut(0.28e-4/general_params.omega_rev[i], 0.75e-4/general_params.omega_rev[i])
    
print("Done!")
print("")
