# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for llrf.filters and llrf.cavity_feedback

:Authors: **Helga Timko**
"""

import numpy as np
from scipy.constants import e
import matplotlib.pyplot as plt 

from toolbox.logger import Logger
from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from beams.beams import Beam
from beams.distributions import bigaussian
from beams.slices import Slices
#from llrf.cavity_feedback import SPSOneTurnFeedback
from llrf.filters import rf_beam_current
#import logging

# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = 4620                    # 200 MHz system harmonic
V = 4.5e6                   # 200 MHz RF voltage
phi = 0.                    # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1000                  # Number of turns to track
# CERN SPS --------------------------------------------------------------------


# Logger for verbose output
Logger()
   

# Set up machine parameters
GeneralParams = GeneralParameters(N_t, C, alpha, p_s)
print("Machine parameters set!")

# Set up RF parameters
RFParams = RFSectionParameters(GeneralParams, 1, h, V, phi)
print("RF parameters set!")

# Define beam and fill it
Beam = Beam(GeneralParams, N_m, N_b)
bigaussian(GeneralParams, RFParams, Beam, 3.2e-9/4, seed = 1234, 
           reinsertion = True) 
print("Beam set! Number of particles %d" %len(Beam.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(Beam.dt), 
                                                     np.max(Beam.dt)))

Slices = Slices(RFParams, Beam, 100, cut_left=-1.e-9, cut_right=6.e-9)
Slices.track()
plt.plot(Slices.bin_centers, Slices.n_macroparticles)
plt.show()
print("Slices set! Integral of slices is %d" %np.sum(Slices.n_macroparticles))
Q_tot = Beam.intensity*Beam.charge*e/Beam.n_macroparticles*np.sum(Slices.n_macroparticles)
print("Total charges %.4e C" %Q_tot)

print(RFParams.omega_rf[0][0]) # To be CORRECTED in RFParams!!!
rf_current = rf_beam_current(Slices, RFParams.omega_rf[0][0])
print("RF current is %.e4 A" %np.max(rf_current.real))
plt.plot(rf_current.real)
plt.plot(rf_current.imag)
plt.show()
#OTFB = SPSOneTurnFeedback(RFParams, Beam, Slices) 

print("")
print("Done!")
