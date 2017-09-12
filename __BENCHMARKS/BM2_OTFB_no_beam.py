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
import logging

from toolbox.logger import Logger
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam, Proton
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from trackers.tracker import RingAndRFTracker


# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                    # 200 MHz system harmonic
V = [4.5e6] #2.2e6                   # 200 MHz RF voltage
phi = [0.]                    # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1                  # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)

plt.rc('font', family='sans-serif')  
plt.rc('legend', fontsize=12)  


# Logger for messages on console & in file
Logger(debug = True)

# Set up machine parameters
ring = Ring(N_t, C, alpha, p_s, Particle=Proton())
print("Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, 1, h, V, phi)
#rf.omega_rf[0,0] = 2*np.pi*200.222e6 # cavity central frequency
logging.debug("RF frequency %.6e Hz", rf.omega_rf[0,0]/(2*np.pi))
logging.debug("Revolution period %.6e s", rf.t_rev[0])
print("RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, 
           reinsertion = True) 
print("Beam set! Number of particles %d" %len(beam.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt), 
                                                     np.max(beam.dt)))

profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9, 
    cut_right=rf.t_rev[0], n_slices=4620))
profile.track()

Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=False,
                                            open_FB=False, open_drive=False)
#Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=True,
#                                            open_FB=False, open_drive=True)
#Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=False,
#                                            open_FB=True, open_drive=False)
OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.5, a_comb=15/16, 
                         turns=50, Commissioning=Commissioning)


