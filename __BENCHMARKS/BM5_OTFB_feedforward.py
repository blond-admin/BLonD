# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for SPS OTFB with FF

:Authors: **Helga Timko**
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import logging
import os

from blond.toolbox.logger import Logger
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC, SPS5Section200MHzTWC
from blond.llrf.signal_processing import feedforward_filter


# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [4.5e6]                 # 200 MHz RF voltage
phi = [0.]                  # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles per bunch for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 25                    # Number of turns to track
N_pretrack = 25             # Number of turns to pre-track
n_bunches = 144             # Number of bunches
bunch_spacing = 5           # In RF buckets
# CERN SPS --------------------------------------------------------------------


FILTER_DESIGN = True


# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)
# Colors
jet = plt.get_cmap('jet')
colors = jet(np.linspace(0,1,N_t))

# Logger for messages on console & in file
Logger(debug=True)

np.set_printoptions(linewidth=70, precision=10)

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
logging.info("...... RF parameters set!")
T_s = bunch_spacing*rf.t_rf[0,0]
logging.info("RF period %.6e s", rf.t_rf[0,0])
logging.info("Sampling period %.6e s", T_s)


if FILTER_DESIGN:

    logging.info("...... Filter design test")
    logging.info("3-section cavity filter design with modified filling time"+
                 " and fixed n_taps")
    TWC = SPS3Section200MHzTWC()
    TWC.tau = 420e-9
    filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 4/125*1e-6,
        debug=True, taps=31, opt_output=True)
#    print(filter)

    logging.info("3-section cavity filter design")
    TWC_3 = SPS3Section200MHzTWC()
    filter = feedforward_filter(TWC_3, T_s, debug=True)
#    print(filter)

    logging.info("4-section cavity filter design")
    TWC_4 = SPS4Section200MHzTWC()
    filter = feedforward_filter(TWC_4, T_s, debug=True)
#    print(filter)

    logging.info("5-section cavity filter design")
    TWC_5 = SPS5Section200MHzTWC()
    filter = feedforward_filter(TWC_5, T_s, debug=True)
    print(filter)



logging.info("")
logging.info("Done!")