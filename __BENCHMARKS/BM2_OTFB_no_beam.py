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

import logging

import matplotlib.pyplot as plt
import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (CavityFeedbackCommissioning,
                                        SPSCavityFeedback)
from blond.toolbox.logger import Logger

# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [4.5e6]                 # 200 MHz RF voltage
# With this setting, amplitude in the two four-section cavity must converge to
# 4.5 MV * 4/18 * 2 = 2.0 MV
phi = [0.]                  # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1                     # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')  
plt.rc('legend', fontsize=12)  

CLOSED_LOOP = True
OPEN_LOOP = True
OPEN_FB = True
POST_LS2 = False

# Logger for messages on console & in file
Logger(debug = True)

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
logging.debug("RF frequency %.6e Hz", rf.omega_rf[0,0]/(2*np.pi))
logging.debug("Revolution period %.6e s", rf.t_rev[0])
logging.info("...... RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed=1234, reinsertion=True)
logging.info("......Beam set!")
logging.info("Number of particles %d" %len(beam.dt))
logging.info("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt),
                                                     np.max(beam.dt)))

profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9, 
    cut_right=rf.t_rev[0], n_slices=4620))
profile.track()

if CLOSED_LOOP:
    logging.info("...... CLOSED LOOP test")
    Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=False,
        open_FB=False, open_drive=False, open_FF=True)
    OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, a_comb=15/16,
                             turns=50, post_LS2=POST_LS2,
                             Commissioning=Commissioning)
    logging.info("Final voltage %.8e V"
                 %np.average(np.absolute(OTFB.OTFB_1.V_coarse_tot[-10])))

if OPEN_LOOP:
    logging.info("...... OPEN LOOP test")
    Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=True,
        open_FB=False, open_drive=True, open_FF=True)
    OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, a_comb=15/16,
                             turns=50, post_LS2=POST_LS2,
                             Commissioning=Commissioning)
    logging.info("Final voltage %.8e V"
                 %np.average(np.absolute(OTFB.OTFB_1.V_coarse_tot[-10])))

if OPEN_FB:
    logging.info("...... OPEN FEEDBACK test")
    Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=False,
        open_FB=True, open_drive=False, open_FF=True)
    OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, a_comb=15/16,
                             turns=50, post_LS2=POST_LS2,
                             Commissioning=Commissioning)
    logging.info("Final voltage %.8e V"
                 %np.average(np.absolute(OTFB.OTFB_1.V_coarse_tot[-10])))

logging.info("")
logging.info("Done!")

