# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Benchmarking SPS OTFB transmitter gain G_tx for every cavity. For this, at the
central frequency 200.222 MHz, the measured values of a generator current
I_gen = 52.5 A for an input generator voltage of V_gen = 1.76 MV should be
reproduced in open feedback response.

:Authors: **Helga Timko**
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from blond.toolbox.logger import Logger
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import SPSCavityFeedback, \
    SPSOneTurnFeedback, CavityFeedbackCommissioning


# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [4.5e6]                 # 200 MHz RF voltage
# With this setting, amplitude in the two PRE_LS2_4SEC cavities must converge
# to 4.5 MV * 4/9 * 2 = 2.0 MV
# With this setting, amplitude in the two PRE_LS2_5SEC cavities must converge
# to 4.5 MV * 5/9 * 2 = 2.5 MV
# With this setting, amplitude in the four POST_LS2_3SEC cavities must converge
# to 4.5 MV * 6/10 * 2 = 2.7 MV
# With this setting, amplitude in the two POST_LS2_4SEC cavities must converge
# to 4.5 MV * 4/10 * 2 = 1.8 MV
phi = [0.]                  # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1                     # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# Printouts
def logging_info():
    logging.info("  Generator resistance R_gen %.8f kOhms", OTFB.TWC.R_gen/1e3)
    voltage = np.average(np.absolute(OTFB.V_coarse_tot[-10]))
    logging.info("  Final voltage cavities %.8e V" %voltage)
    current = np.average(np.absolute(OTFB.I_gen[-10]/OTFB.T_s))
    logging.info("  Final generator current %.8e A" %current)
    logging.info("  Calculated resistance %.8f kOhms" %(voltage/current/1e3))

# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)

PRE_LS2_4SEC = True
PRE_LS2_5SEC = True
POST_LS2_3SEC = True
POST_LS2_4SEC = True

# Logger for messages on console & in file
Logger(debug=True)

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
rf.omega_rf[0,0] = 200.222e6*2*np.pi
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

logging.info("...... OPEN FEEDBACK test")
Commissioning = CavityFeedbackCommissioning(debug=True, open_loop=False,
    open_FB=True, open_drive=False, open_FF=True)

if PRE_LS2_4SEC:
    logging.info("...... PRE-LS2 4-SECTION cavities")
    OTFB = SPSOneTurnFeedback(rf, beam, profile, 4, n_cavities=2,
                              V_part=4/9, G_ff=0, G_llrf=5, G_tx=1.002453405,
                              a_comb=15/16, Commissioning=Commissioning)
    for i in range(50):
        OTFB.track_no_beam()
    logging_info()

if PRE_LS2_5SEC:
    logging.info("...... PRE-LS2 5-SECTION cavities")
    OTFB = SPSOneTurnFeedback(rf, beam, profile, 5, n_cavities=2,
                              V_part=5/9, G_ff=0, G_llrf=5, G_tx=1.00066015,
                              a_comb=15/16, Commissioning=Commissioning)
    for i in range(50):
        OTFB.track_no_beam()
    logging_info()

if POST_LS2_3SEC:
    logging.info("...... POST-LS2 3-SECTION cavities")
    OTFB = SPSOneTurnFeedback(rf, beam, profile, 3, n_cavities=4,
                              V_part=6/10, G_ff=0, G_llrf=5, G_tx=0.99468245,
                              a_comb=15/16, Commissioning=Commissioning)
    for i in range(50):
        OTFB.track_no_beam()
    logging_info()

if POST_LS2_4SEC:
    logging.info("...... POST-LS2 4-SECTION cavities")
    OTFB = SPSOneTurnFeedback(rf, beam, profile, 4, n_cavities=2,
                              V_part=4/10, G_ff=0, G_llrf=5, G_tx=1.002453405,
                              a_comb=15/16, Commissioning=Commissioning)
    for i in range(50):
        OTFB.track_no_beam()
    logging_info()




#OTFB = SPSCavityFeedback(rf, beam, profile, G_llrf=5, G_tx=0.99520546,
#                         a_comb=15/16, turns=50, post_LS2=False,
#                         Commissioning=Commissioning)
#logging.info("Final voltage %.8e V"
#             %np.average(np.absolute(OTFB.OTFB_1.V_coarse_tot[-10])))
