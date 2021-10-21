'''
Script to test possible solutions for the end of turn tracking

author: Birk Emil Karlsen-Bæck
'''
# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utils_test as ut
import sys
import scipy.signal

from blond.utils import bmath as bm

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new


# BLonD Parameters ------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_m = 1e5                                       # Number of macro-particles for tracking
N_b = 1.0e11                                    # Bunch intensity [ppb]
N_t = 1                                         # Number of turns to track


# Quick BLonD Simulation Setup ------------------------------------------------
# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h], [V], [phi], n_rf=1)

# Beam
beam = Beam(SPS_ring, N_m, N_b)
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=4620))
profile.track()

OTFB = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64)
OTFB.track_no_beam()


# Parameters ------------------------------------------------------------------
h = 4620
l = int(h/8)
A = 1


# Arrays ----------------------------------------------------------------------
h0s = [h + int(h/2), 2 * h - l]

sig1 = ut.make_step(2 * h, l, A, h0s[0])
sig2 = ut.make_step(2 * h, l, A, h0s[1])

# plt.plot(np.abs(sig1))
# plt.plot(np.abs(sig2))
# plt.show()


# Convolutions ----------------------------------------------------------------
convolved3 = scipy.signal.fftconvolve(sig1, OTFB.TWC.h_gen, mode='full')
convolved4 = scipy.signal.fftconvolve(sig2, OTFB.TWC.h_gen, mode='full')

sig1 = np.concatenate((sig1, np.zeros(h, dtype=complex)))
sig2 = np.concatenate((sig2, np.zeros(h, dtype=complex)))

convolved1 = scipy.signal.fftconvolve(sig1, OTFB.TWC.h_gen, mode='full')[:np.shape(sig1)[0]]
convolved2 = scipy.signal.fftconvolve(sig2, OTFB.TWC.h_gen, mode='full')[:np.shape(sig2)[0]]

plt.plot(np.abs(convolved1))
plt.plot(np.abs(convolved2))
plt.plot(np.abs(convolved3))
plt.plot(np.abs(convolved4))
plt.show()

