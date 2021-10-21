'''
Tests of the new OTFB with beam.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing..\n')
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
import matplotlib.animation as animation

from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new, CavityFeedbackCommissioning_new
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import matched_from_distribution_function
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.llrf.signal_processing import modulator, moving_average


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]
bl = 2.5e-9                                     # Bunch length [s]

# Parameters for the Simulation
N_m = int(1e5)                                  # Number of macro-particles for tracking
N_b = 2.3e11                                    # Bunch intensity [ppb]
N_t = 1                                         # Number of turns to track
N_bunches = 100                                 # Number of bunches
bunch_spacing = 5.001                               # Number of buckets between bunches


# Objects ---------------------------------------------------------------------
print('Initializing Objects...\n')

# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h], [V], [phi], n_rf=1)

# SINGLE BUNCH FIRST
# Beam
beam_single = Beam(SPS_ring, N_m, N_b)

# Tracker object for full ring
SPS_rf_tracker_single = RingAndRFTracker(rfstation, beam_single)
SPS_tracker_single = FullRingAndRF([SPS_rf_tracker_single])

# Initialize the bunch
matched_from_distribution_function(beam_single, SPS_tracker_single, bunch_length=bl, distribution_type="gaussian")

# MULTIPLE BUNCHES
beam = Beam(SPS_ring, N_m * N_bunches, N_b * N_bunches)

for i in range(N_bunches):
    beam.dt[int(i * N_m):int((i + 1) * N_m)] = beam_single.dt + i * rfstation.t_rf[0, 0] * bunch_spacing
    beam.dE[int(i * N_m):int((i + 1) * N_m)] = beam_single.dE

SPS_rf_tracker = RingAndRFTracker(rfstation, beam)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])


# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=40 * 4620))
profile.track()

# One Turn Feedback
Commissioning = CavityFeedbackCommissioning_new(open_FF=True)
OTFB = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                              Commissioning=Commissioning)


OTFB.track()
plt.plot(OTFB.I_COARSE_BEAM.real/(25e-9), alpha=0.7)
plt.plot(OTFB.I_COARSE_BEAM.imag/(25e-9), alpha=0.7)
plt.plot(np.abs(OTFB.I_COARSE_BEAM)/(25e-9), alpha=0.7)
plt.show()

