'''
Tests of the new OTFB with batches and comparison with the old OTFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing..\n')
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut

from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new, CavityFeedbackCommissioning_new
from blond.llrf.cavity_feedback import SPSOneTurnFeedback, CavityFeedbackCommissioning
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import matched_from_distribution_function
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker


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
N_t = 10                                        # Number of turns to track
N_bunches = 100                                 # Number of bunches
bunch_spacing = 5                               # Number of buckets between bunches
N_bunches_padding = 500                         # Number of empty buckets before first bunch
N_pretrack = 1000                               # Number of turns of pretracking


# Objects ---------------------------------------------------------------------
print('Initializing Objects...\n')

# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)
SPS_ring_old = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h], [V], [phi], n_rf=1)
rfstation_old = RFStation(SPS_ring_old, [h], [V], [phi], n_rf=1)

# SINGLE BUNCH FIRST
# Beam
beam_single = Beam(SPS_ring, N_m, N_b)
beam_single_old = Beam(SPS_ring_old, N_m, N_b)

# Tracker object for full ring
SPS_rf_tracker_single = RingAndRFTracker(rfstation, beam_single)
SPS_tracker_single = FullRingAndRF([SPS_rf_tracker_single])
SPS_rf_tracker_single_old = RingAndRFTracker(rfstation_old, beam_single_old)
SPS_tracker_single_old = FullRingAndRF([SPS_rf_tracker_single_old])

# Initialize the bunch
matched_from_distribution_function(beam_single, SPS_tracker_single, bunch_length=bl, distribution_type="gaussian")
matched_from_distribution_function(beam_single_old, SPS_tracker_single_old, bunch_length=bl, distribution_type="gaussian")

# MULTIPLE BUNCHES
beam = Beam(SPS_ring, N_m * N_bunches, N_b * N_bunches)
beam_old = Beam(SPS_ring_old, N_m * N_bunches, N_b * N_bunches)

for i in range(N_bunches):
    beam.dt[int(i * N_m):int((i + 1) * N_m)] = beam_single.dt + i * rfstation.t_rf[0, 0] * bunch_spacing \
                                               + N_bunches_padding * rfstation.t_rf[0, 0]
    beam.dE[int(i * N_m):int((i + 1) * N_m)] = beam_single.dE

    beam_old.dt[int(i * N_m):int((i + 1) * N_m)] = beam_single_old.dt + i * rfstation_old.t_rf[0, 0] * bunch_spacing \
                                                   + N_bunches_padding * rfstation_old.t_rf[0, 0]
    beam_old.dE[int(i * N_m):int((i + 1) * N_m)] = beam_single_old.dE

SPS_rf_tracker = RingAndRFTracker(rfstation, beam)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])
SPS_rf_tracker_old = RingAndRFTracker(rfstation_old, beam_old)
SPS_tracker_old = FullRingAndRF([SPS_rf_tracker_old])


# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=40 * 4620))
profile.track()
profile_old = Profile(beam_old, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=40 * 4620))
profile_old.track()

# One Turn Feedback
Commissioning = CavityFeedbackCommissioning_new(open_FF=True)
OTFB = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                              Commissioning=Commissioning, V_part=1)

Commissioning_old = CavityFeedbackCommissioning(open_FF=True)
OTFB_old = SPSOneTurnFeedback(rfstation, beam, profile, 3, a_comb=63/64,
                              Commissioning=Commissioning, V_part=1)

# Simulation ------------------------------------------------------------------
print('Simulating...\n')

# Pre-tracking
for i in range(N_pretrack):
    OTFB.track_no_beam()
    OTFB_old.track_no_beam()


# Tracking with the beam
map_ = [SPS_tracker] + [profile] + [OTFB]
map_old = [SPS_tracker_old] + [profile_old] + [OTFB_old]
dt_plot = 2

for i in range(N_t):
    for m in map_:
        m.track()

    for m in map_old:
        m.track()

    if i % dt_plot == 0:
        ut.plot_beam_test_comparison(OTFB, OTFB_old)

