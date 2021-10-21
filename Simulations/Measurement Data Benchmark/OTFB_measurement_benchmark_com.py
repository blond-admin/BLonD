'''
This files was made to benchmark the new SPS OTFB model with measurement data from the commissioning at
30.06.2021.

Author: Birk E. Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing..\n')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


from blond.llrf.new_SPS_OTFB import SPSCavityFeedback_new, CavityFeedbackCommissioning_new
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
V = 10e6                                         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_m = int(1e5)                                  # Number of macro-particles for tracking
N_b = 1.1e11                                    # Bunch intensity [ppb]
N_t = 100                                       # Number of turns to track
N_bunches = 72                                  # Number of bunches
bunch_spacing = 5                               # Number of buckets between bunches
N_pretrack = 1000                               # Number of turns of pretracking


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
Commissioning = CavityFeedbackCommissioning_new(open_FF=True, debug=True)
OTFB = SPSCavityFeedback_new(rfstation, beam, profile)

# Simulation ------------------------------------------------------------------
print('Simulating...\n')

# Pre-tracking
for i in range(N_pretrack):
    OTFB.track_no_beam()

OTFB.calc_power()

print(OTFB.omega_c/(2 * np.pi), OTFB.omega_r/(2 * np.pi))

print('I_gen without beam from code:', np.mean(OTFB.I_GEN[-h:]) / OTFB.T_s)
print('Power without beam from code:', np.mean(OTFB.P_GEN[-h:]))

# Tracking with the beam
map_ = [profile] + [OTFB] + [SPS_tracker]
dt_plot = 100
dt_track = 10


for i in range(N_t):
    for m in map_:
        m.track()

    OTFB.calc_power()

    if i % dt_plot == 0 and not i == 0:
        print(f'Turn: {i}')

        ut.plot_beam_test_full_machine(OTFB)
        #ut.plot_IQ_full_machine(OTFB)

        #plt.plot(OTFB.P_GEN)
        #plt.show()

    if i % dt_track == 0 and not i % dt_plot == 0:
        print(f'Turn: {i}')

print('V_beam with beam from code:', np.mean(OTFB.V_IND_COARSE_BEAM[-h:]))
print('I_gen with beam from code:', np.mean(OTFB.I_GEN[-h:] / OTFB.T_s))
print('Power with beam from code:', np.mean(OTFB.P_GEN[-h:]))
print()
