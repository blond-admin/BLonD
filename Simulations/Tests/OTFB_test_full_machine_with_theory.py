'''
Tests of the new OTFB with full machine.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing..\n')
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
from matplotlib import gridspec

plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble': r'\usepackage{fourier}',
    'font.family': 'serif'
})


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
V = 4e6                                         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]
bl = 2e-9                                       # Bunch length [s]

# Parameters for the Simulation
N_m = int(1e5)                                  # Number of macro-particles for tracking
N_b = 2.3e11                                    # Bunch intensity [ppb]
N_t = 100                                        # Number of turns to track
N_bunches = h//5                                # Number of bunches
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
OTFB = SPSOneTurnFeedback_new(rfstation, beam, profile, n_sections=4, n_cavities=2, a_comb=63/64,
                              Commissioning=Commissioning, V_part=1)

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

I_beam = np.max(np.real(OTFB.I_COARSE_BEAM[-h:]))
I_beam = (I_beam / OTFB.T_s / 5) * OTFB.n_cavities
print(I_beam, I_beam/OTFB.n_cavities, OTFB.T_s)

I_g_no_beam, I_g_beam = ut.theoretical_signals(OTFB, I_beam)

#print('integral over h_gen:',np.sum(OTFB.TWC.h_gen) * OTFB.T_s)
#print('integral over h_beam:',np.sum(OTFB.TWC.h_beam_coarse) * OTFB.T_s)
#print()

#print('I_gen without beam from theory:', I_g_no_beam)
#print('I_gen with beam theory:', I_g_beam)
#print()

#print('Power without beam, theory:', ut.get_power_gen_I2(I_g_no_beam, 50))
#print('Power with beam, theory:', ut.get_power_gen_I2(I_g_beam, 50))












