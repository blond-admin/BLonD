'''
Tests of the new OTFB with full machine.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing..\n')
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
from OTFB_full_machine_theory import theory_calc
from matplotlib import gridspec
from argparse import ArgumentParser

plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble': r'\usepackage{fourier}',
    'font.family': 'serif'
})


from blond.llrf.new_SPS_OTFB import SPSCavityFeedback_new, CavityFeedbackCommissioning_new
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import matched_from_distribution_function
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

# Parse arguments -------------------------------------------------------------
parser = ArgumentParser(description='G_tx values')

parser.add_argument('--G_tx1', type=float)
parser.add_argument('--G_tx2', type=float)

args = parser.parse_args()

# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]
bl = 1.2e-9                                     # Bunch length [s]

# Parameters for the Simulation
N_m = int(1e5)                                  # Number of macro-particles for tracking
N_b = 2.3e11                                    # Bunch intensity [ppb]
N_t = 100                                       # Number of turns to track
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
matched_from_distribution_function(beam_single, SPS_tracker_single, bunch_length=bl, distribution_type="gaussian",
                                   seed=1234)

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
V_part = 0.5172
G_tx_ls = [0.2564371551236985, 0.53055789217211453]
G_tx_ls = [0.2712028956, 0.58279606]

if args.G_tx1 is not None:
    G_tx_ls[0] = args.G_tx1
if args.G_tx2 is not None:
    G_tx_ls[1] = args.G_tx2

G_llrf_ls = [41.751786, 35.24865]

Commissioning = CavityFeedbackCommissioning_new(open_FF=True, debug=False)
OTFB = SPSCavityFeedback_new(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                              Commissioning=Commissioning, G_tx=G_tx_ls, G_llrf=G_llrf_ls)

# Optimizing the transmitter gain ---------------------------------------------
OPTIMIZE = False
if OPTIMIZE:
    V_1_pt = np.abs(np.mean(OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:]))
    V_2_pt = np.abs(np.mean(OTFB.OTFB_2.V_ANT[-OTFB.OTFB_2.n_coarse:]))
    V_1_exp = V_part * V
    V_2_exp = (1 - V_part) * V
    diff_1 = 100 * np.abs(V_1_pt - V_1_exp)/V_1_exp
    diff_2 = 100 * np.abs(V_2_pt - V_2_exp)/V_2_exp

    print(f'Expected voltages are {V_1_exp} for the 3-sections and {V_2_exp} for the 4-sections.')
    print(f'The G_tx are {OTFB.OTFB_1.G_tx} for the 3-sections and {OTFB.OTFB_2.G_tx} for the 4-sections.')
    print(f'The values after pre-tracking are {V_1_pt} for the 3-sections and {V_2_pt} for the 4-sections.')
    print(f'The difference is {diff_1}% for the 3-sections and {diff_2}% for the 4-sections.')


# Simulation ------------------------------------------------------------------
print('Simulating...\n')
OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
ut.print_stats(OTFB)
print('3-section')
print(f'\tI_gen without beam from code:', np.mean(OTFB.OTFB_1.I_GEN[-h:]) / OTFB.OTFB_1.T_s)
print(f'\tPower for 3-section is {np.max(OTFB.OTFB_1.P_GEN[-OTFB.OTFB_1.n_coarse:])}')
print('4-section')
print(f'\tI_gen without beam from code:', np.mean(OTFB.OTFB_2.I_GEN[-h:]) / OTFB.OTFB_2.T_s)
print(f'\tPower for 4-section is {np.max(OTFB.OTFB_2.P_GEN[-OTFB.OTFB_2.n_coarse:])}')

#ut.plot_IQ_full_machine(OTFB.OTFB_1, cav_type=3, with_beam=False, with_theory=True)
#ut.plot_IQ_full_machine(OTFB.OTFB_2, cav_type=4, with_beam=False, with_theory=True)


# Tracking with the beam
map_ = [profile] + [OTFB] + [SPS_tracker]
dt_plot = 100
dt_track = 10

for i in range(N_t):
    for m in map_:
        m.track()

    if i % dt_plot == 0 and not i == 0:
        print(f'Turn: {i}')
        # Plot things here:


    if i % dt_track == 0 and not i % dt_plot == 0:
        print(f'Turn: {i}')


OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
I_beam1 = np.max(np.real(OTFB.OTFB_1.I_COARSE_BEAM[-h:]))
I_beam1 = (I_beam1 / OTFB.OTFB_1.T_s / 5)
I_beam2 = np.max(np.real(OTFB.OTFB_2.I_COARSE_BEAM[-h:]))
I_beam2 = (I_beam2 / OTFB.OTFB_2.T_s / 5)

print('3-section')
print('\t', I_beam1, I_beam1 * OTFB.OTFB_1.n_cavities)
print(f'\tV_beam with beam from code:', np.mean(OTFB.OTFB_1.V_IND_COARSE_BEAM[-h:]))
print(f'\tI_gen with beam from code:', np.mean(OTFB.OTFB_1.I_GEN[-h:] / OTFB.OTFB_1.T_s))
print(f'\tPower is {np.max(OTFB.OTFB_1.P_GEN[-OTFB.OTFB_1.n_coarse:])}')
print('4-section')
print('\t', I_beam2, I_beam2 * OTFB.OTFB_2.n_cavities)
print(f'\tV_beam with beam from code:', np.mean(OTFB.OTFB_2.V_IND_COARSE_BEAM[-h:]))
print(f'\tI_gen with beam from code:', np.mean(OTFB.OTFB_2.I_GEN[-h:] / OTFB.OTFB_2.T_s))
print(f'\tPower is {np.max(OTFB.OTFB_2.P_GEN[-OTFB.OTFB_2.n_coarse:])}')

ut.plot_IQ_full_machine_v2(OTFB, with_beam=True, with_theory=True)

OPTIMIZE = False
if OPTIMIZE:
    V_1_pt = np.abs(np.mean(OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:]))
    V_2_pt = np.abs(np.mean(OTFB.OTFB_2.V_ANT[-OTFB.OTFB_2.n_coarse:]))
    V_1_exp = V_part * V
    V_2_exp = (1 - V_part) * V
    diff_1 = 100 * np.abs(V_1_pt - V_1_exp)/V_1_exp
    diff_2 = 100 * np.abs(V_2_pt - V_2_exp)/V_2_exp

    print(f'Expected voltages are {V_1_exp} for the 3-sections and {V_2_exp} for the 4-sections.')
    print(f'The G_tx are {OTFB.OTFB_1.G_tx} for the 3-sections and {OTFB.OTFB_2.G_tx} for the 4-sections.')
    print(f'The values after tracking are {V_1_pt} for the 3-sections and {V_2_pt} for the 4-sections.')
    print(f'The difference is {diff_1}% for the 3-sections and {diff_2}% for the 4-sections.')


print('Done!')
