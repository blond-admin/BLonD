'''
This files was made to benchmark the new SPS OTFB model with measurement data from the themis presentation.

Author: Birk E. Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
print('Importing...\n')
import matplotlib.pyplot as plt
import numpy as np
import Simulations.Tests.utils_test as ut
from Simulations.Tests.OTFB_full_machine_theory import theory_calc
from matplotlib import gridspec
from argparse import ArgumentParser
import phase_deviation as pd
import scipy.stats as spst
import scipy.io


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
import blond.utils.bmath as bm


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
N_bunches = 72                                  # Number of bunches
bunch_spacing = 5                               # Number of buckets between bunches
N_pretrack = 1000                               # Number of turns of pretracking
first_bunch = 100


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

bunch_pos = np.array([i for i in range(first_bunch * bunch_spacing,
                                       (first_bunch + N_bunches) * bunch_spacing, bunch_spacing)])

for i in range(N_bunches):
    beam.dt[int(i * N_m):int((i + 1) * N_m)] = beam_single.dt + (i + first_bunch) * rfstation.t_rf[0, 0] * bunch_spacing
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
G_llrf_ls = [41.751786, 35.24865]

Commissioning = CavityFeedbackCommissioning_new(open_FF=True, debug=False)
OTFB = SPSCavityFeedback_new(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                              Commissioning=Commissioning, G_tx=G_tx_ls, G_llrf=G_llrf_ls)

print(OTFB.OTFB_1.V_set / 4, OTFB.OTFB_2.V_set / 2)

Ipeak = 2 * Proton().charge * 1.60218e-19 * N_b / OTFB.OTFB_1.T_s / 5

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


# Tracking with the beam
map_ = [profile] + [OTFB] + [SPS_tracker]
map_ = [OTFB]
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


# Compute power and beam.
OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
I_beam1 = np.max(np.real(OTFB.OTFB_1.I_COARSE_BEAM[-h:]))
I_beam1 = (I_beam1 / OTFB.OTFB_1.T_s / 5)
I_beam2 = np.max(np.real(OTFB.OTFB_2.I_COARSE_BEAM[-h:]))
I_beam2 = (I_beam2 / OTFB.OTFB_2.T_s / 5)

print(f'\nThe bunching factor is {I_beam1 / Ipeak}\n')

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


# Various plots and calculations ----------------------------------------------
power_c3 = OTFB.OTFB_1.P_GEN[-h::5]
power_c4 = OTFB.OTFB_2.P_GEN[-h::5]
x_ax = np.linspace(0, h, h//5) * rfstation.t_rf[0, 0]

PLOT_STATIC = True

if PLOT_STATIC:
    # Importing
    mat = scipy.io.loadmat('dataTimeDomainSim.mat')
    V3section = mat['V3section']
    V4section = mat['V4section']
    beamPhase = mat['beamPhase']
    hCav3Section = mat['hCav3Section']
    hCav4Section = mat['hCav4Section']
    transmitter3Section = mat['transmitter3Section']
    transmitter4Section = mat['transmitter4Section']

    plt.figure(1)
    plt.title('3-section')
    plt.plot(x_ax * 1e6, power_c3/1e6, label='BLonD', color='r')
    plt.plot(x_ax * 1e6, transmitter3Section/1e6, label='El.', color='b')
    plt.ylim(0, 1)
    plt.xlim(0, 7)
    plt.vlines(first_bunch * rfstation.t_rf[0,0] * bunch_spacing * 1e6, 0, 1, linestyle='--', color='g',
               label='first bunch')
    plt.vlines((first_bunch + 72) * rfstation.t_rf[0,0] * bunch_spacing * 1e6, 0, 1, linestyle='--', color='black',
               label='last bunch')
    plt.legend()
    plt.xlabel(f't [$\mu$s]')
    plt.ylabel(f'P [MW]')

    plt.figure(2)
    plt.title('4-section')
    plt.plot(x_ax * 1e6, power_c4/1e6, label='BLonD', color='r')
    plt.plot(x_ax * 1e6, transmitter4Section/1e6, label='El.', color='b')
    plt.ylim(0, 1.7)
    plt.xlim(0, 7)
    plt.vlines(first_bunch * rfstation.t_rf[0,0] * bunch_spacing * 1e6, 0, 1.7, linestyle='--', color='g',
               label='first bunch')
    plt.vlines((first_bunch + 72) * rfstation.t_rf[0,0] * bunch_spacing * 1e6, 0, 1.7, linestyle='--', color='black',
               label='last bunch')
    plt.legend()
    plt.xlabel(f't [$\mu$s]')
    plt.ylabel(f'P [MW]')

    # Phase deviation phase
    # V_ANT_batch = OTFB.OTFB_1.V_ANT[-h + first_bunch * bunch_spacing:-h + first_bunch * bunch_spacing + N_bunches * bunch_spacing:bunch_spacing]
    # V_IND_COARSE_BEAM_batch = OTFB.OTFB_1.V_IND_COARSE_BEAM[-h + first_bunch * bunch_spacing:-h + first_bunch * bunch_spacing + N_bunches * bunch_spacing:bunch_spacing]
    # phase = np.angle(V_ANT_batch, deg=True) - np.angle(V_IND_COARSE_BEAM_batch, deg=True)
    #
    # plt.figure(3)
    # plt.title(f'Static Beam - Phase Deviation')
    # plt.scatter(np.linspace(0, N_bunches, N_bunches), phase, marker='.',
    #             label='phase')
    # plt.scatter(np.linspace(0, N_bunches, N_bunches), np.angle(V_ANT_batch, deg=True), marker='.',
    #             label=r'$V_{ant}$')
    # plt.scatter(np.linspace(0, N_bunches, N_bunches), np.angle(V_IND_COARSE_BEAM_batch, deg=True), marker='.',
    #             label=r'$V_{ind,beam}$')
    # plt.legend()
    # plt.ylabel(f'Phase [Degrees]')
    # plt.xlabel(f'Bunch number [-]')
    # plt.show()

    phi_beam = pd.beam_phase_multibunch(profile, rfstation, OTFB, N_bunches, rfstation.t_rf[0,0],
                                        bunch_pos)

    phi_beam = (180/np.pi) * (phi_beam - np.mean(phi_beam))
    beamPhase = beamPhase[~np.isnan(beamPhase)]
    beamPhase = beamPhase - np.mean(beamPhase)

    plt.figure(3)
    plt.title('Beam Phase')
    plt.plot(phi_beam, color='r', label='BLonD')
    plt.plot(beamPhase, color='b', label='El.')
    plt.ylabel(r'$\varphi_{\textrm{rf}}$ [degrees]')
    plt.xlabel(r'Bunch number [-]')
    plt.legend()
    plt.show()

# Calculate the phase deviation along the batch
prof = profile.n_macroparticles
dt = profile.bin_centers

peaks, peaks_t = pd.get_peaks(prof, dt)
n_bunch = np.linspace(0, N_bunches, N_bunches)

peaks_t = peaks_t - peaks_t[0]

slope, intercept, r_val, p_val, stderr = spst.linregress(n_bunch, peaks_t)

x__ = np.linspace(0, N_bunches, N_bunches)

regcurv = slope * x__ + intercept

diff = peaks_t - regcurv

np.save('deviations', diff)

print('Done!')