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
V = 4.4e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]
bl = 2.5e-9                                     # Bunch length [s]

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
OTFB = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                              Commissioning=Commissioning, V_part=1)


# Simulation ------------------------------------------------------------------
print('Simulating...\n')

jet = plt.get_cmap('jet')
colors = jet(np.linspace(0,1,N_t))

# Pre-tracking
for i in range(N_pretrack):
    OTFB.track_no_beam()

# Plotting set up
PLOT_JET = True
if PLOT_JET:
    fig = plt.figure(1)
    gs1 = gridspec.GridSpec(2, 1)
    ax1_1 = plt.subplot(gs1[0])
    ax1_2 = plt.subplot(gs1[1], sharex=ax1_1)
    plt.setp(ax1_1.get_xticklabels(), visible=False)

    yticks = ax1_1.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    ax1_1.set_ylabel(r"$Abs(V_{\mathsf{cav}})$ [MV]")
    ax1_2.set_xlabel(r"Bucket [-]")
    ax1_2.set_ylabel(r"$Arg(V_{\mathsf{cav}})$ [MV]")

    fig2 = plt.figure(2)
    gs2 = gridspec.GridSpec(2, 1)
    ax2_1 = plt.subplot(gs2[0])
    ax2_2 = plt.subplot(gs2[1], sharex=ax2_1)
    plt.setp(ax2_1.get_xticklabels(), visible=False)
    yticks = ax2_1.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    ax2_1.set_ylabel(r"$Abs(I_{\mathsf{gen}})$ [MV]")
    ax2_2.set_xlabel(r"Buckets [$\mu$s]")
    ax2_2.set_ylabel(r"$Arg(I_{\mathsf{gen}})$ [MV]")

    fig3 = plt.figure(3)
    gs3 = gridspec.GridSpec(2,1)
    ax3_1 = plt.subplot(gs3[0])
    ax3_2 = plt.subplot(gs3[1], sharex=ax3_1)
    plt.setp(ax3_1.get_xticklabels(), visible=False)
    yticks = ax3_1.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    ax3_1.set_ylabel(r"$Abs(V_{\mathsf{ind,beam}})$ [MV]")
    ax3_2.set_xlabel(r"Buckets [$\mu$s]")
    ax3_2.set_ylabel(r"$Arg(I_{\mathsf{ind,beam}})$ [MV]")



# Tracking with the beam
map_ = [profile] + [OTFB] + [SPS_tracker]
dt_plot = 10
dt_track = 10

for i in range(N_t):
    for m in map_:
        m.track()

    OTFB.calc_power()

    if i % dt_plot == 0 and not i == 0:
        print(f'Turn: {i}')
        if not PLOT_JET:
            ut.plot_beam_test_full_machine(OTFB)
            ut.plot_IQ_full_machine(OTFB)

            plt.plot(OTFB.P_GEN)
            plt.show()

    if i % dt_track == 0 and not i % dt_plot == 0:
        print(f'Turn: {i}')

    # Jet plots:
    if PLOT_JET:
        ax1_1.plot(np.abs(OTFB.V_ANT[-h:]), color=colors[i])
        ax1_2.plot(np.angle(OTFB.V_ANT[-h:]), color=colors[i])

        ax2_1.plot(np.abs(OTFB.I_GEN[-h:] / OTFB.T_s), color=colors[i])
        ax2_2.plot(np.angle(OTFB.I_GEN[-h:] / OTFB.T_s), color=colors[i])

        ax3_1.plot(np.abs(OTFB.V_IND_COARSE_BEAM[-h:]), color=colors[i])
        ax3_2.plot(np.angle(OTFB.V_IND_COARSE_BEAM[-h:]), color=colors[i])

if PLOT_JET:
    plt.show()

    plt.title('Power')
    plt.plot(OTFB.P_GEN[-h:])
    plt.show()

