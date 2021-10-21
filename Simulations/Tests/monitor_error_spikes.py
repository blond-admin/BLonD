

# Imports ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
import matplotlib.animation as animation

from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new, CavityFeedbackCommissioning_new
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.llrf.signal_processing import modulator, moving_average


# Parameters ------------------------------------------------------------------
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


# Objects ---------------------------------------------------------------------

# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h], [V], [phi], n_rf=1)

# Beam
beam = Beam(SPS_ring, N_m, N_b)
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=4620))
profile.track()

# Modulated V_SET for the OTFB for different times in a turn
SMOOTH = True

slope = int(h/10)
l = 2 * slope + 200
h0s = [h-l, h-l-int(h/16), int(h/2)]

if SMOOTH:
    V_SET_1 = ut.make_smooth_step(h, l, V * 4 / 9, slope, slope, h0s[0])
    sil = np.copy(V_SET_1)
    V_SET_1 = np.concatenate((V_SET_1, V_SET_1))

    V_SET_2 = ut.make_smooth_step(h, l, V * 4 / 9, slope, slope, h0s[1])
    V_SET_2 = np.concatenate((V_SET_2, V_SET_2))

    V_SET_3 = ut.make_smooth_step(h, l, V * 4 / 9, slope, slope, h0s[2])
    V_SET_3 = np.concatenate((V_SET_3, V_SET_3))
else:
    V_SET_1 = ut.make_step(h,  l, V * 4/9, h0s[0])
    V_SET_1 = np.concatenate((V_SET_1, V_SET_1))

    V_SET_2 = ut.make_step(h,  l, V * 4/9, h0s[1])
    V_SET_2 = np.concatenate((V_SET_2, V_SET_2))

    V_SET_3 = ut.make_step(h,  l, V * 4/9, h0s[2])
    V_SET_3 = np.concatenate((V_SET_3, V_SET_3))

print(h0s)
print(l)

# Cavity
Commissioning_new_1 = CavityFeedbackCommissioning_new(V_SET=V_SET_1)
Commissioning_new_2 = CavityFeedbackCommissioning_new(V_SET=V_SET_2)
Commissioning_new_3 = CavityFeedbackCommissioning_new(V_SET=V_SET_3)


OTFB_new_1 = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                                  Commissioning=Commissioning_new_1)
OTFB_new_2 = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                                  Commissioning=Commissioning_new_2)
OTFB_new_3 = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                                  Commissioning=Commissioning_new_3)


# Tracking the two versions of the cavity
N_no_beam = 1000
er_window = h

OTFBs = [OTFB_new_1, OTFB_new_2, OTFB_new_3]
RMS_error_array = np.zeros((N_no_beam, 6, len(OTFBs) - 1))
ERROR_TURN_ARRAY = np.zeros((er_window, N_no_beam))

dt_plot = 100
PLOT_TRACK = False
PLOT_SPIKE = False
PLOT_SPIKE_ALL = False

for i in range(N_no_beam):
    PLOT = False
    for O in OTFBs:
        O.track_no_beam()

    if i % dt_plot == 0 and PLOT_TRACK:
        print(f"Turn: {i}")
        PLOT = True
        ut.plot_cont_everything(OTFBs, h0s)

    if i % dt_plot == 0 and not PLOT_TRACK:
        print(f'Trun: {i}')

    RMS_error_array[i,:,:] = ut.calc_rms_error_module(OTFBs, h0s, l, er_window, PLOT)
    ind_OTFB = 1
    ERROR_TURN_ARRAY[:,i] = ut.difference_array(OTFBs[ind_OTFB], OTFBs[-1], h0s[ind_OTFB], h0s[-1],
                                                l, er_window, 'I_GEN')

    SPIKE = ut.find_spike(RMS_error_array[:,0,1], i, 25, 100)

    if SPIKE and PLOT_SPIKE:
        print(f"Turn: {i}")
        ut.plot_spikes(OTFBs, h0s, l, er_window, PLOT_SPIKE_ALL)

ut.plot_errors(RMS_error_array)

# ANIMATION -------------------------------------------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plot_scale = 5.0e-14
sil = sil / np.max(sil)
sil = plot_scale * np.abs(sil)

def animate_ERROR_TURN(i):
    ax1.clear()
    ax1.plot(ERROR_TURN_ARRAY[:,i])
    ax1.plot(sil, linestyle='dotted')
    ax1.set_ylim(0, plot_scale)

    plt.title(f'I_GEN, turn {i}')


ani = animation.FuncAnimation(fig, animate_ERROR_TURN, interval=50)
plt.show()