

# Imports ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut

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


# Options ---------------------------------------------------------------------
RUN_PLOTS = True
RUN_PLOTS_REAL_IMAG = False
MOD_SQUARES = False
MOV_AVG_TEST = False
IN_OUT_MOV_AVG = False

'''
Notes: Check what comes in the moving average and what comes out for the different pulse-placements.
'''


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
h0s = [h-int(h/8), h-int(h/8)-int(h/16), int(h/2)]
V_SET_1 = ut.make_step(h,  int(h/8), V * 4/9, h0s[0])
V_SET_1 = np.concatenate((V_SET_1, V_SET_1))

V_SET_2 = ut.make_step(h,  int(h/8), V * 4/9, h0s[1])
V_SET_2 = np.concatenate((V_SET_2, V_SET_2))

V_SET_3 = ut.make_step(h,  int(h/8), V * 4/9, h0s[2])
V_SET_3 = np.concatenate((V_SET_3, V_SET_3))

print(h0s)
print(int(h/8))

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

OTFBs = [OTFB_new_1, OTFB_new_2, OTFB_new_3]
RMS_error_array = np.zeros((N_no_beam, 6, len(OTFBs) - 1))

dt_plot = 50
PLOT_TRACK = False
if RUN_PLOTS:
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

        RMS_error_array[i,:,:] = ut.calc_rms_error_module(OTFBs, h0s, int(h/8), h, PLOT)

    ut.plot_errors(RMS_error_array)


if RUN_PLOTS_REAL_IMAG:
    for i in range(N_no_beam):
        OTFB_new_1.track_no_beam()

        if i % dt_plot == 0:
            print(f"Turn: {i}")
            ut.plot_everything_real_imag(OTFB_new_1)


if MOD_SQUARES:
    start_square = ut.make_step(h, int(h/8), 1, 0)
    mid_square = ut.make_step(h, int(h / 8), 1, int(h/2))

    start_square = start_square + 1j * 0
    mid_square = mid_square + 1j * 0

    start_square = modulator(start_square, OTFB_new_1.omega_c, OTFB_new_1.omega_r, OTFB_new_1.rf.t_rf[0, 0])
    mid_square = modulator(mid_square, OTFB_new_1.omega_c, OTFB_new_1.omega_r, OTFB_new_1.rf.t_rf[0, 0])

    start_square = modulator(start_square, OTFB_new_1.omega_r, OTFB_new_1.omega_c, OTFB_new_1.rf.t_rf[0, 0])
    mid_square = modulator(mid_square, OTFB_new_1.omega_r, OTFB_new_1.omega_c, OTFB_new_1.rf.t_rf[0, 0])

    plt.plot(np.roll(np.abs(start_square), 0))
    plt.plot(np.roll(np.abs(mid_square), -int(h/2)))
    plt.show()


if MOV_AVG_TEST:

    MOD = True
    n_shift = 0
    n_mov_av = 92
    MOV_AVG1 = np.zeros(2 * h, dtype=complex)
    MOV_AVG2 = np.zeros(2 * h, dtype=complex)

    sig1 = ut.make_step(2*h, int(h/8), 1, -h)
    sig2 = ut.make_step(2*h, int(h/8), 1, -h + n_shift)

    if MOD:
        sig1[-h:] = modulator(sig1[-h:], OTFB_new_1.omega_c, OTFB_new_1.omega_r, OTFB_new_1.rf.t_rf[0, 0])
        sig2[-h:] = modulator(sig2[-h:], OTFB_new_1.omega_c, OTFB_new_1.omega_r, OTFB_new_1.rf.t_rf[0, 0])

    plt.plot(np.abs(sig1), label='sig1')
    plt.plot(np.abs(sig2), label='sig2')
    plt.legend()
    plt.show()

    MOV_AVG1[-h:] = moving_average(sig1[-n_shift -n_mov_av -1 - h:], n_mov_av)[-h:]
    MOV_AVG2[-h + n_mov_av - 1:] = moving_average(sig2[-h:], n_mov_av)

    plt.plot(np.abs(MOV_AVG1), label='MOV_AVG1')
    plt.plot(np.abs(MOV_AVG2), label='MOV_AVG2')
    plt.legend()
    plt.show()

    plt.plot(np.abs(MOV_AVG1), label='MOV_AVG1')
    plt.plot(np.roll(np.abs(MOV_AVG2), -n_shift), label='MOV_AVG2')
    plt.legend()
    plt.show()


if IN_OUT_MOV_AVG:
    RMS_error_array = np.zeros((N_no_beam, 2, len(OTFBs) - 1))
    for i in range(N_no_beam):
        PLOT = False
        for O in OTFBs:
            O.track_no_beam()

        if i % dt_plot==0:
            print(f'Turn: {i}')
            PLOT = True

        RMS_error_array[i,:,:] = ut.in_out_mov_avg_calc(OTFBs, h0s, 4000, 2000, PLOT=PLOT)

    plt.figure(1)
    plt.title('Error in DV_MOD_FR')
    for i in range(len(OTFBs) - 1):
        plt.plot(RMS_error_array[:,0,i], label=str(h0s[i]))
    plt.legend()

    plt.figure(2)
    plt.title('Error in DV_MOV_AVG')
    for i in range(len(OTFBs) - 1):
        plt.plot(RMS_error_array[:, 1, i], label=str(h0s[i]))
    plt.legend()
    plt.show()
