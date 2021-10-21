

# Imports ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut

from blond.llrf.cavity_feedback import SPSOneTurnFeedback, CavityFeedbackCommissioning
from blond.llrf.new_SPS_OTFB import SPSOneTurnFeedback_new, CavityFeedbackCommissioning_new
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_m = int(1e5)                                  # Number of macro-particles for tracking
N_b = 1.0e11                                    # Bunch intensity [ppb]
N_t = 1                                         # Number of turns to track


# Options ---------------------------------------------------------------------
PLOT_EVERYTHING = False
ASYMPTOTIC_STABILITY = True
MODULATION_TO_FR = False
V_SET_MODULATION = False


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

# Modulated V_SET for the OTFB
V_SET = ut.make_step(h,  int(h/8), V * 4/9, h - int(h/8))
V_SET = np.concatenate((V_SET, V_SET))

# Cavity
Commissioning_old = CavityFeedbackCommissioning()
Commissioning_new = CavityFeedbackCommissioning_new(V_SET=None)

OTFB_old = SPSOneTurnFeedback(rfstation, beam, profile, 3, a_comb=63/64,
                              Commissioning=Commissioning_old)
OTFB_new = SPSOneTurnFeedback_new(rfstation, beam, profile, 3, a_comb=63/64,
                                  Commissioning=Commissioning_new)

# Tracking the two versions of the cavity
N_no_beam = 1000

# This part of the code plots all the different signals in both the new SPS OTFB and the old.
if PLOT_EVERYTHING:
    # Initialize arrays for the old OTFB such that it can be plotted over two turns like the new OTFB.
    DV_GEN_old = np.zeros(2*h, dtype=complex)
    V_SET_old = np.zeros(2*h, dtype=complex)
    V_ANT_old = np.zeros(2*h, dtype=complex)
    DV_COMB_OUT_old = np.zeros(2*h, dtype=complex)
    DV_DELAYED_old = np.zeros(2*h, dtype=complex)
    DV_MOV_AVG_old = np.zeros(2*h, dtype=complex)
    I_GEN_old = np.zeros(2*h, dtype=complex)
    V_IND_COARSE_GEN_old = np.zeros(2*h, dtype=complex)
    DV_MOD_FR_old = np.zeros(2*h, dtype=complex)
    DV_MOD_FRF_old = np.zeros(2*h, dtype=complex)
    x_a = np.array(range(2 * h))

    for i in range(N_no_beam):
        OTFB_old.track_no_beam()
        OTFB_new.track_no_beam()
        # TODO: Moving average and tau of TWC
        # Make lots of different plots here
        DV_GEN_old = ut.update_signal_array(OTFB_old, DV_GEN_old, 'DV_GEN', h)
        V_SET_old = ut.update_signal_array(OTFB_old, V_SET_old, 'V_SET', h)
        V_ANT_old = ut.update_signal_array(OTFB_old, V_ANT_old, 'V_ANT', h)
        DV_COMB_OUT_old = ut.update_signal_array(OTFB_old, DV_COMB_OUT_old, 'DV_COMB_OUT', h)
        DV_DELAYED_old = ut.update_signal_array(OTFB_old, DV_DELAYED_old, 'DV_DELAYED', h)
        DV_MOV_AVG_old = ut.update_signal_array(OTFB_old, DV_MOV_AVG_old, 'DV_MOV_AVG', h)
        I_GEN_old = ut.update_signal_array(OTFB_old, I_GEN_old, 'I_GEN', h)
        V_IND_COARSE_GEN_old = ut.update_signal_array(OTFB_old, V_IND_COARSE_GEN_old, 'V_IND_COARSE_GEN', h)
        DV_MOD_FR_old = ut.update_signal_array(OTFB_old, DV_MOD_FR_old, 'DV_MOD_FR', h)
        DV_MOD_FRF_old = ut.update_signal_array(OTFB_old, DV_MOD_FRF_old, 'DV_MOD_FRF', h)

        # Print Stats
        ut.print_stats(OTFB_old, OTFB_new)
        print(f"Turn: {i}")

        # Plot all the arrays and compare
        plt.title("error and gain")
        plt.plot(x_a, np.abs(OTFB_new.DV_GEN), label="DV_GEN", color="r")
        plt.plot(x_a, np.abs(OTFB_new.V_SET), label="V_SET", color="b")
        plt.plot(x_a, np.abs(OTFB_new.V_ANT_START), label="V_ANT", color='g')
        plt.plot(x_a, np.abs(DV_GEN_old), label="DV_GEN_old", linestyle="--", color="r")
        plt.plot(x_a, np.abs(V_SET_old), label="V_SET_old", linestyle="--", color="b")
        plt.plot(x_a, np.abs(V_ANT_old), label="V_ANT_old", linestyle="--", color="g")
        plt.legend()
        plt.show()

        plt.title("Comb")
        plt.plot(x_a, np.abs(OTFB_new.DV_GEN), label="DV_GEN", color="r")
        plt.plot(x_a, np.abs(OTFB_new.DV_COMB_OUT), label="DV_COMB_OUT", color="b")
        plt.plot(x_a, np.abs(DV_GEN_old), label="DV_GEN_old", color="r", linestyle="--")
        plt.plot(x_a, np.abs(DV_COMB_OUT_old), label="DV_COMB_OUT_old", color="b", linestyle="--")
        plt.legend()
        plt.show()

        plt.title("one turn delay")
        plt.plot(x_a, np.abs(OTFB_new.DV_COMB_OUT), label="DV_COMB_OUT", color="r")
        plt.plot(x_a, np.abs(OTFB_new.DV_DELAYED), label="DV_DELAYED", color="b")
        plt.plot(x_a, np.abs(DV_COMB_OUT_old), label="DV_COMB_OUT_old", linestyle="--", color="r")
        plt.plot(x_a, np.abs(DV_DELAYED_old), label="DV_DELAYED_old", linestyle="--", color="b")
        plt.legend()
        plt.show()

        plt.title("mov avg")
        plt.plot(x_a, np.abs(OTFB_new.DV_MOV_AVG), label="DV_MOV_AVG", color="r")
        plt.plot(x_a, np.abs(OTFB_new.DV_MOD_FR), label="DV_MOD_FR", color="b")
        plt.plot(x_a, np.abs(DV_MOV_AVG_old), label="DV_MOV_AVG_old", color="r", linestyle="--")
        plt.plot(x_a, np.abs(DV_MOD_FR_old), label="DV_MOD_FR_old", color="b", linestyle="--")
        plt.legend()
        plt.show()

        plt.title("sum and gain")
        plt.plot(x_a, np.abs(OTFB_new.DV_MOD_FRF), label="DV_MOD_FRF", color="r")
        #plt.plot(x_a, np.abs(OTFB_new.V_SET), label="V_SET")
        plt.plot(x_a, np.abs(OTFB_new.I_GEN), label="I_GEN", color="b")
        plt.plot(x_a, np.abs(DV_MOD_FRF_old), label="DV_MOD_FRF_old", color="r", linestyle="--")
        #plt.plot(x_a, np.abs(V_SET_old), label="V_SET_old")
        plt.plot(x_a, np.abs(I_GEN_old), label="I_GEN_old", color="b", linestyle="--")
        plt.legend()
        plt.show()

        plt.subplot(211)
        plt.title("gen response")
        plt.plot(x_a, np.abs(OTFB_new.V_IND_COARSE_GEN), label="V_IND_COARSE_GEN")
        plt.plot(x_a, np.abs(V_IND_COARSE_GEN_old), label="V_IND_COARSE_GEN_old")
        plt.legend()
        plt.subplot(212)
        plt.plot(x_a, np.abs(OTFB_new.I_GEN), label="I_GEN")
        plt.plot(x_a, np.abs(I_GEN_old), label="I_GEN_old")
        plt.legend()
        plt.show()


if ASYMPTOTIC_STABILITY:
    # Initialize arrays to plot the old OTFB with the new.
    V_ANT_old = np.zeros(2*h, dtype=complex)
    x_a = np.array(range(2 * h))

    # Plots for every
    dt_plot = 50

    for i in range(N_no_beam):
        # Track the OTFBs
        OTFB_old.track_no_beam()
        OTFB_new.track_no_beam()

        # Update the array
        V_ANT_old = ut.update_signal_array(OTFB_old, V_ANT_old, 'V_ANT', h)

        if i % dt_plot == 0:
            # Print Stats
            ut.print_stats(OTFB_old, OTFB_new)
            print(f"Turn: {i}")
            print(f"Maximum deviation from mean: {ut.ripple_calculation(np.abs(OTFB_new.V_ANT_START))}")
            print(f"Maximum deviation from mean: {ut.ripple_calculation(np.abs(V_ANT_old))}")

            plt.title("Antenna Voltage")
            plt.plot(x_a, np.abs(V_ANT_old), label='V_ANT_old')
            plt.plot(x_a, np.abs(OTFB_new.V_ANT_START), label='V_ANT_new')
            plt.legend()
            plt.show()


if MODULATION_TO_FR:
    OTFB_new.track_no_beam()

    plt.plot(OTFB_new.DV_DELAYED.real, color='r', linestyle='--')
    plt.plot(OTFB_new.DV_DELAYED.imag, color='b', linestyle='--')
    plt.plot(OTFB_new.DV_MOD_FR.real, color='r')
    plt.plot(OTFB_new.DV_MOD_FR.imag, color='b')
    plt.show()


if V_SET_MODULATION:
    dt_plot = 1
    for i in range(N_no_beam):
        OTFB_new.track_no_beam()

        if i % dt_plot == 0:
            print(f"Turn: {i}")
            ut.plot_everything(OTFB_new)