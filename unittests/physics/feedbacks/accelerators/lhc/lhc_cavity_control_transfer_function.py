import beam_dynamics_tools.analytical_functions.transfer_functions as tf
import beam_dynamics_tools.data_visualisation.make_plots_pretty
import matplotlib.pyplot as plt
import numpy as np
from beam_dynamics_tools.analytical_functions.mathematical_functions import (
    to_dB,
    to_linear,
)

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)
from blond.llrf.transfer_function import TransferFunction

# Constants
N_m = 50000  # Macro-particles
NB = 144  # Number of bunches
C = 26658.8832  # Machine circumference [m]
p_s = 450e9  # Synchronous momentum [eV/c]
V = 1e6
h = 35640  # Harmonic number
dphi = 0  # Phase modulation/offset
R_over_Q = 45  # Cavity R/Q [Ohms]
gamma_t = 53.8  # Transition gamma
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

G_a = 6.79e-6  # Analog FB gain [A/V]
G_d = 10  # Digital FB gain [-]
tau_loop = 650e-9  # Overall loop delay [s]
tau_a = 170e-6  # Analog FB delay [s]
tau_d = 400e-6  # Digital FB delay [s]
a_comb = 15 / 16  # Comb filter alpha [-]
Q_L = 20000  # Loaded Quality factor [-]
G_otfb = 10
tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
G_gen = 1
tau_o = 110e-6


ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
rf = RFStation(ring, [h], [V], [dphi])
beam = Beam(ring, 1, 1)
profile = Profile(
    beam,
    CutOptions=CutOptions(
        cut_left=2 * rf.t_rf[0, 0], cut_right=3 * rf.t_rf[0, 0], n_slices=64
    ),
)

print("Measuring the open-loop transfer function...")
# Open-loop transfer function measurement
commissioning = LHCCavityLoopCommissioning(
    alpha=a_comb,
    d_phi_ad=0,
    G_a=G_a,
    G_d=G_d,
    G_o=G_otfb,
    tau_a=tau_a,
    tau_d=tau_d,
    tau_o=tau_o,
    open_drive=False,
    open_loop=True,
    open_otfb=True,
    open_rffb=False,
    open_tuner=True,
    full_detuning=False,
    excitation=True,
    enable_klystron=False,
)

cavity_feedback = LHCCavityLoop(
    rf,
    profile,
    n_cavities=1,
    f_c=rf.omega_rf[0, 0] / (2 * np.pi),
    G_gen=G_gen,
    n_pretrack=200,
    Q_L=Q_L,
    R_over_Q=R_over_Q,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    RFFB=commissioning,
)

n_turns_excite = 200
cavity_feedback.track_no_beam_excitation(n_turns_excite)

transfer_function = TransferFunction(
    cavity_feedback.V_EXC_IN,
    cavity_feedback.V_EXC_OUT,
    T_s=cavity_feedback.T_s,
)

transfer_function.analyse(3564 * 5)

H_est_open_loop = transfer_function.H_est
freq_est_open_loop = transfer_function.f_est


print("Measuring the closed-loop transfer function...")
# Closed-loop transfer function measurement
commissioning = LHCCavityLoopCommissioning(
    alpha=a_comb,
    d_phi_ad=0,
    G_a=G_a,
    G_d=G_d,
    G_o=G_otfb,
    tau_a=tau_a,
    tau_d=tau_d,
    tau_o=tau_o,
    open_drive=False,
    open_loop=False,
    open_otfb=True,
    open_rffb=False,
    open_tuner=True,
    full_detuning=False,
    excitation=True,
    enable_klystron=False,
)

cavity_feedback = LHCCavityLoop(
    rf,
    profile,
    n_cavities=1,
    f_c=rf.omega_rf[0, 0] / (2 * np.pi),
    G_gen=G_gen,
    n_pretrack=200,
    Q_L=Q_L,
    R_over_Q=R_over_Q,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    RFFB=commissioning,
)

n_turns_excite = 200
cavity_feedback.track_no_beam_excitation(n_turns_excite)

transfer_function = TransferFunction(
    cavity_feedback.V_EXC_IN,
    cavity_feedback.V_EXC_OUT,
    T_s=cavity_feedback.T_s,
)

transfer_function.analyse(3564 * 5)

H_est_closed_loop = transfer_function.H_est
freq_est_closed_loop = transfer_function.f_est


print("Measuring the closed-loop transfer function with otfb...")
# Closed-loop transfer function with otfb measurement
commissioning = LHCCavityLoopCommissioning(
    alpha=a_comb,
    d_phi_ad=0,
    G_a=G_a,
    G_d=G_d,
    G_o=G_otfb,
    tau_a=tau_a,
    tau_d=tau_d,
    tau_o=tau_o,
    open_drive=False,
    open_loop=False,
    open_otfb=False,
    open_rffb=False,
    open_tuner=True,
    full_detuning=False,
    excitation=True,
    enable_klystron=False,
)

cavity_feedback = LHCCavityLoop(
    rf,
    profile,
    n_cavities=1,
    f_c=rf.omega_rf[0, 0] / (2 * np.pi),
    G_gen=G_gen,
    n_pretrack=200,
    Q_L=Q_L,
    R_over_Q=R_over_Q,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    RFFB=commissioning,
)

n_turns_excite = 200
cavity_feedback.track_no_beam_excitation(n_turns_excite)

transfer_function = TransferFunction(
    cavity_feedback.V_EXC_IN,
    cavity_feedback.V_EXC_OUT,
    T_s=cavity_feedback.T_s,
)

transfer_function.analyse(3564 * 5)

H_est_full_loop = transfer_function.H_est
freq_est_full_loop = transfer_function.f_est


fig, ax = plt.subplots(nrows=2, figsize=(10, 8), sharex="all")

ax[0].plot(
    freq_est_open_loop / 1e3,
    np.abs(H_est_open_loop),
    lw=0.5,
    label="Open Loop",
)
ax[0].plot(
    freq_est_closed_loop / 1e3,
    np.abs(H_est_closed_loop),
    lw=0.5,
    label="Closed Loop",
)
ax[0].plot(
    freq_est_full_loop / 1e3,
    np.abs(H_est_full_loop),
    lw=0.5,
    label="Full Loop",
)

ax[0].set_ylim(1e-2, 3)
ax[0].set_yscale("log")
ax[0].grid()
ax[0].legend()
ax[0].set_ylabel("Amplitude [-]")

ax[1].plot(
    freq_est_open_loop / 1e3,
    np.angle(H_est_open_loop, deg=True),
    lw=0.5,
    linestyle="dashed",
)
ax[1].plot(
    freq_est_closed_loop / 1e3,
    np.angle(H_est_closed_loop, deg=True),
    lw=0.5,
    linestyle="dashed",
)
ax[1].plot(
    freq_est_full_loop / 1e3, np.angle(H_est_full_loop, deg=True), lw=0.5
)

ax[1].set_xlim(-750, 750)
ax[1].set_ylabel("Phase [deg.]")
ax[1].set_xlabel("Frequency [kHz]")
# ax.set_ylim(1e-2, 10)

ax[1].grid()

fig.tight_layout()

fig_scal = 0.75


plt.show()


np.savez(
    "generate_blond2_data/feedbacks/lhc/data/lhc_cavity_control_transfer_function_freq.npz",
    open_loop_transfer_function=H_est_open_loop,
    closed_loop_transfer_function=H_est_closed_loop,
    full_loop_transfer_function=H_est_full_loop,
    open_loop_freq=freq_est_open_loop,
    closed_loop_freq=freq_est_closed_loop,
    full_loop_freq=freq_est_full_loop,
)
