import logging
import time
from math import tau

import matplotlib.pyplot as plt
import numpy as np

from blond.legacy.blond2.beam.beam import Beam, Proton
from blond.legacy.blond2.beam.distributions import (
    bigaussian,
    matched_from_distribution_function,
)
from blond.legacy.blond2.beam.profile import CutOptions, FitOptions, Profile
from blond.legacy.blond2.impedances.impedance_sources import Resonators
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.legacy.blond2.llrf.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)
from blond.legacy.blond2.toolbox.logger import Logger
from blond.legacy.blond2.trackers.tracker import (
    FullRingAndRF,
    RingAndRFTracker,
)


def optimum_ql(V, I_rf, R_over_Q=45):
    return 2 * V / (R_over_Q * I_rf)


def P_avgopt(Q_L, V, I_rf, R_over_Q=45):
    return (1 / 8) * V**2 / (R_over_Q * Q_L) + (
        1 / 32
    ) * R_over_Q * Q_L * I_rf**2


voltages_tot = 7.9e6
intensities = 2.3e11
loaded_q = 20000
bunch_lengths = 1.25e-9


# Constants
n_macroparticles = int(1e6)  # Macro-particles
n_bunches = 2  # Number of bunches
C = 26658.8832  # Machine circumference [m]
p_s = 450e9  # Synchronous momentum [eV/c]
h = 35640  # Harmonic number
dphi = 0  # Phase modulation/offset
R_over_Q = 45  # Cavity R/Q [Ohms]
gamma_t = 53.8  # Transition gamma
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

n_turns = 100

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


ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=n_turns)
rf = RFStation(
    ring, [h], [voltages_tot], [dphi]
)  # Assume filamented with SPS emittance
bunch = Beam(ring, n_macroparticles, intensities)

print(rf.energy[0])  # 450000978170.6162
print(ring.eta_0[0, 0])  # 0.00034114256309499084
print(rf.beta[0])  # 0.9999978262922445
print(rf.phi_s[0])  # 3.141592653589793

n_slices = 2**7
profile = Profile(
    bunch,
    CutOptions(n_slices=n_slices, cut_left=0, cut_right=rf.t_rf[0, 0]),
    FitOptions=FitOptions(fit_option="rms"),
)
bigaussian(ring, rf, bunch, sigma_dt=bunch_lengths / 4, seed=1234)

# tracker = RingAndRFTracker(rf, bunch, Profile=profile)
# tracker = FullRingAndRF([tracker])

# matched_from_distribution_function(bunch, tracker,
#    distribution_type = 'binomial', bunch_length = bunch_lengths, distribution_exponent = 1.5,
#    distribution_variable = 'Hamiltonian', bunch_length_fit = 'fwhm',
#    n_iterations=1
# )


beam = Beam(ring, n_macroparticles * n_bunches, intensities * n_bunches)
buckets = rf.t_rf[0, 0] * 10

for i in range(n_bunches):
    beam.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
        bunch.dt[0:n_macroparticles] + 100 * buckets + 10 * rf.t_rf[0, 0] * i
    )
    beam.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = bunch.dE[
        0:n_macroparticles
    ]

profile = Profile(
    beam,
    CutOptions(
        n_slices=int(2**6 * (10 * n_bunches + 10)),
        cut_left=(1000 - 5) * rf.t_rf[0, 0],
        cut_right=(1000 + 10 * n_bunches + 5) * rf.t_rf[0, 0],
    ),
)
# print(buckets)
# print(100 * buckets - 5 * rf.t_rf[0, 0], 100 * buckets + (10 * n_bunches + 5) * rf.t_rf[0, 0])
profile.track()

np.savez(
    "lhc_36bunches_7.9MV.npz",
    dE=beam.dE,
    dt=beam.dt,
)


RFFB = LHCCavityLoopCommissioning(
    G_a=G_a,
    G_d=G_d,
    tau_d=tau_d,
    tau_a=tau_a,
    alpha=a_comb,
    tau_o=tau_o,
    open_otfb=False,
    G_o=G_otfb,
    mu=-20,
    open_tuner=True,
    d_phi_ad=0,
)

CL = LHCCavityLoop(
    rf_station=rf,
    profile=profile,
    f_c=rf.omega_rf[0, 0] / (2 * np.pi) - 5e3,
    I_gen_offset=0,
    n_cavities=8,
    n_pretrack=100,
    Q_L=Q_L,
    R_over_Q=R_over_Q,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    G_gen=G_gen,
    RFFB=RFFB,
)


tracker = RingAndRFTracker(rf, beam, Profile=profile, CavityFeedback=CL)
tracker = FullRingAndRF([tracker])

rf_power = np.zeros((n_turns, CL.n_coarse), dtype=complex)
i_beam = np.zeros((n_turns, CL.n_coarse), dtype=complex)
rf_voltage = np.zeros((n_turns, CL.n_coarse), dtype=complex)
line_density = np.zeros((n_turns, profile.n_slices))

print()

for i in range(n_turns):
    profile.track()
    tracker.track()
    rf_power[i, :] = CL.generator_power()[-CL.n_coarse :]
    i_beam[i, :] = CL.I_BEAM_COARSE[-CL.n_coarse :]
    rf_voltage[i, :] = CL.V_ANT_COARSE[-CL.n_coarse :]
    line_density[i, :] = profile.n_macroparticles


# fig, ax = plt.subplots()
#
# ax.plot(np.abs(rf_power).T)

[plt.plot(profile) for profile in i_beam]
plt.show()

np.savez(
    "lhc_cavity_control_injection_power_no_bpl.npz",
    rf_power=rf_power,
    i_beam=i_beam,
    rf_voltage=rf_voltage,
    line_density=line_density,
)
