import cProfile
import os
import pstats

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from tqdm import tqdm

# import beam_dynamics_tools.data_visualisation.make_plots_pretty
plt.style.use("science")
from beam_dynamics_tools.beam_profiles.bunch_profile_tools import (
    get_beam_pattern,
)

# Import blond objects
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian, parabolic
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.tracker import RingAndRFTracker

# Options
PLT_SIMS = True
SAVE_SIM = True
DISABLE_PL = False

data_folder = "data/convergence_to_steadystate/"


# Initialize the accelerator

# The synchrotron ring
C = 26658.8832  # Machine circumference [m]
p_s = 450e9  # Synchronous momentum [eV/c]
gamma_t = 53.8  # Transition gamma [-]
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor [-]
n_turns = 2000  # Number of turns to track [-]

ring = Ring(C, alpha, p_s, Proton(), n_turns=n_turns + 1)

# The RF station
h = 35640  # Harmonic number [-]
V = 5e6  # RF voltage [V]
dphi = 0  # Phase modulation/offset [rad]

rfstation = RFStation(ring, [h], [V], [dphi], n_rf=1)


# The beam
number_of_bunches = 1  # Length of the batch [number of bunches]
bunch_intensity = 1.6e11  # Bunch intensity [p/b]
n_macroparticles = 1_000_000  # Number of macroparticles per bunch [-]
tau_bunch = 1.2e-9  # Bunch length [s]
bunch_spacing = 10  # Bunch spacing [number of rf buckets]
injection_energy_error = 0  # Injection energy error [eV]
injection_phase_error = 40
bucket_shift = 1000


# First generate a single gaussian bunch
beam = Beam(ring, n_macroparticles, bunch_intensity)
bigaussian(ring, rfstation, beam, sigma_dt=tau_bunch / 4, seed=1234)

# Add final corrections to the bunch positions
bucket_shift = 0
beam.dt += (
    bucket_shift * rfstation.t_rf[0, 0]
    + injection_phase_error * rfstation.t_rf[0, 0] / 360
)
beam.dE += injection_energy_error

# The beam profile
cut_options = CutOptions(
    cut_left=(-5.5 + bucket_shift) * rfstation.t_rf[0, 0],
    cut_right=(6.5 + number_of_bunches * bunch_spacing + bucket_shift)
    * rfstation.t_rf[
        0,
        0,
    ],
    n_slices=(10 * number_of_bunches + 12) * 2**5,
)
profile = Profile(beam, cut_options)

# Plot profile
profile.track()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(profile.bin_centers * 1e6, profile.n_macroparticles)
ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
ax.set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
ax.set_yticks([])

plt.show()

# Cavity Controller
G_a = 6.79e-6  # Analog FB gain [A/V]
G_d = 10  # Digital FB gain [-]
tau_loop = 650e-9  # Overall loop delay [s]
tau_a = 170e-6  # Analog FB delay [s]
tau_d = 400e-6  # Digital FB delay [s]
a_comb = 15 / 16  # Comb filter alpha [-]
Q_L = 20000  # Loaded Quality factor [-]
G_otfb = 10  # OTFB gain [-]
tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
delta_f = -3480  # Initial detuning due to 12 bunches [Hz]


# Beam-phase loop
# Beam Loops
PL_gain = 1 / (5 * ring.t_rev[0]) * int(not DISABLE_PL)
SL_gain = PL_gain / 10
bl_config = {"machine": "LHC", "PL_gain": PL_gain, "SL_gain": SL_gain}

beam_loop = BeamFeedback(ring, rfstation, profile, bl_config)

# The RF tracker
rftracker = RingAndRFTracker(
    rfstation,
    beam,
    Profile=profile,
    interpolation=True,
    BeamFeedback=beam_loop,
)


# Initialize data arrays
bunch_length = np.zeros(n_turns)
bunch_length_spread = np.zeros(n_turns)
beam_loop_error = np.zeros(n_turns)
synchro_loop_error = np.zeros(n_turns)

omega_rf = np.zeros(n_turns)
phi_rf = np.zeros(n_turns)

print(profile.bin_size * 1e12)

# if DISABLE_PL:
#    profile.track()
#    beam_loop.track()
# Tracking

for i in tqdm(range(n_turns)):
    profile.track()
    rftracker.track()

    bpos, blen, bpk, bpkpos, bint = get_beam_pattern(
        profile.n_macroparticles,
        profile.bin_centers,
        height_factor=100,
        distance=500,
        single_turn=True,
    )

    # print(beam_loop.lhc_y)

    bunch_length[i] = np.mean(blen)
    bunch_length_spread[i] = np.std(blen)
    beam_loop_error[i] = beam_loop.dphi * 180 / np.pi
    synchro_loop_error[i] = rfstation.dphi_rf * 180 / np.pi
    omega_rf[i] = rfstation.omega_rf[0, i]
    phi_rf[i] = rfstation.phi_rf[0, i]


print(beam_loop.lhc_a)
print(beam_loop.lhc_t)


beam_loop_error = beam_loop_error - beam_loop_error[0] + injection_phase_error

plt.figure("Phase evolution")
plt.plot(beam_loop_error, color="r", label="Beam-phase loop")
plt.legend()
plt.tight_layout()
plt.grid()
plt.xlim(0, n_turns - 1)

plt.figure("Phase evolution")
plt.plot(synchro_loop_error, color="r", label="Beam-phase loop")
plt.legend()
plt.tight_layout()
plt.grid()
plt.xlim(0, n_turns - 1)


plt.show()


if SAVE_SIM:
    if DISABLE_PL:
        np.savez(
            f"generate_blond2_data/feedbacks/lhc/data/lhc_beam_control_{injection_phase_error:.1f}deg_nopl",
            unch_length=bunch_length,
            bunch_length_spread=bunch_length_spread,
            beam_loop_error=beam_loop_error,
            synchro_loop_error=synchro_loop_error,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
        )
    else:
        np.savez(
            f"generate_blond2_data/feedbacks/lhc/data/lhc_beam_control_{injection_phase_error:.1f}deg",
            bunch_length=bunch_length,
            bunch_length_spread=bunch_length_spread,
            beam_loop_error=beam_loop_error,
            synchro_loop_error=synchro_loop_error,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
        )
