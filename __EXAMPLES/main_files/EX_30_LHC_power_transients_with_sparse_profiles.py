import copy
import os, sys

from blond.beam.sparse_profiles import SparseBucket, SparseBatch
from blond.llrf.signal_processing import rf_beam_current

# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
# %matplotlib inline

# Import blond objects
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.llrf.cavity_feedback import (
    LHCCavityLoopCommissioning,
    LHCCavityLoop,
)

# Initialize the accelerator

# The synchrotron ring
C = 26658.883  # Machine circumference [m]
p_s = 450e9  # Synchronous momentum [eV/c]
gamma_t = 53.606713  # Transition gamma [-]
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor [-]
n_turns = 2  # Number of turns to track [-]

ring = Ring(C, alpha, p_s, Proton(), n_turns=n_turns)
print(f"Synchronous energy is {ring.energy[0, 0] * 1e-9:.1f} GeV")

# The RF station
h = 35640  # Harmonic number [-]
V = 5e6  # RF voltage [V]
dphi = 0  # Phase modulation/offset [rad]

rfstation = RFStation(ring, [h], [V], [dphi], n_rf=1)
rfstation_sparse = RFStation(ring, [h], [V], [dphi], n_rf=1)
print(f"RF voltage is {rfstation.voltage[0, 0] * 1e-6:.1f} MV")

# The beam
number_of_bunches = 5  # Length of the batch [number of bunches]
bunch_intensity = 2.3e11  # Bunch intensity [p/b]
n_macroparticles = 10000  # Number of macroparticles per bunch [-]
tau_bunch = 1.6e-9  # Bunch length [s]
bunch_spacing = 1000  # Bunch spacing [number of rf buckets]
injection_energy_error = 0  # Injection energy error [eV]

# Beam object for the batch
N_m = n_macroparticles * number_of_bunches
N_p = bunch_intensity * number_of_bunches
beam = Beam(ring, N_m, N_p)
beam_sparse = Beam(ring, N_m, N_p)


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

os.makedirs(this_directory + "../output_files/LHC_std_sparse/", exist_ok=True)

# First generate a single gaussian bunch
single_bunch = Beam(ring, n_macroparticles, bunch_intensity)
bigaussian(
    ring, rfstation, single_bunch, sigma_dt=tau_bunch / 4 / 2, seed=1234
)

# Copy the bunch throughout the batch
for i in range(number_of_bunches):
    beam.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = (
        single_bunch.dE
    )
    beam.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
        single_bunch.dt + i * bunch_spacing * rfstation.t_rf[0, 0]
    )
    beam_sparse.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = (
        single_bunch.dE
    )
    beam_sparse.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
        single_bunch.dt + i * bunch_spacing * rfstation.t_rf[0, 0]
    )

# Add final corrections to the bunch positions
bucket_shift = 10000
beam.dt += bucket_shift * rfstation.t_rf[0, 0]
beam.dE += injection_energy_error
beam_sparse.dt += bucket_shift * rfstation.t_rf[0, 0]
beam_sparse.dE += injection_energy_error

# The beam profile
cut_options = CutOptions(
    cut_left=(0 + bucket_shift) * rfstation.t_rf[0, 0],
    cut_right=(5 + bunch_spacing * number_of_bunches + bucket_shift)
    * rfstation.t_rf[
        0,
        0,
    ],
    n_slices=(5 + bunch_spacing * number_of_bunches) * 2**10,
)
profile = Profile(beam, cut_options)

filling_pattern = np.zeros(h)

for k in range(number_of_bunches):
    filling_pattern[bucket_shift + k * bunch_spacing] = 1

profile_sparse = SparseBatch(
    rf_station=rfstation_sparse,
    beam=beam_sparse,
    number_of_slices_per_profile=10 * 2**10,
    batch_list=filling_pattern,
    batch_length=10,
    tracker_mode="onebyone",
    do_track_on_init=False,
)

# Plot profile
profile.track()
profile_sparse.track()
fig, ax = plt.subplots(nrows=3, figsize=(10, 5))
ax[0].plot(
    profile.bin_centers * 1e6,
    profile.n_macroparticles,
    label="Standard profile",
)
for profile_ind in profile_sparse.profiles_list:
    ax[0].plot(
        profile_ind.bin_centers * 1e6,
        profile_ind.n_macroparticles,
        ls="--",
        label="Sparse profile",
    )
ax[0].set_xlabel(r"$\Delta t$ [$\mu$s]")
ax[0].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
ax[0].legend()
ax[0].set_yticks([])

ax[1].plot(
    profile.bin_centers * 1e6,
    profile.n_macroparticles,
    label="Standard profile",
)
for profile_ind in profile_sparse.profiles_list:
    ax[1].plot(
        profile_ind.bin_centers * 1e6,
        profile_ind.n_macroparticles,
        ls="--",
        label="Sparse profile",
    )
ax[1].set_xlabel(r"$\Delta t$ [$\mu$s]")
ax[1].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
ax[1].legend()
ax[1].set(xlim=[24.95, 24.954])
ax[1].set_yticks([])

ax[2].plot(
    profile.bin_centers * 1e6,
    profile.n_macroparticles,
    label="Standard profile",
)
for profile_ind in profile_sparse.profiles_list:
    ax[2].plot(
        profile_ind.bin_centers * 1e6,
        profile_ind.n_macroparticles,
        ls="--",
        label="Sparse profile",
    )
ax[2].set_xlabel(r"$\Delta t$ [$\mu$s]")
ax[2].set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
ax[2].legend()
ax[2].set(xlim=[27.445, 27.449])
ax[2].set_yticks([])
plt.savefig(this_directory + "../output_files/LHC_std_sparse/bunches")


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
delta_f = 0  # Initial detuning due to 12 bunches [Hz]

commissioning = LHCCavityLoopCommissioning(
    G_a=G_a,
    G_d=G_d,
    tau_d=tau_d,
    tau_a=tau_a,
    alpha=a_comb,
    G_o=G_otfb,
    open_tuner=True,
    open_rffb=False,
)

cavity_loop = LHCCavityLoop(
    rfstation,
    profile,
    RFFB=commissioning,
    f_c=rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f,
    Q_L=Q_L,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    n_pretrack=50,
    n_cavities=7,
    n_h=0,
)
cavity_loop_sparse = LHCCavityLoop(
    rfstation_sparse,
    profile_sparse,
    RFFB=commissioning,
    f_c=rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f,
    Q_L=Q_L,
    tau_loop=tau_loop,
    tau_otfb=tau_comp,
    n_pretrack=50,
    n_cavities=7,
    n_h=0,
)

# The RF tracker
rftracker = RingAndRFTracker(
    rfstation,
    beam,
    Profile=profile,
    CavityFeedback=cavity_loop,
    interpolation=True,
)
rftracker_sparse = RingAndRFTracker(
    rfstation_sparse,
    beam_sparse,
    Profile=profile_sparse,
    CavityFeedback=cavity_loop_sparse,
    interpolation=True,
)
# Initialize data arrays
rf_power = np.zeros((h // 10, n_turns), dtype=complex)
rf_power_sparse = np.zeros((h // 10, n_turns), dtype=complex)
I_beam_coarse = np.zeros((h // 10, n_turns), dtype=complex)
I_beam_coarse_sparse = np.zeros((h // 10, n_turns), dtype=complex)
# Tracking
fig, ax_vcorr = plt.subplots(nrows=n_turns)
for i in range(n_turns):
    profile.track()
    profile_sparse.track()
    rftracker.track()
    rftracker_sparse.track()

    # Store data
    rf_power[:, i] = cavity_loop.generator_power()[-h // 10 :]
    rf_power_sparse[:, i] = cavity_loop_sparse.generator_power()[-h // 10 :]
    I_beam_coarse[:, i] = cavity_loop.I_BEAM_COARSE[-h // 10 :]
    I_beam_coarse_sparse[:, i] = cavity_loop_sparse.I_BEAM_COARSE[-h // 10 :]

    ax_vcorr[i].plot(
        cavity_loop.profile.bin_centers,
        cavity_loop.V_corr,
        label="Standard profile",
    )
    for p, profile in enumerate(cavity_loop_sparse.profile.profiles_list):
        ax_vcorr[i].plot(
            profile.bin_centers,
            cavity_loop_sparse.V_corr[
                p * profile.n_slices : (p + 1) * profile.n_slices
            ],
            label=f"Sparse profile #{p}",
        )
    # np.testing.assert_array_almost_equal(cavity_loop.I_BEAM_COARSE[-h // 10 :],
    #                               cavity_loop_sparse.I_BEAM_COARSE[-h // 10
    #                                                                :],
    #                                      decimal= 12,
    #                               )
    #
    # np.testing.assert_array_equal(cavity_loop.I_BEAM_FINE[0:
    #                                   profile_sparse.n_slices],
    #                               cavity_loop_sparse.I_BEAM_FINE)

    # I_beam_fine[:, i] = cavity_loop.I_BEAM_FINE
    # I_beam_fine_sparse[:, i] = cavity_loop_sparse.I_BEAM_FINE
    ax_vcorr[i].set(
        xlabel="bin centers",
        ylabel="V_corr",
    )
    ax_vcorr[i].legend()
plt.show()

# Bucket-by-bucket RF power
fig, (ax, ax_sparse) = plt.subplots(nrows=2, figsize=(10, 5))

cmap = plt.get_cmap("RdYlBu_r", n_turns)
delta_t = np.linspace(0, rfstation.t_rev[0], h // 10) * 1e6
bpos = (
    np.linspace(
        bucket_shift * rfstation.t_rf[0, 0],
        (bucket_shift + bunch_spacing * number_of_bunches)
        * rfstation.t_rf[0, 0],
        number_of_bunches,
    )
    * 1e6
)
for i in range(n_turns):
    ax.plot(delta_t, np.abs(rf_power[:, i]) / 1e3, color=cmap(i), alpha=0.4)
    ax_sparse.plot(
        delta_t, np.abs(rf_power_sparse[:, i]) / 1e3, color=cmap(i), alpha=0.4
    )
ax.fill_betweenx(
    np.linspace(69, 71, 100),
    bpos[0] * np.ones(100),
    bpos[-1] * np.ones(100),
    alpha=0.1,
    color="r",
    label="Beam",
)
ax_sparse.fill_betweenx(
    np.linspace(69, 71, 100),
    bpos[0] * np.ones(100),
    bpos[-1] * np.ones(100),
    alpha=0.1,
    color="r",
    label="Beam",
)
# ax.set_xlim((delta_t[800], delta_t[1300]))
# ax.set_ylim((50, 80))
ax.grid()
ax.legend()
ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
ax.set_ylabel(r"Generation power [kW]")
# ax_sparse.set_xlim((delta_t[800], delta_t[1300]))
# ax_sparse.set_ylim((50, 80))
ax_sparse.grid()
ax_sparse.legend()
ax_sparse.set_xlabel(r"$\Delta t$ [$\mu$s]")
ax_sparse.set_ylabel(r"Generation power [kW]")
plt.savefig(this_directory + "../output_files/LHC_std_sparse/gen_power_")

# Turn-by-turn RF power
fig, (ax, ax_current) = plt.subplots(nrows=2, figsize=(10, 5))

cmap = plt.get_cmap("RdYlBu_r", n_turns)

ax.plot(
    np.max(np.abs(rf_power) / 1e3, axis=0), color="r", label="Standard profile"
)
ax.plot(
    np.max(np.abs(rf_power_sparse) / 1e3, axis=0),
    ls="--",
    label="Sparse profile",
)
ax.grid()
ax.legend()
ax.set_xlabel(r"Turns")
ax.set_ylabel(r"Generation power [kW]")
ax.set_xlim((0, n_turns - 1))

ax_current.plot(
    np.linspace(0, rfstation.t_rev[0], len(np.abs(I_beam_coarse.flatten())))
    * 1e6,
    np.abs(I_beam_coarse.flatten()) * 1e3,
    label="Standard profile",
)
ax_current.plot(
    np.linspace(
        0, rfstation.t_rev[0], len(np.abs(I_beam_coarse_sparse.flatten()))
    )
    * 1e6,
    np.abs(I_beam_coarse_sparse.flatten()) * 1e3,
    ls="--",
    label="Sparse profile",
)
ax_current.legend()
ax_current.set_xlim([24.9, 27.5])
ax_current.set_xlabel(r"t [us]")
ax_current.set_ylabel(r"RF component of beam current [mA]")

plt.savefig(
    this_directory
    + "../output_files/LHC_std_sparse/max_gen_power_beam_current"
)

plt.show()
