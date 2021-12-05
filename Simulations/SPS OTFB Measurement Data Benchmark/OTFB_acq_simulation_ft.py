
# Imports -------------------------------------------------------------------------------------------------------------
import os

print('Importing...\n')
import matplotlib.pyplot as plt
import numpy as np
import Simulations.Tests.utils_test as ut
from Simulations.Tests.OTFB_full_machine_theory import theory_calc
from matplotlib import gridspec
from argparse import ArgumentParser
import scipy.stats as spst
import scipy.io
import data_utilities as dut
import os.path
import tqdm


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
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
import blond.utils.bmath as bm


# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

SINGLE_BATCH = False
GENERATE = False

# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 10000                                      # Number of turns to track

if SINGLE_BATCH:
    N_bunches = 72                              # Number of bunches
else:
    N_bunches = 288                             # Number of bunches

fit_type = 'fwhm'
print('Fit type:', fit_type)

no_impedances = True
print('Impedances:', not no_impedances)
print('Number of Bunches:', N_bunches)
print('GENERATE:', GENERATE)
print()


# Objects -------------------------------------------------------------------------------------------------------------
print('Initializing Objects...\n')


# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h], [V], [phi], n_rf=1)

# SINGLE BUNCH FIRST
# Beam
bunch_intensities = np.load('avg_bunch_intensities_red.npy')

bunch_intensities = 3385.8196 * 10**10 * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)

beam = Beam(SPS_ring, int(np.sum(n_macro[:N_bunches])), np.sum(bunch_intensities[:N_bunches]))

# Tracker object for full ring
SPS_rf_tracker = RingAndRFTracker(rfstation, beam)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])


# Initialize the bunch
bunch_lengths_fl = np.load('avg_bunch_length_full_length_red.npy')
bunch_lengths_fwhm = np.load('avg_bunch_length_FWHM.npy')
exponents = np.load('avg_exponent_red.npy')
positions = np.load('avg_positions_red.npy')

if fit_type == 'fwhm':
    bunch_length_list = bunch_lengths_fwhm * 1e-9
else:
    bunch_length_list = bunch_lengths_fl * 1e-9


distribution_options_list = {'bunch_length': bunch_length_list[:N_bunches],
                              'type': 'binomial',
                              'density_variable': 'Hamiltonian',
                              'bunch_length_fit': fit_type,
                              'exponent': exponents[:N_bunches]}

bunch_positions = (positions - positions[0]) / 5e-9


if GENERATE:
    matched_from_distribution_density_multibunch(beam, SPS_ring, SPS_tracker, distribution_options_list,
                                                 N_bunches, bunch_positions[:N_bunches],
                                                 intensity_list=bunch_intensities[:N_bunches],
                                                 n_iterations=20)
    beam.dt += 1000 * 5e-9

    np.save(f'generated_beam_{fit_type}_{N_bunches}_dE.npy', beam.dE)
    np.save(f'generated_beam_{fit_type}_{N_bunches}_dt.npy', beam.dt)
else:
    beam.dE = np.load(f'generated_beam_FWHM_dE.npy')
    beam.dt = np.load(f'generated_beam_FWHM_dt.npy')

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))
profile.track()

#plt.plot(profile.bin_centers, profile.n_macroparticles)
#plt.xlim(0, 20e-6)
#plt.show()

np.save(f'generated_profile_{fit_type}_{N_bunches}', profile.n_macroparticles)
np.save(f'generated_profile_bins_{fit_type}_{N_bunches}', profile.bin_centers)

# One Turn Feedback
V_part = 0.5442095845867135
G_tx_ls = [0.2564371551236985, 0.53055789217211453]

#G_tx_ls = [0.2712028956, 0.58279606]
#G_llrf_ls = [41.751786, 35.24865]



Commissioning = CavityFeedbackCommissioning_new(open_FF=True, debug=False)
OTFB = SPSCavityFeedback_new(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                              Commissioning=Commissioning, G_tx=G_tx_ls)


# Tracking ------------------------------------------------------------------------------------------------------------
# Tracking with the beam
map_ = [profile] + [OTFB] + [SPS_tracker]

dt_plot = 100
dt_track = 100

for i in tqdm.trange(N_t):
    for m in map_:
        m.track()

    #if i % dt_track == 0:
    #    print(i)

    if i % dt_plot == 0:
        OTFB.OTFB_1.calc_power()
        OTFB.OTFB_2.calc_power()

        dut.save_plots_OTFB(OTFB, f'fig/', i)


OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()

print(rfstation.t_rev[0] / rfstation.t_rf[0,0])


dir = f'sim_data/{N_t}turns_{fit_type}_{N_bunches}/'

if not os.path.exists(dir):
    os.makedirs(f'sim_data/{N_t}turns_{fit_type}_{N_bunches}/')

dut.save_data(OTFB, f'sim_data/{N_t}turns_{fit_type}_{N_bunches}/')

np.save(f'profile_data/generated_beam_{fit_type}_{N_bunches}_dE_end.npy', beam.dE)
np.save(f'profile_data/generated_beam_{fit_type}_{N_bunches}_dt_end.npy', beam.dt)

np.save(f'profile_data/generated_profile_{fit_type}_{N_bunches}_end', profile.n_macroparticles)
np.save(f'profile_data/generated_profile_bins_{fit_type}_{N_bunches}_end', profile.bin_centers)
