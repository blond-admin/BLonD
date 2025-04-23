from __future__ import division

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_e
from scipy.optimize import curve_fit

from blond.beam.beam import Beam, Positron
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.synchrotron_radiation.synchrotron_radiation import \
    SynchrotronRadiation
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.utils import bmath as bm

DRAFT_MODE = bool(int(os.environ.get("BLOND_EXAMPLES_DRAFT_MODE", False)))
# To check if executing correctly, rather than to run the full simulation

mpl.use('Agg')

this_directory = "/Users/lvalle/cernbox/Documents/SY-RF-BR/BLonD_SR/SR_comparison_and_tests/"

os.makedirs(this_directory + 'output_files/EX_13_fig_comparison/', exist_ok=True)

######### COPIE FROM EX_13_synchrotron_radiation.py file in the BLonD example folder ###################################

# SIMULATION PARAMETERS -------------------------------------------------------
# Beam parameters
particle_type = Positron()
n_particles = int(1.7e11)
n_macroparticles = int(1001) if DRAFT_MODE else int(1e5)

# Targets
sigmaE = 1e-3
bl = 4e-3 / c # 4 times rms

radiation_integrals = np.array([0.646747216157, 0.0005936549319, 5.6814536525e-08, 5.92870407301e-09 , 1.698280783E-11])
#

jz = 2 + radiation_integrals[3]/radiation_integrals[1]
E0 = particle_type.mass
energy = E0 * np.sqrt(sigmaE **2 / particle_type.c_q * (jz * radiation_integrals[1]/radiation_integrals[2])) # [eV]

U0 = particle_type.c_gamma / (2 * np.pi)* energy**4 * radiation_integrals[1]

distribution_type = 'gaussian'
emittance = 1000
distribution_variable = 'Action'

tracking = True

# Machine and RF parameters
C = 90.65874532 * 1e3
# Tracking details
n_turns = int(40000)
n_turns_between_two_plots = 100

# Derived parameters
alpha_0 = 7.120435962 * 1e-6
# Cavities parameters
n_rf_systems = 1
harmonic_numbers = 2 * 121200
voltage_program = U0/e
phi_offset = 0

bucket_length = 4e-3 / c

# DEFINE RING------------------------------------------------------------------

n_sections = 1
general_params = Ring(np.ones(n_sections) * C / n_sections,
                      np.tile(alpha_0, (1, n_sections)).T,
                      np.tile(energy, (n_sections, n_turns + 1)),
                      particle_type, n_turns, synchronous_data_type='total energy', n_sections=n_sections)

RF_sct_par = []
for i in np.arange(n_sections) + 1:
    RF_sct_par.append(RFStation(general_params,
                                [harmonic_numbers], [voltage_program / n_sections],
                                [phi_offset], n_rf_systems, section_index=i))

# DEFINE BEAM------------------------------------------------------------------

beam = Beam(general_params, n_macroparticles, n_particles)
beam_bef = Beam(general_params, n_macroparticles, n_particles)


# DEFINE TRACKER---------------------------------------------------------------
longitudinal_tracker = []
for i in range(n_sections):
    longitudinal_tracker.append(RingAndRFTracker(RF_sct_par[i], beam))

full_tracker = FullRingAndRF(longitudinal_tracker)

# BEAM GENERATION--------------------------------------------------------------

matched_from_distribution_function(beam, full_tracker, emittance=emittance,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1000)
matched_from_distribution_function(beam_bef, full_tracker, emittance=emittance,
                                   distribution_type=distribution_type,
                                   distribution_variable=distribution_variable, seed=1000)
# Synchrotron radiation objects without quantum excitation
rho = 11e3
SR = []
for i in range(n_sections):
    SR.append(SynchrotronRadiation(general_params, RF_sct_par[i], beam, rad_int=radiation_integrals,
                                   quantum_excitation=True, python=True, shift_beam=True))
SR[0].print_SR_params()

# ACCELERATION MAP-------------------------------------------------------------
map_ = []
for i in range(n_sections):
    map_ += [longitudinal_tracker[i]] + [SR[i]]

# TRACKING + PLOTS-------------------------------------------------------------

if DRAFT_MODE:
    n_turns = 5
if tracking :
    avg_dt = np.zeros(n_turns)
    std_dt = np.zeros(n_turns)
    std_dE = np.zeros(n_turns)
    avg_dt_compare = np.zeros(n_turns)
    std_dt_compare = np.zeros(n_turns)
    Ecompare = np.zeros(n_turns)
    U0 = np.zeros(n_turns)
    beam.statistics()
    for i in range(n_turns):
        # BLonD SR integration
        # while beam.sigma_dE > 1e-3:
        for m in map_:
            m.track()
        print(f'Turn: {i}')
        avg_dt[i] = np.mean(beam.dt)
        std_dt[i] = np.std(beam.dt)
        std_dE[i] = np.std(beam.dE)
        #if i == 40000:
        #    beam.statistics()
        #    np.save("damped_distribution_dt", beam.dt)
        #    print(np.std(beam.dE))
        #    np.save("damped_distribution_dE", beam.dE)
# Fitting routines for synchrotron radiation damping
    i_turn = np.argmin(np.abs(std_dt - 4e-3 / c / 4))
    print(i_turn)
else :
    beam.dt = np.load("damped_distribution_dt.npy")
    beam.dE = np.load("damped_distribution_dE.npy")
    beam.statistics()
    print(np.std(beam.dE))

def sine_exp_fit(x, y, **keywords):
    try:
        init_values = keywords['init_values']
        offset = init_values[-1]
        init_values[-1] = 0
    except Exception:
        offset = np.mean(y)
        # omega estimation using FFT
        npoints = 12
        y_fft = np.fft.fft(y - offset, 2 ** npoints)
        omega_osc = (2.0 * np.pi * np.abs(y_fft[:2 ** (npoints - 1)]).argmax() /
                     len(y_fft) / (x[1] - x[0]))
        init_amp = (y.max() - y.min()) / 2.0
        init_omega = omega_osc
        init_values = [init_omega, 0, init_amp, 0, 0]

    popt, pcov = curve_fit(sine_exp_f, x, y - offset, p0=init_values)

    popt[0] = np.abs(popt[0])
    popt[2] = np.abs(popt[2])
    popt[3] += offset
    if np.isinf(pcov).any():
        pcov = np.zeros([5, 5])

    return popt, pcov


def sine_exp_f(x, omega, phi, amp, offset, tau):
    return offset + np.abs(amp) * np.sin(omega * x + phi) * np.exp(tau * x)


def exp_f(x, amp, offset, tau):
    return offset + np.abs(amp) * np.exp(-np.abs(tau) * x)


# Fit of the bunch length
plt.figure(figsize=[6, 4.5])
plt.plot(1e3 * 4.0 * std_dt * c, lw=2, label='no deceleration')
# plt.ylim(0, plt.ylim()[1])
plt.xlabel('Turns')
plt.ylabel('Bunch length [mm]')
plt.legend()
plt.savefig(this_directory + 'output_files/EX_13_fig_comparison/bl_comparison.png')
plt.close()

plt.figure(figsize=[6, 4.5])
plt.scatter(beam_bef.dt, beam_bef.dE, lw=2, label='before')
beam_bef.dt = np.load("initial_distribution_dt.npy")
beam_bef.dE = np.load("initial_distribution_dE.npy")
plt.scatter(beam_bef.dt, beam_bef.dE, lw=2, label='wanted')
plt.scatter(beam.dt, beam.dE, lw=2, label=f'after {n_turns} turns')
plt.legend()
plt.savefig(this_directory + 'output_files/EX_13_fig_comparison/beam_comparison.png')
plt.close()

itemindex = bm.nonzero(beam.id)[0]
beam.statistics()
beam.dt[itemindex] += - beam.mean_dt
beam.dE[itemindex] += - beam.mean_dE

plt.figure(figsize=[6, 4.5])

plt.scatter(beam.dt, beam.dE, lw=2, label='after')
plt.scatter(beam_bef.dt, beam_bef.dE, lw=2, label='before')
plt.legend()
plt.savefig(this_directory + 'output_files/EX_13_fig_comparison/beam_comparison_shifted.png')
plt.show()
# plt.close()

plt.figure(figsize=[6, 4.5])
plt.plot(avg_dt * 1e9, lw=2)
plt.xlabel('Turns')
plt.ylabel('Bunch position [ns]')
plt.legend()
plt.savefig(this_directory + 'output_files/EX_13_fig_comparison/pos_comparison')
plt.close()

plt.figure(figsize=[6, 4.5])
plt.plot(np.pi * std_dt * std_dE, lw=2)
plt.xlabel('Turns')
plt.ylabel('Emittance [eVs]')
plt.legend()
plt.savefig(this_directory + 'output_files/EX_13_fig_comparison/emittance_evolution')
plt.close()
