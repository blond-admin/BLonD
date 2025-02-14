# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for llrf.cavity_feedback LHCCavityLoop
Benchmarking open loop gain

:Authors: **Helga Timko**
"""


from blond.toolbox.logger import Logger
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions, FitOptions
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCRFFeedback
from blond.llrf.transfer_function import TransferFunction

import logging
import numpy as np
import matplotlib.pyplot as plt


# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9            # Intensity
N_p = 50000          # Macro-particles
tau_0 = 0.4e-9       # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450e9          # Synchronous momentum [eV/c]
h = 35640            # Harmonic number
V = 4e6              # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 53.8       # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 1           # Number of turns to track
# -----------------------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=12, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)


# Logger for messages on console & in file
Logger(debug=True)


ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
rf = RFStation(ring, [h], [V], [dphi])

beam = Beam(ring, N_p, N_b)
profile = Profile(beam, CutOptions(n_slices=100),
                  FitOptions(fit_option='gaussian'))

logging.info('Initialising LHCCavityLoop, tuned to injection (with no beam current)')
logging.info('CLOSED LOOP, no excitation, 1 turn tracking')
CL = LHCCavityLoop(rf, profile, f_c=rf.omega_rf[0,0]/(2*np.pi), G_gen=1,
                   I_gen_offset=0, n_cav=8, n_pretrack=1, Q_L=20000,
                   R_over_Q=45, tau_loop=650e-9, #T_s=25e-9,
                   RFFB=LHCRFFeedback(open_loop=False, G_a=6.8e-6, G_d=10, #G_a=0.00001, G_d=10,
                                      excitation=False))
logging.info('Initial generator current is %.4f A', np.mean(np.absolute(CL.I_GEN[0:10])))
logging.info('Samples (omega x T_s) is %.4f', CL.samples)
logging.info('Cavity response to generator current')
logging.info('Antenna voltage is %.10f MV', np.mean(np.absolute(CL.V_ANT[-10:]))*1.e-6)
logging.info('Final generator current is %.10f A', np.mean(np.absolute(CL.I_GEN[-10:])))
P_gen = CL.generator_power()
logging.info('Generator power is %.10f kW', np.mean(P_gen)*1e-3)

plt.figure('Generator current')
plt.plot(np.real(CL.I_GEN), label='real')
plt.plot(np.imag(CL.I_GEN), label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Generator current [A]')

plt.figure('Antenna voltage')
plt.plot(np.real(CL.V_ANT)*1e-6, label='real')
plt.plot(np.imag(CL.V_ANT)*1e-6, label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Antenna voltage [MV]')
plt.legend()
plt.show()


logging.info('CLOSED LOOP, with excitation, 10 turns tracking')
CL = LHCCavityLoop(rf, profile, f_c=rf.omega_rf[0,0]/(2*np.pi), G_gen=1,
                   I_gen_offset=0, n_cav=8, n_pretrack=10, Q_L=20000,
                   R_over_Q=45, tau_loop=650e-9, #T_s=25e-9,
                   RFFB=LHCRFFeedback(open_loop=False, G_a=6.8e-6, G_d=10,
                                      excitation=True))

plt.figure('Noise injected into Set Point')
plt.plot(np.real(CL.V_EXC_IN), label='real')
plt.plot(np.imag(CL.V_EXC_IN), label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Set point voltage [V]')
plt.legend()
plt.show()

T_s = CL.T_s

logging.info('Calculating transfer function')
TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s, plot=False)
TF.analyse(data_cut=0)

# Same with 60 k QL
logging.info('Re-track with Q_L = 60k')
CL.Q_L = 60000
CL.track_no_beam_excitation(CL.n_pretrack)
logging.info('Calculating transfer function')
TF60k = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s, plot=False)
TF60k.analyse(data_cut=0)

fig = plt.figure('Transfer functions')
gs = plt.GridSpec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Transfer function')
ax1.plot(TF.f_est/10 ** 6, 20 * np.log10(np.abs(TF.H_est)), 'b', linewidth=0.3, label='20k')
ax1.plot(TF60k.f_est/10 ** 6, 20 * np.log10(np.abs(TF60k.H_est)), 'r', linewidth=0.3, label='60k')
ax1.set_xlabel('Frequency [MHz]')
ax1.set_ylabel('Gain [dB]')
plt.legend()
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(TF.f_est/10**6, (180/np.pi)*np.unwrap(np.angle(TF.H_est)), 'b', linewidth=0.3, label='20k')
ax2.plot(TF60k.f_est/10**6, (180/np.pi)*np.unwrap(np.angle(TF60k.H_est)), 'r', linewidth=0.3, label='60k')
ax2.set_xlabel('Frequency [MHz]')
ax2.set_ylabel('Phase [degrees]')
plt.legend()
plt.show()

fig = plt.figure('Closed loop response')
gs = plt.GridSpec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Transfer function')
ax1.plot(TF60k.f_est/10 ** 3, 20 * np.log10(np.abs(TF60k.H_est)), 'b', linewidth=0.3)
ax1.set_xlabel('Frequency [kHz]')
ax1.set_ylabel('Gain [dB]')
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(TF60k.f_est/10**3, (180/np.pi)*np.unwrap(np.angle(TF60k.H_est)), 'b', linewidth=0.3)
ax2.set_xlabel('Frequency [kHz]')
ax2.set_ylabel('Phase [degrees]')
ax1.set_xlim((-2000, 2000))
ax1.set_ylim((-30, 10))
ax2.set_ylim((-180, 180))
plt.tight_layout()
plt.show()

logging.info('Done.')
