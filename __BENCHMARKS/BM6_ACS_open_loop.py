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
plt.rc('axes', labelsize=16, labelweight='normal')
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
CL = LHCCavityLoop(rf, profile, G_gen=1, f_c=rf.omega_rf[0,0]/(2*np.pi),
                   I_gen_offset=0.2778, n_cav=8, Q_L=20000, R_over_Q=45,
                   tau_loop=650e-9, n_pretrack=1,
                   RFFB=LHCRFFeedback(open_drive=True, G_a=0.00001))
logging.info('Initial generator current is %.4f A', np.mean(np.absolute(CL.I_GEN[0:10])))
logging.info('Samples (omega x T_s) is %.4f', CL.samples)
logging.info('Cavity response to generator current')
logging.info('Antenna voltage is %.10f MV', np.mean(np.absolute(CL.V_ANT[-10:]))*1.e-6)

plt.figure('Generator current (to cav)')
plt.plot(np.real(CL.I_GEN), label='real')
plt.plot(np.imag(CL.I_GEN), label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Generator current [A]')

plt.figure('Generator current (to gen)')
plt.plot(np.real(CL.I_TEST), label='real')
plt.plot(np.imag(CL.I_TEST), label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Generator current [A]')

plt.figure('Antenna voltage')
plt.plot(np.real(CL.V_ANT)*1e-6, label='real')
plt.plot(np.imag(CL.V_ANT)*1e-6, label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Antenna voltage [MV]')
plt.legend()
plt.show()
logging.info('RF feedback action')
logging.info('Updated generator current is %.10f A', np.mean(np.absolute(CL.I_GEN)))
P_gen = CL.generator_power()
logging.info('Generator power is %.10f kW', np.mean(P_gen[-10:])*1e-3)

#TF = TransferFunction(CL.I_GEN, CL.I_TEST, CL.T_s, plot=True)
#TF.analyse(data_cut=CL.n_coarse)