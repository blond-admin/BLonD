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
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions, FitOptions
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCRFFeedback
from blond.llrf.transfer_function import TransferFunction

import logging
import numpy as np
import matplotlib.pyplot as plt


# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_p = 2.3e11         # Intensity
N_m = 50000          # Macro-particles
NB = 156             # Number of bunches
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

bunch = Beam(ring, N_m, N_p)
bigaussian(ring, rf, bunch, sigma_dt=tau_0)

beam = Beam(ring, N_m*NB, N_p)
for i in range(12):
    beam.dt[i*N_m:(i+1)*N_m] = bunch.dt[0:N_m] + i*25e-9
    beam.dE[i*N_m:(i+1)*N_m] = bunch.dE[0:N_m]
for i in range(12,60):
    beam.dt[i*N_m:(i+1)*N_m] = bunch.dt[0:N_m] + i*25e-9 + 0.8e-6
    beam.dE[i*N_m:(i+1)*N_m] = bunch.dE[0:N_m]
for i in range(60,108):
    beam.dt[i*N_m:(i+1)*N_m] = bunch.dt[0:N_m] + i*25e-9 + 1.2e-6
    beam.dE[i*N_m:(i+1)*N_m] = bunch.dE[0:N_m]
for i in range(108,156):
    beam.dt[i*N_m:(i+1)*N_m] = bunch.dt[0:N_m] + i*25e-9 + 1.4e-6
    beam.dE[i*N_m:(i+1)*N_m] = bunch.dE[0:N_m]

buckets = 1 + 10*NB + 1.4e-6/2.5e-9
logging.debug('Maximum of beam coordinates %.4e s', np.max(beam.dt))
logging.info('Number of buckets considered %d', buckets)
logging.debug('Profile cut set at %.4e s', buckets*2.5e-9)
profile = Profile(beam, CutOptions(n_slices=int(100*buckets), cut_left=0,
                                   cut_right=buckets*2.5e-9),
                  FitOptions(fit_option='gaussian'))
profile.track()
plt.figure('Bunch profile')
plt.plot(profile.bin_centers, profile.n_macroparticles)
plt.xlabel('Bin centers [s]')
plt.ylabel('Macroparticles [1]')
plt.show()

logging.info('Initialising LHCCavityLoop, tuned to injection (with no beam current)')
logging.info('CLOSED LOOP, no excitation, 1 turn tracking')
CL = LHCCavityLoop(rf, profile, f_c=rf.omega_rf[0,0]/(2*np.pi), G_gen=1,
                   I_gen_offset=0, n_cav=8, n_pretrack=2, Q_L=20000,
                   R_over_Q=45, tau_loop=650e-9, T_s=25e-9,
                   RFFB=LHCRFFeedback(open_loop=False, G_a=0.00001, G_d=10,
                                      excitation=False))
CL.rf_beam_current()
plt.figure('RF beam current')
plt.plot(np.real(CL.I_BEAM), label='real')
plt.plot(np.imag(CL.I_BEAM), label='imag')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('RF beam current [A]')
plt.show()


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



logging.info('Done.')
