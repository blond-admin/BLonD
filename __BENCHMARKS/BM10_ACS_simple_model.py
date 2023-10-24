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
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import LHCCavityLoop, LHCRFFeedback

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e


# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_p = 2.3e11         # Intensity
N_m = 50000          # Macro-particles
#I_rf_pk = 2*N_p*e/25e-9           # RF peak current [A]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450e9          # Synchronous momentum [eV/c]
h = 35640            # Harmonic number
V = 6e6              # RF voltage [V]
dphi = 0             # Phase modulation/offset
R_over_Q = 45        # Cavity R/Q [Ohms]
gamma_t = 53.8       # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 1           # Number of turns to track
# -----------------------------------------------------------------------------

PLOT_NO_BEAM = True

# Plot settings
plt.rc('axes', labelsize=12, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)


# Logger for messages on console & in file
Logger(debug=True)


ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
rf = RFStation(ring, [h], [V], [dphi])

# dummy beam & profile
beam = Beam(ring, N_m, N_p)
profile = Profile(beam, CutOptions(n_slices=100, cut_left=0, cut_right=2.5e-9))
I_rf_pk = 2*N_p*e/(10*rf.t_rf[0,0])

# Parameter assumptions
logging.info('Assuming following beam parameters')
logging.info('    Form factor 1, 2.3e11 p/b, 25 ns spacing')
logging.info('    RF frequency %.8e Hz', rf.omega_rf[0,0]/(2*np.pi))
logging.info('    RF peak current is %.3f A', I_rf_pk)
d_f = LHCCavityLoop.half_detuning(rf.omega_rf[0,0]/(2*np.pi), I_rf_pk, R_over_Q, V/8)
logging.info('    Optimum detuning in half-detuning scheme %.4e Hz', d_f)
power = LHCCavityLoop.half_detuning_power(I_rf_pk, V/8)
logging.info('    Optimum power in half-detuning scheme %.4e kW', power*1e-3)
Q_L = LHCCavityLoop.optimum_Q_L(d_f, rf.omega_rf[0,0]/(2*np.pi))
logging.info('    Optimum loaded Q %.0f', Q_L)

logging.info('Initialising LHCCavityLoop, tuned to injection (with no beam current)')
logging.info('CLOSED LOOP, no excitation, 5 turns pretracking')

CL = LHCCavityLoop(rf, profile, f_c=rf.omega_rf[0,0]/(2*np.pi)-d_f, G_gen=1,
                   I_gen_offset=0, n_cav=8, n_pretrack=1, Q_L=Q_L,
                   R_over_Q=R_over_Q, tau_loop=650e-9,
                   RFFB=LHCRFFeedback(open_loop=False, G_a=6.8e-6, G_d=10,
                                      excitation=False))

CL.track_simple(0)
CL.track_simple(0)
CL.track_simple(0)
CL.track_simple(0)
CL.track_simple(0)


if PLOT_NO_BEAM:
    plt.figure('Generator current')
    plt.plot(np.real(CL.I_GEN), label='real')
    plt.plot(np.imag(CL.I_GEN), label='imag')
    plt.xlabel('Samples [at 40 MS/s]')
    plt.ylabel('Generator current [A]')
    plt.legend()
    #plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/I_gen_no_beam.png')

    plt.figure('Antenna voltage')
    plt.plot(np.real(CL.V_ANT)*1e-6, label='real')
    plt.plot(np.imag(CL.V_ANT)*1e-6, label='imag')
    plt.xlabel('Samples [at 40 MS/s]')
    plt.ylabel('Antenna voltage [MV]')
    plt.legend()
    plt.show()
    #plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/V_ant_no_beam.png')

    logging.info('V_ant without beam %.8e + 1j * %.8e V', CL.V_ANT.real[-1], CL.V_ANT.imag[-1])
    logging.info('I_gen without beam %.8e + 1j * %.8e A', CL.I_GEN.real[-1], CL.I_GEN.imag[-1])

CL.track_simple(I_rf_pk)
CL.track_simple(I_rf_pk)

plt.figure('RF beam current, coarse grid')
plt.plot(np.real(CL.I_BEAM[CL.n_coarse:]), 'b', alpha=0.5, label='real')
plt.plot(np.imag(CL.I_BEAM[CL.n_coarse:]), 'r', alpha=0.5, label='imag')
plt.plot(np.absolute(CL.I_BEAM[CL.n_coarse:]), 'g', alpha=0.5, label='ampl')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('RF beam current [A]')
plt.legend()
plt.show()
#plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/I_beam.png')

logging.info('Total DC beam current %.4e A', np.sum(CL.profile.n_macroparticles)/beam.n_macroparticles*beam.intensity*e/ring.t_rev[0])

logging.info('Maximum RF beam current %.4e A', np.max(np.absolute(CL.I_BEAM)))

logging.info('Initial generator current is %.4f A', np.mean(np.absolute(CL.I_GEN[0:10])))
logging.info('Samples (omega x T_s) is %.4f', CL.samples)
logging.info('Cavity response to generator current')
logging.info('Antenna voltage is %.10f MV', np.mean(np.absolute(CL.V_ANT[-10:]))*1.e-6)
logging.info('Final generator current is %.10f A', np.mean(np.absolute(CL.I_GEN[-10:])))
P_gen = CL.generator_power()
logging.info('Average generator power before beam injection is %.10f kW', np.mean(P_gen)*1e-3)


fig = plt.figure('Antenna voltage, first turns with beam', figsize=(10,5))
gs = plt.GridSpec(2,4)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(np.absolute(CL.V_ANT)*1e-6, 'b', linewidth=0.3)
ax1.set_xlabel('Samples [at 40 MS/s]')
ax1.set_ylabel('Antenna voltage [MV]')
ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
ax2.plot(np.angle(CL.V_ANT, deg=True), 'b', linewidth=0.3)
ax2.set_xlabel('Samples [at 40 MS/s]')
ax2.set_ylabel('Phase [degrees]')
ax3 = fig.add_subplot(gs[:, 2:4])
ax3.scatter(CL.V_ANT.real*1e-6, CL.V_ANT.imag*1e-6)
ax3.set_xlabel('Voltage, I [MV]')
ax3.set_ylabel('Voltage, Q [MV]')
plt.tight_layout()
#plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/V_ant.png')

fig = plt.figure('Generator current, first turns with beam', figsize=(10,5))
gs = plt.GridSpec(2, 4)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(np.absolute(CL.I_GEN), 'b', linewidth=0.3)
ax1.set_xlabel('Samples [at 40 MS/s]')
ax1.set_ylabel('Generator current [A]')
ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
ax2.plot(np.angle(CL.I_GEN, deg=True), 'b', linewidth=0.3)
ax2.set_xlabel('Samples [at 40 MS/s]')
ax2.set_ylabel('Phase [degrees]')
ax3 = fig.add_subplot(gs[:, 2:4])
ax3.scatter(CL.I_GEN.real, CL.I_GEN.imag)
ax3.set_xlabel('Current, I [A]')
ax3.set_ylabel('Current, Q [A]')
plt.tight_layout()
plt.show()
#plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/I_gen.png')

P_gen = CL.generator_power()
plt.figure('Generator forward power')
plt.plot(P_gen[:CL.n_coarse]*1e-3, 'b', label='first turn')
plt.plot(P_gen[CL.n_coarse:]*1e-3, 'g', label='second turn')
plt.xlabel('Samples [at 40 MS/s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()
#plt.savefig('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/P_gen.png')

#P_gen = CL.generator_power()
logging.info('Average generator power after beam injection is %.10f kW', np.mean(P_gen)*1e-3)

logging.info('Done.')

#np.savetxt('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/BLonD_I_gen.txt', CL.I_GEN, fmt='%.8e')
#np.savetxt('/Users/timko/Documents/BLonD/CodeToCodeComp/ACSCavLoop/Case1/BLonD_V_ant.txt', CL.V_ANT, fmt='%.8e')