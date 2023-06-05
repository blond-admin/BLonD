# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for llrf.filters and llrf.cavity_feedback

:Authors: **Helga Timko**
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (CavityFeedbackCommissioning,
                                        SPSOneTurnFeedback)
from blond.llrf.impulse_response import (SPS3Section200MHzTWC,
                                         SPS4Section200MHzTWC)
from blond.llrf.signal_processing import rf_beam_current
from blond.toolbox.logger import Logger

# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [4.5e6]                 # 200 MHz RF voltage
phi = [0.]                  # 200 MHz RF phase
f_rf = 200.222e6            # Operational frequency of TWC, range ~200.1-200.36 MHz

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.0e11                # Bunch intensity [ppb]
N_t = 1000                  # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# OPTIONS TO TEST -------------------------------------------------------------
LOGGING = True              # Logging messages
RF_CURRENT = True           # RF beam current
RF_CURRENT2 = True          # RF beam current
IMP_RESP = True             # Impulse response of travelling wave cavity
FINE_COARSE = True          # Beam-induced voltage on fine/coarse grid
VIND_BEAM = True            # Beam-induced voltage

# OPTIONS TO TEST -------------------------------------------------------------

# Plot settings
plt.rc('axes', labelsize=12, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')  
plt.rc('legend', fontsize=12)  


# Logger for messages on console & in file
if LOGGING == True:
    Logger(debug = True)
else:
    Logger().disable()

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
logging.info("...... RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, reinsertion = True)
logging.info("...... Beam set!")
logging.info("Number of particles %d" %len(beam.dt))
logging.info("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt),
                                                              np.max(beam.dt)))

profile = Profile(beam, CutOptions=CutOptions(cut_left=-1.e-9,
                                              cut_right=6.e-9, n_slices=100))
profile.track()

if RF_CURRENT == True:

    # RF current calculation for Gaussian profile
    rf_current = rf_beam_current(profile, 2*np.pi*f_rf, ring.t_rev[0],
                                 lpf=False)
    rf_current_filt = rf_beam_current(profile, 2*np.pi*f_rf, ring.t_rev[0],
                                      lpf=True)
    #np.set_printoptions(precision=10)
    #print(repr(rf_current_filt.real))
    #print(repr(rf_current_filt.imag))

    fig, ax1 = plt.subplots()
    ax1.plot(profile.bin_centers, profile.n_macroparticles, 'g')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Macro-particle count [1]")
    ax2 = ax1.twinx()
    ax2.plot(profile.bin_centers, rf_current.real, 'b', label='current, real')
    ax2.plot(profile.bin_centers, rf_current.imag, 'r', label='current, imag')
    ax2.set_ylabel("RF current, charge count [C]")
    ax2.legend()

    fig, ax3 = plt.subplots()
    ax3.plot(profile.bin_centers, profile.n_macroparticles, 'g')
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Macro-particle count [1]")
    ax4 = ax3.twinx()
    ax4.plot(profile.bin_centers, rf_current_filt.real, 'b', label='filtered, real')
    ax4.plot(profile.bin_centers, rf_current_filt.imag, 'r', label='filtered, imag')
    ax4.set_ylabel("RF current, charge count [C]")
    ax4.legend()



if RF_CURRENT2 == True:

    # Create a batch of 100 equal, short bunches at HL-LHC intensity
    bunches = 100
    T_s = 5*rf.t_rev[0]/rf.harmonic[0, 0]
    N_m = int(1e5)
    N_b = 2.3e11
    bigaussian(ring, rf, beam, 0.1e-9, seed=1234, reinsertion=True)
    beam2 = Beam(ring, bunches*N_m, bunches*N_b)
    bunch_spacing = 5*rf.t_rf[0, 0]
    buckets = 5 * bunches
    for i in range(bunches):
        beam2.dt[i*N_m:(i+1)*N_m] = beam.dt + i*bunch_spacing
        beam2.dE[i*N_m:(i+1)*N_m] = beam.dE
    profile2 = Profile(beam2, CutOptions=CutOptions(cut_left=0,
        cut_right=bunches*bunch_spacing, n_slices=1000*buckets))
    profile2.track()

    tot_charges = np.sum(profile2.n_macroparticles) / \
                  beam2.n_macroparticles*beam2.intensity
    logging.info("Total number of charges %.10e p" %(np.sum(profile2.n_macroparticles)/beam2.n_macroparticles*beam2.intensity))

    # Calculate fine- and coarse-grid RF current
    rf_current_fine, rf_current_coarse = rf_beam_current(profile2,
        rf.omega_rf[0, 0], ring.t_rev[0], lpf=False,
        downsample={'Ts': T_s, 'points': rf.harmonic[0, 0]/5})
    rf_current_coarse /= T_s

    fig, ax5 = plt.subplots()
    ax5.plot(profile2.bin_centers*1e6, profile2.n_macroparticles, 'g')
    ax5.set_xlabel("Time [us]")
    ax5.set_ylabel("Macro-particle count [1]")
    ax6 = ax5.twinx()
    ax6.plot(profile2.bin_centers*1e6, rf_current_fine.real, 'b', label='fine, real')
    ax6.plot(profile2.bin_centers*1e6, rf_current_fine.imag, 'r', label='fine, imag')
    ax6.set_ylabel("RF charge distribution [C]")
    ax6.legend()

    t_coarse = np.linspace(0, rf.t_rev[0], num=int(rf.harmonic[0,0]/5))

    # Peak RF current on coarse grid
    peak_rf_current = np.max(np.absolute(rf_current_coarse))

    fig, ax7 = plt.subplots()
    ax7.plot(profile2.bin_centers*1e6, profile2.n_macroparticles, 'g', label='beam profile')
    ax7.set_xlabel("Time [us]")
    ax7.set_ylabel("Macro-particle count [1]")
    ax8 = ax7.twinx()
    ax8.plot(t_coarse*1e6, rf_current_coarse.real, 'b', label='coarse, real')
    ax8.plot(t_coarse*1e6, rf_current_coarse.imag, 'r', label='coarse, imag')
    ax8.plot(t_coarse*1e6, np.absolute(rf_current_coarse), 'purple', label='coarse, abs')
    ax8.set_ylabel("RF current [A]")
    ax8.legend()
    logging.info("Peak beam current, meas %.10f A" %(peak_rf_current))
    logging.info("Peak beam current, theor %.4f A" %(2*N_b*e/bunch_spacing))


if IMP_RESP == True:

    # Impulse response of beam- and generator-induced voltage for TWC
    # Comparison of wake and impulse resonse
    time = np.linspace(-1e-6, 4.e-6, 10000)
    TWC_v1 = SPS4Section200MHzTWC()
    TWC_v1.impulse_response_beam(2*np.pi*f_rf, time)
    TWC_v1.impulse_response_gen(2*np.pi*f_rf, time)
    TWC_v1.compute_wakes(time)
    TWC_v2 = TravelingWaveCavity(0.876e6, f_rf, 2*np.pi*6.207e-7)
    TWC_v2.wake_calc(time - time[0])
    t_beam = time - time[0]
    t_gen = time - time[0] - 0.5*TWC_v1.tau

    plt.figure()
    plt.plot(TWC_v2.time_array, TWC_v2.wake, 'b', marker='.', label='wake, impedances')
    plt.plot(t_beam, TWC_v1.W_beam, 'r', marker='.', label='cav wake, OTFB')
    plt.plot(t_beam, TWC_v1.h_beam.real, 'g', marker='.', label='hs_cav, OTFB')
    plt.plot(t_gen, TWC_v1.W_gen, 'orange', marker='.', label='gen wake, OTFB')
    plt.plot(t_gen, TWC_v1.h_gen.real, 'purple', marker='.', label='hs_gen, OTFB')
    plt.xlabel("Time [s]")
    plt.ylabel("Wake/impulse response [Ohms/s]")
    plt.legend()


if FINE_COARSE == True:

    # Create a batch of 100 equal, short bunches at HL-LHC intensity
    bunches = 100
    N_m = int(1e5)
    N_b = 2.3e11
    bigaussian(ring, rf, beam, 1.8e-9/4, seed=1234, reinsertion=True)
    beam2 = Beam(ring, bunches*N_m, bunches*N_b)
    bunch_spacing = 5*rf.t_rf[0, 0]
    buckets = 5 * bunches
    for i in range(bunches):
        beam2.dt[i*N_m:(i+1)*N_m] = beam.dt + i*bunch_spacing
        beam2.dE[i*N_m:(i+1)*N_m] = beam.dE
    profile2 = Profile(beam2, CutOptions=CutOptions(cut_left=0,
        cut_right=bunches*bunch_spacing, n_slices=1000*buckets))
    profile2.track()

    # Compare beam impulse response on coarse and fine grid
    time_fine = profile2.bin_centers - 0.5*profile2.bin_size
    time_coarse = np.linspace(0, rf.t_rev[0], 4620)

    TWC = SPS3Section200MHzTWC()
    TWC.impulse_response_beam(rf.omega_rf[0,0], time_fine, time_coarse)
    h_beam_fine = TWC.h_beam
    h_beam_coarse = TWC.h_beam_coarse
    print(len(time_fine), len(h_beam_fine))
    print(len(time_coarse), len(h_beam_coarse))

    # Calculate fine- and coarse-grid RF charge distribution
    rf_current_fine, rf_current_coarse = rf_beam_current(profile2,
        rf.omega_rf[0, 0], ring.t_rev[0], lpf=False,
        downsample={'Ts': rf.t_rev[0]/rf.harmonic[0, 0], 'points': rf.harmonic[0, 0]})

    OTFB = SPSOneTurnFeedback(rf, beam2, profile2, 3, n_cavities=1,
        Commissioning=CavityFeedbackCommissioning(open_FF=True))
    V_beam_fine = -OTFB.matr_conv(rf_current_fine, h_beam_fine)
    V_beam_coarse = -OTFB.matr_conv(rf_current_coarse, h_beam_coarse)
    print(len(time_fine), rf_current_fine.shape, V_beam_fine.shape)
    print(len(time_coarse), rf_current_coarse.shape, V_beam_coarse.shape)

    # Impulse response and induced voltage through OTFB object
    OTFB.TWC.impulse_response_beam(OTFB.omega_c, OTFB.profile.bin_centers,
                                   OTFB.rf_centers)
    OTFB.beam_response()
    np.set_printoptions(precision=10)
    #print(repr((OTFB.TWC.h_beam[::1000])[:100]))
    #print(repr(OTFB.TWC.h_beam_coarse[:100]))
    #print(repr((OTFB.V_fine_ind_beam[::1000])[:100]))
    #print(repr(OTFB.V_coarse_ind_beam[:100]))

    plt.figure()
    plt.plot(time_fine*1e6, h_beam_fine.real, 'b', marker='.', label='h_beam, fine, real')
    plt.plot(time_fine*1e6, OTFB.TWC.h_beam.real, 'b', marker='.', alpha=0.5, label='h_beam, fine, real')
    plt.plot(time_coarse*1e6, h_beam_coarse.real, 'teal', marker='.', label='h_beam, coarse, real')
    plt.plot(time_coarse*1e6, OTFB.TWC.h_beam_coarse.real, 'teal', marker='.', alpha=0.5, label='h_beam, coarse, real')
    plt.plot(time_fine*1e6, h_beam_fine.imag, 'r', marker='.', label='h_beam, fine, imag')
    plt.plot(time_fine*1e6, OTFB.TWC.h_beam.imag, 'r', marker='.', alpha=0.5, label='h_beam, fine, imag')
    plt.plot(time_coarse*1e6, h_beam_coarse.imag, 'orange', marker='.', label='h_beam, coarse, imag')
    plt.plot(time_coarse*1e6, OTFB.TWC.h_beam_coarse.imag, 'orange', marker='.', alpha=0.5, label='h_beam, coarse, imag')
    plt.xlabel("Time [us]")
    plt.ylabel("Wake/impulse response [Ohms/s]")
    plt.xlim((-1,5))
    plt.legend()

    plt.figure()
    plt.plot(time_fine*1e6, V_beam_fine.real*1e-6, 'b', marker='.', label='V_beam, fine, real')
    plt.plot(time_fine*1e6, OTFB.V_IND_FINE_BEAM[-OTFB.profile.n_slices:].real*1e-6, 'b', marker='.', alpha=0.5, label='V_beam, fine, real')
    plt.plot(time_coarse*1e6, V_beam_coarse.real*1e-6, 'teal', marker='.', label='V_beam, coarse, real')
    plt.plot(time_coarse*1e6, OTFB.V_IND_COARSE_BEAM[-h[0]:].real*1e-6, 'teal', marker='.', alpha=0.5, label='V_beam, coarse, real')
    plt.plot(time_fine*1e6, V_beam_fine.imag*1e-6, 'r', marker='.', label='V_beam, fine, imag')
    plt.plot(time_fine*1e6, OTFB.V_IND_FINE_BEAM[-OTFB.profile.n_slices:].imag*1e-6, 'r', marker='.', alpha=0.5, label='V_beam, fine, imag')
    plt.plot(time_coarse*1e6, V_beam_coarse.imag*1e-6, 'orange', marker='.', label='V_beam, coarse, imag')
    plt.plot(time_coarse*1e6, OTFB.V_IND_COARSE_BEAM[-h[0]:].imag*1e-6, 'orange', marker='.', alpha=0.5, label='V_beam, coarse, imag')
    plt.xlabel("Time [us]")
    plt.ylabel("Induced voltage per cavity [MV]")
    plt.xlim((-1,5))
    plt.legend()


if VIND_BEAM == True:

    profile = Profile(beam, CutOptions=CutOptions(cut_left=-1.e-9,
        cut_right=6.e-9, n_slices=140))
    profile.track()

    # One-turn feedback around 3-, 4-, and 5-section cavities
    omega_c = 2*np.pi*f_rf
    OTFB_3 = SPSOneTurnFeedback(rf, beam, profile, 3,
        Commissioning=CavityFeedbackCommissioning(open_FF=True))
    OTFB_4 = SPSOneTurnFeedback(rf, beam, profile, 4,
        Commissioning=CavityFeedbackCommissioning(open_FF=True))
    OTFB_5 = SPSOneTurnFeedback(rf, beam, profile, 5,
        Commissioning=CavityFeedbackCommissioning(open_FF=True))
    OTFB_3.counter = 0 # First turn
    OTFB_4.counter = 0 # First turn
    OTFB_5.counter = 0 # First turn
    OTFB_3.omega_c = omega_c
    OTFB_4.omega_c = omega_c
    OTFB_5.omega_c = omega_c
    #OTFB_3.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    #OTFB_4.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    #OTFB_5.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    OTFB_3.track()
    OTFB_4.track()
    OTFB_5.track()
    V_ind_beam = OTFB_3.V_IND_FINE_BEAM[-OTFB_3.profile.n_slices:] + OTFB_4.V_IND_FINE_BEAM[-OTFB_4.profile.n_slices:] + \
                 OTFB_5.V_IND_FINE_BEAM[-OTFB_5.profile.n_slices:]
    plt.figure()
    convtime = np.linspace(-1e-9, -1e-9+len(V_ind_beam.real)*
                           profile.bin_size, len(V_ind_beam.real))
    plt.plot(convtime, V_ind_beam.real, 'b--')
    plt.plot(convtime[:140], V_ind_beam.real[:140], 'b', label='Re(Vind), OTFB')
    plt.plot(convtime, V_ind_beam.imag, 'r--')
    plt.plot(convtime[:140], V_ind_beam.imag[:140], 'r', label='Im(Vind), OTFB')
    plt.plot(convtime[:140], V_ind_beam.real[:140]*np.cos(OTFB_4.omega_c*convtime[:140]) \
             + V_ind_beam.imag[:140]*np.sin(OTFB_4.omega_c*convtime[:140]), 
             color='purple', label='Total, OTFB')

    # Comparison with impedances: FREQUENCY DOMAIN
    TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 3.899e-6)
    TWC200_5 = TravelingWaveCavity(1.38e6, 200.222e6, 4.897e-6)
    indVoltageTWC = InducedVoltageTime(beam, profile, [TWC200_4, TWC200_4, TWC200_5, TWC200_5])
    indVoltage = TotalInducedVoltage(beam, profile, [indVoltageTWC])
    indVoltage.induced_voltage_sum()
    plt.plot(indVoltage.time_array, indVoltage.induced_voltage, color='limegreen', label='Time domain w FFT')
    
    # Comparison with impedances: TIME DOMAIN
    TWC200_4.wake_calc(profile.bin_centers - profile.bin_centers[0])
    TWC200_5.wake_calc(profile.bin_centers - profile.bin_centers[0])
    wake1 = 2*(TWC200_4.wake + TWC200_5.wake)
    Vind = -profile.Beam.ratio*profile.Beam.Particle.charge*e*\
        np.convolve(wake1, profile.n_macroparticles, mode='full')[:140]
    plt.plot(convtime[:140], Vind, color='teal', label='Time domain w conv')
    
    # Wake from impulse response
    OTFB_4.TWC.impulse_response_gen(omega_c, profile.bin_centers)
    OTFB_5.TWC.impulse_response_gen(omega_c, profile.bin_centers)
    OTFB_4.TWC.compute_wakes(profile.bin_centers)
    OTFB_5.TWC.compute_wakes(profile.bin_centers)
    wake2 = 2*(OTFB_4.TWC.W_beam + OTFB_5.TWC.W_beam)
    Vind = -profile.Beam.ratio*profile.Beam.Particle.charge*e*\
        np.convolve(wake2, profile.n_macroparticles, mode='full')[:140]
    plt.plot(convtime[:140], Vind, color='turquoise', label='Wake, OTFB')
    plt.xlabel("Time [s]")
    plt.ylabel("Induced voltage [V]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=2)
    
    plt.figure()
    plt.plot(profile.bin_centers, wake1, label='from impedances')
    plt.plot(profile.bin_centers, wake2, label='from OTFB')
    plt.xlabel("Time [s]")
    plt.ylabel("Wake field [Ohms/s]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=4)
    


plt.show()
logging.info("")
logging.info("Done!")



