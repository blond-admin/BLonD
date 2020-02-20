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

import numpy as np
from scipy.constants import e
import matplotlib.pyplot as plt 

from blond.toolbox.logger import Logger
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import SPSOneTurnFeedback
from blond.llrf.signal_processing import rf_beam_current
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.llrf.impulse_response import SPS4Section200MHzTWC
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage


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
N_b = 1.e11                 # Bunch intensity [ppb]
N_t = 1000                  # Number of turns to track
# CERN SPS --------------------------------------------------------------------

# OPTIONS TO TEST -------------------------------------------------------------
LOGGING = True              # Logging messages
RF_CURRENT = True           # RF beam current
IMP_RESP = True             # Impulse response of travelling wave cavity
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
print("Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
print("RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, reinsertion = True)
print("Beam set! Number of particles %d" %len(beam.dt))
print("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt),
                                                       np.max(beam.dt)))

profile = Profile(beam, CutOptions=CutOptions(cut_left=-1.e-9,
                                              cut_right=6.e-9, n_slices = 140))
profile.track()

if RF_CURRENT == True:

    # RF current calculation for Gaussian profile
    rf_current = rf_beam_current(profile, 2*np.pi*f_rf, ring.t_rev[0],
                                 lpf=False)
    np.set_printoptions(precision=10)

    fig, ax1 = plt.subplots()
    ax1.plot(profile.bin_centers, profile.n_macroparticles, 'g')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Macro-particle count [1]")
    ax2 = ax1.twinx()
    ax2.plot(profile.bin_centers, rf_current.real, 'b', label='current, real')
    ax2.plot(profile.bin_centers, rf_current.imag, 'r', label='current, imag')
    ax2.set_ylabel("RF current, charge count [C]")
    ax2.legend()


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

    plt.figure(2)
    plt.plot(TWC_v2.time_array, TWC_v2.wake, 'b', marker='.', label='wake, impedances')
    plt.plot(t_beam, TWC_v1.W_beam, 'r', marker='.', label='cav wake, OTFB')
    plt.plot(t_beam, TWC_v1.h_beam.real, 'g', marker='.', label='hs_cav, OTFB')
    plt.plot(t_gen, TWC_v1.W_gen, 'orange', marker='.', label='gen wake, OTFB')
    plt.plot(t_gen, TWC_v1.h_gen.real, 'purple', marker='.', label='hs_gen, OTFB')
    plt.xlabel("Time [s]")
    plt.ylabel("Wake/impulse response [Ohms/s]")
    plt.legend()


if VIND_BEAM == True:

    # One-turn feedback around 3-, 4-, and 5-section cavities
    omega_c = 2*np.pi*f_rf
    OTFB_3 = SPSOneTurnFeedback(rf, beam, profile, 3)
    OTFB_4 = SPSOneTurnFeedback(rf, beam, profile, 4)
    OTFB_5 = SPSOneTurnFeedback(rf, beam, profile, 5)
    OTFB_3.counter = 0 # First turn
    OTFB_4.counter = 0 # First turn
    OTFB_5.counter = 0 # First turn
    OTFB_3.omega_c = omega_c
    OTFB_4.omega_c = omega_c
    OTFB_5.omega_c = omega_c
    OTFB_3.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    OTFB_4.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    OTFB_5.TWC.impulse_response_beam(omega_c, profile.bin_centers)
    OTFB_3.beam_induced_voltage(lpf=False)
    OTFB_4.beam_induced_voltage(lpf=False)
    OTFB_5.beam_induced_voltage(lpf=False)
    V_ind_beam = OTFB_3.V_fine_ind_beam +OTFB_4.V_fine_ind_beam + OTFB_5.V_fine_ind_beam
    plt.figure(3)
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
    
    plt.figure(4)
    plt.plot(profile.bin_centers, wake1, label='from impedances')
    plt.plot(profile.bin_centers, wake2, label='from OTFB')
    plt.xlabel("Time [s]")
    plt.ylabel("Wake field [Ohms/s]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=4)
    


plt.show()
print("")
print("Done!")



