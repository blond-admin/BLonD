# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for SPS OTFB with FF

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (SPSCavityLoopCommissioning,
                                        SPSCavityFeedback)
from blond.llrf.impulse_response import (SPS3Section200MHzTWC,
                                         SPS4Section200MHzTWC,
                                         SPS5Section200MHzTWC)
from blond.llrf.signal_processing import feedforward_filter
from blond.plots.plot_beams import plot_long_phase_space
from blond.toolbox.logger import Logger
from blond.trackers.tracker import RingAndRFTracker

# CERN SPS --------------------------------------------------------------------
# Machine and RF parameters
C = 2 * np.pi * 1100.009    # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 450e9                 # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [10e6]                  # 200 MHz RF voltage
phi = [0.]                  # 200 MHz RF phase

# Beam and tracking parameters
N_m = 1e4                   # Number of macro-particles per bunch for tracking
N_b = 2.3e11                # Bunch intensity [ppb]
N_t = 25                    # Number of turns to track
N_pretrack = 25             # Number of turns to pre-track
n_bunches = 72              # Number of bunches
bunch_spacing = 5           # In RF buckets
# CERN SPS --------------------------------------------------------------------
FILTER_DESIGN = True
FEEDFORWARD = True


# Plot settings
plt.rc('axes', labelsize=16, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)
# Colors
jet = plt.get_cmap('jet')
colors = jet(np.linspace(0, 1, N_t))

# Logger for messages on console & in file
Logger(debug=True)

np.set_printoptions(linewidth=70, precision=10)

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
logging.info("...... RF parameters set!")
T_s = bunch_spacing * rf.t_rf[0,0]
logging.info("RF period %.6e s", rf.t_rf[0, 0])
logging.info("Sampling period %.6e s", T_s)


if FILTER_DESIGN:

    logging.info("...... Filter design test")
    logging.info("3-section cavity filter design with modified filling time"+
                 " and fixed n_taps")
    TWC = SPS3Section200MHzTWC()
    TWC.tau = 420e-9
    filter, n_taps, n_filling, n_fit = feedforward_filter(TWC, 4/125 * 1e-6,
        taps=31, opt_output=True)
#    print(filter)

    logging.info("3-section cavity filter design")
    TWC_3 = SPS3Section200MHzTWC()
    filter = feedforward_filter(TWC_3, T_s)
#    print(filter)

    logging.info("4-section cavity filter design")
    TWC_4 = SPS4Section200MHzTWC()
    filter = feedforward_filter(TWC_4, T_s)
#    print(filter)

    logging.info("5-section cavity filter design")
    TWC_5 = SPS5Section200MHzTWC()
    filter = feedforward_filter(TWC_5, T_s)
#    print(filter)


if FEEDFORWARD:
    # Single bunch
    bunch = Beam(ring, N_m, N_b)
    bigaussian(ring, rf, bunch, 3.2e-9/4, seed=1234, reinsertion=True)
    logging.info("Bunch spacing %.3e s", rf.t_rf[0, 0] * bunch_spacing)

    # Create beam
    beam = Beam(ring, n_bunches * N_m, n_bunches * N_b)
    for i in range(n_bunches):
        beam.dt[int(i*N_m):int((i+1)*N_m)] = bunch.dt + (i + 10) * rf.t_rf[0, 0] * bunch_spacing
        beam.dE[int(i*N_m):int((i+1)*N_m)] = bunch.dE

    profile = Profile(beam, CutOptions=CutOptions(cut_left=0.e-9,
                                                  cut_right=rf.t_rf[0, 0] * (n_bunches * bunch_spacing + 200),
                                                  n_slices=2**5 * (n_bunches * bunch_spacing)))
    profile.track()
    logging.debug("Beam q/m ratio %.3e", profile.Beam.ratio)

    OTFB = SPSCavityFeedback(
        rf, profile, G_ff=1, G_llrf=10, G_tx=1,
        a_comb=63/64, turns=N_pretrack, post_LS2=False,
        commissioning=SPSCavityLoopCommissioning(debug=True, open_ff=False)
    )

    tracker = RingAndRFTracker(
        rf, beam, CavityFeedback=OTFB,
        interpolation=True, Profile=profile
    )

    if not os.path.exists("fig"):
        os.mkdir("fig")

    plot_long_phase_space(ring, rf, beam, 0, 5e-9, -2e8, 2e8,
                          dirname='fig', alpha=0.5, color=colors[0])

    map = [profile] + [tracker]

    # Plot 1: cavity voltage
    fig1 = plt.figure(4, figsize=(8, 10))
    gs1 = gridspec.GridSpec(2, 1)
    ax1_1 = plt.subplot(gs1[0])
    ax1_2 = plt.subplot(gs1[1], sharex=ax1_1)
    plt.setp(ax1_1.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1_1.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    ax1_1.set_ylabel(r"$Re(V_{\mathsf{cav}})$ [MV]")
    ax1_2.set_xlabel(r"Time [$\mu$s]")
    ax1_2.set_ylabel(r"$Im(V_{\mathsf{cav}})$ [MV]")
    ax1_1.set_ylim((-2.5, 5))
    ax1_2.set_ylim((-7, 0))
    ax1_1.plot(1e6 * profile.bin_centers, 1e-6 * OTFB.V_sum.real, color='grey')
    ax1_1.fill_between(1e6 * profile.bin_centers, 0, 1e-6 * OTFB.V_sum.real,
                       alpha=0.2, color='grey')
    ax1_2.plot(1e6 * profile.bin_centers, 1e-6 * OTFB.V_sum.imag, color='grey')
    ax1_2.fill_between(1e6 * profile.bin_centers, 0, 1e-6 * OTFB.V_sum.imag,
                       alpha=0.2, color='grey')

    # Plot 2: beam-induced voltage
    fig2 = plt.figure(5, figsize=(8, 10))
    gs2 = gridspec.GridSpec(2, 1)
    ax2_1 = plt.subplot(gs2[0])
    ax2_2 = plt.subplot(gs2[1], sharex=ax2_1)
    plt.setp(ax2_1.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax2_1.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    ax2_1.set_ylabel(r"$Re(V_{\mathsf{ind,beam}})$ [MV]")
    ax2_2.set_xlabel(r"Time [$\mu$s]")
    ax2_2.set_ylabel(r"$Im(V_{\mathsf{ind,beam}})$ [MV]")
    ax2_1.set_ylim((-2.5, 5))
    ax2_2.set_ylim((-1.8, 1.8))

    OTFB.OTFB_1.V_ind_beam = np.zeros(profile.n_slices)
    OTFB.OTFB_2.V_ind_beam = np.zeros(profile.n_slices)

    ax1_1.annotate('Beam in', xy=(0, 0), xytext=(0.95, 0.95),
                   textcoords='figure fraction', horizontalalignment='right',
                   verticalalignment='center')

    logging.info("...... Starting to track!")
    for i in range(N_t):

        logging.info("Turn %d", i)
        # Track
        for m in map:
            m.track()

        ax1_1.plot(1e6 * profile.bin_centers, 1e-6 * OTFB.V_sum.real,
                   color=colors[i])
        ax1_1.fill_between(1e6 * profile.bin_centers, 0,
                           1e-6 * OTFB.V_sum.real, alpha=0.2, color=colors[i])
        ax1_2.plot(1e6 * profile.bin_centers, 1e-6 * OTFB.V_sum.imag,
                   color=colors[i])
        ax1_2.fill_between(1e6 * profile.bin_centers, 0,
                           1e-6 * OTFB.V_sum.imag, alpha=0.2, color=colors[i])
        fig1.savefig("fig/V_ant_" + "%d" % (i + 1) + ".pdf")

        ax2_1.plot(1e6 * profile.bin_centers,
                   1e-6 * OTFB.OTFB_1.V_IND_FINE_BEAM.real + 1e-6 *
                   OTFB.OTFB_2.V_IND_FINE_BEAM.real,
                   color=colors[i])
        ax2_1.fill_between(1e6 * profile.bin_centers, 0,
                           1e-6 * OTFB.OTFB_1.V_IND_FINE_BEAM.real + 1e-6 *
                           OTFB.OTFB_2.V_IND_FINE_BEAM.real,
                           alpha=0.2, color=colors[i])
        ax2_2.plot(1e6 * profile.bin_centers,
                   1e-6 * OTFB.OTFB_1.V_IND_FINE_BEAM.imag + 1e-6 *
                   OTFB.OTFB_2.V_IND_FINE_BEAM.imag,
                   color=colors[i])
        ax2_2.fill_between(1e6 * profile.bin_centers, 0,
                           1e-6 * OTFB.OTFB_1.V_IND_FINE_BEAM.imag + 1e-6 * OTFB.OTFB_2.V_IND_FINE_BEAM.imag,
                           alpha=0.2, color=colors[i])
        fig2.savefig("fig/V_ind_beam_" + "%d" % (i + 1) + ".pdf")

        # This plot is slow, comment out to speed up
        plot_long_phase_space(ring, rf, beam, 0, 5e-9, -2e8, 2e8,
                              dirname='fig', alpha=0.5, color=colors[i])

logging.info("")
logging.info("Done!")