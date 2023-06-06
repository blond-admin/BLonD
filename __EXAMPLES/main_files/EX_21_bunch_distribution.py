
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for toolbox.action oscillation_amplitude_from_coordinates()

:Authors: **Helga Timko**
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.toolbox.action import oscillation_amplitude_from_coordinates

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_21_fig/', exist_ok=True)


# LHC parameters --------------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 1000000         # Macro-particles
tau_0 = 1.0e-9      # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p = 450e9         # Synchronous momentum [eV/c]
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track


# Simulation setup ------------------------------------------------------------
ring = Ring(C, alpha, p, Proton(), N_t)
rf = RFStation(ring, [h], [V], [dphi])
beam = Beam(ring, N_p, N_b)
bigaussian(ring, rf, beam, tau_0 / 4, reinsertion=True, seed=1)
profile = Profile(beam, CutOptions=CutOptions(n_slices=100, cut_left=0,
                                              cut_right=2.5e-9))
profile.track()

# Calculate oscillation amplitude from coordinates
dtmax, bin_centres, histogram = oscillation_amplitude_from_coordinates(ring,
                                                                       rf, beam.dt, beam.dE, Np_histogram=100)

# Normalise profiles
profile.n_macroparticles /= np.sum(profile.n_macroparticles)
histogram /= np.sum(histogram)

# Plot
plt.plot(profile.bin_centers, profile.n_macroparticles, 'b',
         label=r'$\lambda(t)$')
plt.plot(bin_centres + 1.25e-9, histogram, 'r',
         label=r'$\lambda(t_{\mathsf{max}})$')
plt.plot(profile.bin_centers[51:], profile.n_macroparticles[51:] * 2 * 1.41 *
         np.sin(2 * np.pi * 400e6 * (profile.bin_centers[51:] - 1.25e-9) / 1.41),
         'g', label=r'$\lambda(t)*\sin{(\omega_{\mathsf{rf}}t/\sqrt{2})}$')
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Particle density [1/s]")
plt.savefig(this_directory + '../output_files/EX_21_fig/profiles.png')

print("Done!")
