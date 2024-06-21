# General imports
import matplotlib.pyplot as plt
import numpy as np
import sys
import time as tm

#  BLonD Imports
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian, parabolic
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
import blond.utils.turn_counter as tc
import blond.utils.track_iteration as trackIter

#%% Parameter definition
N_p = 50000         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 1000           # Number of turns to track

#%% Machine prep

ring = Ring([C/5]*5, alpha, p_i, Proton(), N_t, n_sections = 5,
            counter_name='Simulation')
beam = Beam(ring, N_p, 0)

rf_sections = []
long_trackers = []
for i in range(5):
    rf = RFStation(ring, [h], [V], [dphi], section_index=i+1)
    rf_sections.append(rf)
    long_tracker = RingAndRFTracker(rf, beam)
    long_trackers.append(long_tracker)


full_ring = FullRingAndRF(long_trackers)

profile = Profile(beam, CutOptions(n_slices=100))

#%% Bunch generation

bigaussian(ring, rf, beam, tau_0 / 4, reinsertion=True, seed=1)
beam.dE *= 1.5

#%% Tracking

track_map = [full_ring, profile]
track_it = trackIter.TrackIteration(track_map, counter_name='Simulation')
counter = tc.get_turn_counter("Simulation")

for t, s in track_it:

    if t % 20 == 0:
        print(counter)
        plt.scatter(beam.dt*1E9, beam.dE/1E9)
        plt.xlim([0, 2.5])
        plt.ylim([-0.5, 0.5])
        plt.xlabel("dt (ns)")
        plt.ylabel("dE (GeV)")
        plt.show()
