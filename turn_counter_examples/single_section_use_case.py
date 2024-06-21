# General imports
import matplotlib.pyplot as plt
import numpy as np
import sys

#  BLonD Imports
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian, parabolic
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import RingAndRFTracker
import blond.utils.turn_counter as tc
import blond.utils.track_iteration as trackIter


#%% Parameter definition
N_p = 50000         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
h = 35640            # Harmonic number

dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 1000           # Number of turns to track

V = [6E6]*500 + [12E6]*501

#%% Machine prep

ring = Ring(C, alpha, p_i, Proton(), N_t)
beam = Beam(ring, N_p, 0)
rf = RFStation(ring, [h], V, [dphi])
long_tracker = RingAndRFTracker(rf, beam)
profile = Profile(beam, CutOptions(n_slices=100))

#%% Bunch generation

bigaussian(ring, rf, beam, tau_0 / 4, reinsertion=True, seed=1)
beam.dE *= 1

#%% Tracking

track_map = [long_tracker, profile]
track_it = trackIter.TrackIteration(track_map)
counter = tc.get_turn_counter()

print(counter)
track_it(500)
print(counter)

shortest_turn = counter.current_turn
shortest_length = np.max(beam.dt) - np.min(beam.dt)

print(shortest_length*1E9, shortest_turn)

for t in track_it:
    length = np.max(beam.dt) - np.min(beam.dt)
    if length < shortest_length:
        shortest_length = length
        shortest_turn = counter.current_turn

print(shortest_length*1E9, shortest_turn)
