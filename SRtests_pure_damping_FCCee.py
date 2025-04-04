from __future__ import division, print_function
import os
import matplotlib.pyplot as plt
import json
import pickle as pkl
import datetime
import numpy as np

from blond.beam.beam import Beam, Electron
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.distributions import bigaussian, parabolic
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from blond.beam.distributions import matched_from_distribution_function
from ramp_modules.Ramp_optimiser_functions import HEBee_Eramp_parameters, optimizer_voltage_ramp
from ring_parameters.generate_rings import generate_HEB_ring
from scipy.constants import c, e
from ramp_modules.energy_ramp_optimiser import optimise_energy_ramp
import ramp_analysers as ra
#from Acceleration_simulations.acceleration_blond import track_through_acceleration

test_mode = False
optimise = False
verbose  = False
# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 1001 if test_mode else  50000  # Macro-particles
# Machine and RF parameters
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1. / gamma_t / gamma_t        # First order mom. comp. factor

# Tracking details
Nturns = 1000           # Number of turns to track
dt_plt = 5         # Time steps between plots
# Tracking details GET THAT FOR HBE

time_str = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
# Ramp based on time and not on turns
# Conversion from time to turns to be done

tracking_parameters = HEBee_Eramp_parameters(op_mode='Z', dec_mode = True)
ring_HEB = generate_HEB_ring(op_mode='Z', Nturns=Nturns, momentum=20e9*np.ones(Nturns+1))
beam = Beam(ring_HEB, N_p, N_b)
rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, 45e6, phi_rf_d= 0)
long_tracker = RingAndRFTracker(rfcav, beam)
print(f"Synchrotron radiation taken into account: {rfcav.sr_flag}")

"""From the Synchrotron Radiation example"""
full_tracker = FullRingAndRF([long_tracker])
matched_from_distribution_function(beam, full_tracker, emittance=0.02,
                                  distribution_type='gaussian',
                                  distribution_variable='Hamiltonian',
                                  seed=1000)
#matched_from_distribution_function(beam, full_tracker, emittance=(np.pi * tracking_parameters.sigmaz_0 / c * tracking_parameters.sigmaE_0 * tracking_parameters.E_flat_bottom),
#                                  distribution_type='gaussian',
#                                  distribution_variable='Hamiltonian',
#                                  seed=1000)
