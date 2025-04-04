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

particle_type = Electron()
n_particles = int(1.7e11)
n_macroparticles = int(1e5)

Nturns = 1000           # Number of turns to track

tracking_parameters = HEBee_Eramp_parameters(op_mode='Z', dec_mode = True)
ring_HEB = generate_HEB_ring(op_mode='Z', Nturns=Nturns, momentum=50e9*np.ones(Nturns+1))

beam = Beam(ring_HEB, n_macroparticles, n_particles)
rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, 70e6, phi_rf_d= np.pi)
bigaussian(ring_HEB, rfcav, beam, tracking_parameters.sigmaz_0 / c / 4, sigma_dE = tracking_parameters.sigmaE_0 * tracking_parameters.E_flat_bottom , reinsertion=True, seed=1)
number_slices = 500

plt.scatter(beam.dt, beam.dE)
plt.show()

cut_options = CutOptions(cut_left=0., cut_right=tracking_parameters.sigmaz_0 / c / 4, n_slices=number_slices)
slice_beam = Profile(beam, CutOptions=cut_options)
# Trackers
long_tracker = RingAndRFTracker(rfcav, beam)
print(f"Synchrotron radiation taken into account: {rfcav.sr_flag}")

full_tracker = FullRingAndRF([long_tracker])
#matched_from_distribution_function(beam, full_tracker, emittance=0.02,
#                                  distribution_type='gaussian',
#                                  dt_margin_percent=1,
#                                  distribution_variable='Hamiltonian',
#                                  seed=1000)

SR = [SynchrotronRadiation(ring_HEB, rfcav, beam, quantum_excitation=True, python=True,shift_beam=True )]
SR[0].print_SR_params()

map_ = [long_tracker] + SR #+ [slice_beam]

bl = []
eml = []
for i in range(1, Nturns + 1):
    bl.append(4. * beam.sigma_dt * c * 1e3)
    eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
    expected_eml = 0
    expected_bl = 0
    # Track
    for m in map_:
        m.track()
    print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))

fig, ax = plt.subplots()

ax.plot(eml)

plt.show()