# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Test case to show the consequences of omega_rf != h*omega_rev
(CERN PS Booster context).

:Authors: **Danilo Quartullo**
"""

import os

import matplotlib as mpl
import numpy as np

from blond.legacy.blond2.utils import bmath as bm
from blond.legacy.blond2.beam.beam import Beam, Proton
from blond.legacy.blond2.beam.distributions import matched_from_distribution_function
from blond.legacy.blond2.beam.profile import CutOptions, Profile
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.legacy.blond2.llrf.beam_feedback import BeamFeedback
from blond.legacy.blond2.monitors.monitors import BunchMonitor
from blond.legacy.blond2.plots.plot import Plot
from blond.legacy.blond2.trackers.tracker import FullRingAndRF, RingAndRFTracker

DRAFT_MODE = bool(int(os.environ.get("BLOND_EXAMPLES_DRAFT_MODE", False)))
# To check if executing correctly, rather than to run the full simulation

mpl.use("Agg")


this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

USE_GPU = os.environ.get("USE_GPU", "0")
if len(USE_GPU) and int(USE_GPU):
    USE_GPU = True
else:
    USE_GPU = False

os.makedirs(this_directory + "../gpu_output_files/EX_10_fig", exist_ok=True)


# Beam parameters
n_macroparticles = 1001 if DRAFT_MODE else 100000
n_particles = 0


# Machine and RF parameters
radius = 25  # [m]
gamma_transition = 4.076750841  # [1]
alpha = 1 / gamma_transition**2  # [1]
C = 2 * np.pi * radius  # [m]

n_turns = 10000

general_params = Ring(C, alpha, 310891054.809, Proton(), n_turns)
# Cavities parameters
n_rf_systems = 1
harmonic_numbers_1 = 1  # [1]
voltage_1 = 8000  # [V]
phi_offset_1 = np.pi  # [rad]
rf_params = RFStation(
    general_params,
    [harmonic_numbers_1],
    [voltage_1],
    [phi_offset_1],
    n_rf_systems,
    omega_rf=[1.00001 * 2.0 * np.pi / general_params.t_rev[0]],
)

my_beam = Beam(general_params, n_macroparticles, n_particles)


cut_options = CutOptions(cut_left=0, cut_right=2.0 * 0.9e-6, n_slices=200)
slices_ring = Profile(my_beam, cut_options)


# Phase loop
configuration = {
    "machine": "PSB",
    "PL_gain": 0.0,
    "RL_gain": [0.0, 0.0],
    "period": 10.0e-6,
}
phase_loop = BeamFeedback(
    general_params, rf_params, slices_ring, configuration
)


# Long tracker
long_tracker = RingAndRFTracker(rf_params, my_beam, beam_feedback=phase_loop)

full_ring = FullRingAndRF([long_tracker])

distribution_type = "gaussian"
bunch_length = 200.0e-9
distribution_variable = "Action"

matched_from_distribution_function(
    my_beam,
    full_ring,
    bunch_length=bunch_length,
    distribution_type=distribution_type,
    distribution_variable=distribution_variable,
    seed=3,
)
slices_ring.track()


# Monitor
bunch_monitor = BunchMonitor(
    general_params,
    rf_params,
    my_beam,
    this_directory + "../gpu_output_files/EX_10_output_data",
    profile=slices_ring,
    phase_loop=phase_loop,
)


# Plots
format_options = {"dirname": this_directory + "../gpu_output_files/EX_10_fig"}
plots = Plot(
    general_params,
    rf_params,
    my_beam,
    1000,
    10000,
    0.0,
    2.0 * 0.9e-6,
    -1.0e6,
    1.0e6,
    separatrix_plot=True,
    profile=slices_ring,
    format_options=format_options,
    h5file=this_directory + "../gpu_output_files/EX_10_output_data",
    phase_loop=phase_loop,
)

# For testing purposes
test_string = ""
test_string += "{:<17}\t{:<17}\t{:<17}\t{:<17}\n".format(
    "mean_dE", "std_dE", "mean_dt", "std_dt"
)
test_string += "{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n".format(
    np.mean(my_beam.dE),
    np.std(my_beam.dE),
    np.mean(my_beam.dt),
    np.std(my_beam.dt),
)

# Accelerator map
map_ = [long_tracker] + [slices_ring]  # + [bunch_monitor] + [plots]

if USE_GPU:
    bm.use_gpu()
    my_beam.to_gpu()
    long_tracker.to_gpu()
    slices_ring.to_gpu()
    phase_loop.to_gpu()
    rf_params.to_gpu()

if DRAFT_MODE:
    n_turns = 10
    n_plot = 3
else:
    n_plot = 100
for i in range(1, n_turns + 1):
    for m in map_:
        m.track()

    if i % 1000 == 0:
        if USE_GPU:
            bm.use_cpu()
            my_beam.to_cpu()
            long_tracker.to_cpu()
            slices_ring.to_cpu()
            phase_loop.to_cpu()
            rf_params.to_cpu()

        plots.track()

        if USE_GPU:
            bm.use_gpu()
            my_beam.to_gpu()
            long_tracker.to_gpu()
            slices_ring.to_gpu()
            phase_loop.to_gpu()
            rf_params.to_gpu()

    slices_ring.cut_options.track_cuts(my_beam)
    slices_ring.set_slices_parameters()

    if i % 100 == 0:
        print("Time step %d" % i)
        print("    Radial error %.4e" % (phase_loop.drho))
        print(
            "    Radial loop frequency correction %.4e 1/s"
            % (phase_loop.domega_rf)
        )
        print("    RF phase %.4f rad" % (rf_params.phi_rf[0, i]))
        print("    RF frequency %.6e 1/s" % (rf_params.omega_rf[0, i]))
        print(
            "    Tracker phase %.4f rad"
            % (long_tracker.rf_params.phi_rf[0, i])
        )
        print(
            "    Tracker frequency %.6e 1/s"
            % (long_tracker.rf_params.omega_rf[0, i])
        )

if USE_GPU:
    bm.use_cpu()
    my_beam.to_cpu()
    long_tracker.to_cpu()
    slices_ring.to_cpu()
    phase_loop.to_cpu()
    rf_params.to_cpu()

print("dE mean: ", np.mean(my_beam.dE))
print("dE std: ", np.std(my_beam.dE))
print("profile mean: ", np.mean(slices_ring.n_macroparticles))
print("profile std: ", np.std(slices_ring.n_macroparticles))

# For testing purposes
test_string += "{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n".format(
    np.mean(my_beam.dE),
    np.std(my_beam.dE),
    np.mean(my_beam.dt),
    np.std(my_beam.dt),
)
with open(
    this_directory + "../gpu_output_files/EX_10_test_data.txt", "w"
) as f:
    f.write(test_string)

print("Done!")
