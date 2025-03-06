# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example input for simulation of acceleration
No intensity effects

:Authors: **Helga Timko**
"""
#  General Imports

import os
import time

import matplotlib as mpl
import numpy as np
from blond.utils import bmath as bm
from blond.beam.beam_distributed import BeamDistributedSingleNode
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.input_parameters.rf_parameters import RFStation

#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot
from blond.trackers.tracker import RingAndRFTracker

# To check if executing correctly, rather than to run the full simulation
DRAFT_MODE = False or bool(
    int(bool(int(os.environ.get("BLOND_EXAMPLES_DRAFT_MODE", False))))
)
mpl.use("Agg")

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

os.makedirs(this_directory + "../output_files/EX_01_fig", exist_ok=True)
def main(mock_n_gpus):
    N_t = 2000  # Number of turns to track
    dt_plt = 200  # Time steps between plots


    # Define general parameters
    ring = Ring(
        26658.883,
        1.0 / 55.759505**2,
        np.linspace(450e9, 460.005e9, N_t + 1),
        Proton(),
        N_t,
    )
    a = ring.ring_length
    # Define beam and distribution
    _beam = Beam(ring, 1001 if DRAFT_MODE else 50000, 1e9)


    # Define RF station parameters and corresponding tracker
    rf = RFStation(ring, [35640], [6e6], [0])

    bigaussian(ring, rf, _beam, 0.4e-9 / 4, reinsertion=True, seed=1)
    beam = BeamDistributedSingleNode(
        ring=ring,
        intensity=_beam.intensity,
        dE=_beam.dE,
        dt=_beam.dt,
        id=_beam.id,
        mock_n_gpus=mock_n_gpus,
    )
    print(f"{beam.n_gpus=}")
    long_tracker = RingAndRFTracker(rf, beam)

    # parabolic(ring, rf, beam, tau_0, seed=1)


    # Need slices for the Gaussian fit
    profile = Profile(
        beam, CutOptions(n_slices=100)
    )

    # Define what to save in file
    bunchmonitor = BunchMonitor(
        ring,
        rf,
        beam,
        this_directory + "../output_files/EX_01_output_data",
        profile=profile,
    )

    format_options = {"dirname": this_directory + "../output_files/EX_01_fig"}
    """plots = Plot(
        ring,
        rf,
        beam,
        dt_plt,
        N_t,
        0,
        0.0001763 * 35640,
        -400e6,
        400e6,
        xunit="rad",
        separatrix_plot=True,
        profile=profile,
        h5file=this_directory + "../output_files/EX_01_output_data",
        format_options=format_options,
    )"""

    # For testing purposes
    test_string = ""
    test_string += "{:<17}\t{:<17}\t{:<17}\t{:<17}\n".format(
        "mean_dE", "std_dE", "mean_dt", "std_dt"
    )
    test_string += "{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n".format(
        (beam.dE_mean()), (beam.dE_std()), (beam.dt_mean()), (beam.dt_std())
    )
    bm.use_gpu()
    locals_ = locals().copy()
    for thing in locals_:
        try:
            thing.to_gpu()
        except AttributeError:
            pass
    # Accelerator map
    map_ = [long_tracker] + [profile]
    print("Map set")
    print("")

    # Tracking --------------------------------------------------------------------
    if DRAFT_MODE:
        N_t = 20
    t0 = time.perf_counter()
    for i in range(1, N_t + 1):

        # Plot has to be done before tracking (at least for cases with separatrix)
        if (i % dt_plt) == 0:
            print("Outputting at time step %d..." % i)
            print("   Beam momentum %.6e eV" % beam.momentum)
            print("   Beam gamma %3.3f" % beam.gamma)
            print("   Beam beta %3.3f" % beam.beta)
            print("   Beam energy %.6e eV" % beam.energy)
            print(
                "   Four-times r.m.s. bunch length %.4e s" % (4.0 * beam.sigma_dt)
            )
            #print("   Gaussian bunch length %.4e s" % profile.bunchLength)
            print("")

        # Track
        for m in map_:
            m.track()

        # Define losses according to separatrix and/or longitudinal position
        beam.losses_separatrix(ring, rf)
        beam.losses_longitudinal_cut(0.0, 2.5e-9)
    t1 = time.perf_counter()
    print(t1-t0,"s runtime")

    # For testing purposes
    """
    test_string += "{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n".format(
        np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt)
    )
    with open(this_directory + "../output_files/EX_01_test_data.txt", 
              "w") as f:
        f.write(test_string)"""

    print("Done!")

if __name__ == "__main__":
    main(mock_n_gpus=1)
    main(mock_n_gpus=None)
