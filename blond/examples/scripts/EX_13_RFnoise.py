# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example input for simulation with RF noise.

Notes
-----
No intensity effects.

Authors:
Simon Lauber
Helga Timko
"""
# pragma: no cover

import os
from importlib.resources import as_file, files

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BeamStatisticsOncePerTurn,
    BiGaussian,
    BoxLosses,
    DriftSimple,
    DynamicProfileConstNBins,
    MagneticCyclePerTurn,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.cycles.noise_generators import VariNoise
from blond.specifics.cern.lhc.varinoise import lhc_spectrum_gain_y
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"


def main():
    n_turns = 20000
    ring = Ring(circumference=26658.883)
    rf_station_1 = SingleHarmonicRFStation()
    rf_station_1.voltage = 6e6
    # Band-limited phase noise: a fixed frequency band along the cycle, with an
    # (LHC) spectral shape supplied by the caller. Requires the external
    # rf-noise-cpp library.
    gain_y = lhc_spectrum_gain_y
    noise = VariNoise(
        frequency_high=np.full(n_turns, 200.0),
        frequency_low=np.full(n_turns, 100.0),
        gain_y=gain_y,
        sampling_rate=11245.49,
        rms=0.1,
    )
    rf_station_1.schedule(
        attribute="phi_rf_design",
        value=noise.get_noise(n_turns=n_turns),
    )
    rf_station_1.harmonic = 35640

    energy_cycle = MagneticCyclePerTurn(
        value_init=450.0e9,
        values_after_turn=np.linspace(450.0e9, 450.0e9, n_turns),
        reference_particle=proton,
    )

    beam = Beam(
        intensity=1.0e9,
        particle_type=proton,
    )

    profile = DynamicProfileConstNBins(n_bins=100)

    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=55.759505
        ),
        orbit_length=ring.circumference,
    )

    t_rf = (
        energy_cycle.get_t_rev_init(ring.circumference) / rf_station_1.harmonic
    )

    losses = BoxLosses(
        purge_flagged_macroparticles=True, t_min=-t_rf, t_max=t_rf
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()
    sim.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            n_macroparticles=1001,
            sigma_dt=0.4e-9 / 4,
            reinsertion=True,
            seed=1,
        ),
    )
    beam_obs = BeamStatisticsOncePerTurn(each_turn_i=1)
    rf_obs = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station_1,
    )

    def callback(sim: Simulation, beam: Beam) -> None:  # pragma: no cover
        plt.cla()
        beam.plot_hist2d()
        plt.draw()
        plt.pause(0.01)

    sim.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        observe=(rf_obs, beam_obs),
        # callbacks=callback
    )

    plt.figure("phase")
    ax = plt.subplot(2, 1, 1)
    plt.plot(rf_obs.turns_array, rf_obs.phases)
    plt.ylabel("RF Station Phase [rad]")
    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(beam_obs.turns_array, beam_obs.bunch_length * 1e9)
    plt.ylabel("Bunch Length [nm]")
    plt.xlabel("Turn")


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
