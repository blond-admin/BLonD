# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

# pragma: no cover
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    DriftSimple,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.experimental import ProfileMatcherAddon, SemiEmpiricMatcher
from blond.generals.cupy.no_cupy_import import copy_to_cpu

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore


def get_test_profile(noisy=False):
    """
    Create a histogram that looks like a measurement.

    Parameters
    ----------
    noisy
        If it should be generated with or without noise.

    Returns
    -------
    hist_x
        Histogram time axis.
    hist_y
        Histogram amplitude.
    """
    # Parameters
    mean = 2.5e-9 / 2  # Mean of the distribution
    std_dev = 2.5e-9 / 30  # Standard deviation
    size = 10000  # Number of data points

    if noisy:
        # Generate random data from a Gaussian distribution
        data = np.random.normal(loc=mean, scale=std_dev, size=size)

        # Get the histogram (density=False for raw counts)
        hist_y, bin_edges = np.histogram(data, bins=512, density=False)
        hist_x = bin_edges[0:-1] + np.diff(bin_edges[:2])[0] / 2
    else:
        hist_x = np.linspace(*(0, 2.5e-9))
        hist_y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -((hist_x - mean) ** 2) / (2 * std_dev**2)
        )
    return hist_x, hist_y


N_TURNS = int(1e3)
animate_fitting = True
plot_result = True
n_macroparticles = 1e6


def main():
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf_design = 0

    values = np.linspace(450e9, 455e9, int(1e3) + 1)
    energy_cycle = MagneticCyclePerTurn(
        value_init=values[0],
        values_after_turn=values[1:],
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    hist_x, hist_y = get_test_profile()
    hist_y = hist_y
    matcher_addon = ProfileMatcherAddon(hist_x=hist_x, hist_y=hist_y)
    matcher_addon.smoothness = 0.01
    matcher_addon.atol = 1e-3
    matcher_addon.recenter = True
    matcher_addon.animate_fitting = animate_fitting
    matcher_addon.plot_result = plot_result
    matcher_addon.plot_result_blocking = False

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.5e-9),
            n_macroparticles=n_macroparticles,
            seed=0,
            maxiter_intensity_effects=0,
            internal_grid_shape=(1024, 1024),
            hamilton_to_density_function=matcher_addon.hamilton_to_density_function,
            hamilton_to_density_kwargs=dict(),
            animate=True,
        ),
    )

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value % 10 != 0:
            return

        plt.hist2d(
            copy_to_cpu(beam.read_partial_dt()),
            copy_to_cpu(beam.read_partial_dE()),
            bins=256,
            range=[(0, 2.5e-9), (-4e8, 4e8)],
        )
        plt.xlim((0, 2.5e-9))
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            beams=(beam1,),
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
        print(
            f"Loaded {phase_observation.common_filepath}"
        )  # pragma: no cover
    except (FileNotFoundError, AssertionError):
        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS,
            # observe=(phase_observation, bunch_observation),
            callbacks=custom_action,
        )
    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                bins=256,
                range=[[0, 2.5e-9], [-4e8, 4e8]],
            )
            plt.draw()
            plt.pause(0.1)

        plt.show()
    return beam1


if __name__ == "__main__":  # pragma: no cover
    main()
