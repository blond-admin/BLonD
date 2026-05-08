# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
PSB-like ramp simulation using time-interpolated momentum programs.

Demonstrates MagneticCycleByTime as the new-API equivalent of the legacy
preprocess_ramp workflow. The momentum vs. time array is interpolated
just-in-time each turn via PchipInterpolator, enabling smooth ramps from
arbitrary data tables.

Notes
-----
Authors:
Danilo Quartullo (legacy EX_06)
Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCycleByTime,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.handle_results.observables import BeamStatisticsOncePerTurn

N_TURNS = 200


def main():
    circumference = 2 * np.pi * 25.0  # PSB circumference [m]

    # Synthetic momentum ramp: 1.4 → 2.0 GeV/c over 400 ms.
    # In a real use case this array would be loaded from a measurement file.
    time_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # [s]
    momentum_points = np.array([1.4e9, 1.5e9, 1.7e9, 1.9e9, 2.0e9])  # [eV/c]

    ring = Ring(circumference=circumference)
    energy_cycle = MagneticCycleByTime(
        reference_particle=proton,
        base_time=time_points,
        base_values=momentum_points,
        in_unit="momentum",
        interpolator=PchipInterpolator,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=1,
        voltage=8e3,
        phi_rf=np.pi,
    )
    drift = DriftSimple(orbit_length=circumference)
    drift.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=4.076750841
    )
    beam = Beam(intensity=1e11, particle_type=proton)

    sim = Simulation.from_locals(locals())
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=100e-9 / 4,
            reinsertion=False,
            seed=1,
            n_macroparticles=1001,
        ),
        beam=beam,
    )

    bunch_obs = BeamStatisticsOncePerTurn(each_turn_i=10)
    sim.run_simulation(
        beams=(beam,),
        n_turns=N_TURNS,
        observe=(bunch_obs,),
    )

    bunch_length = bunch_obs.bunch_length
    with AllowPlotting():
        plt.figure()
        plt.plot(bunch_length * 1e9, label="bunch length [ns]")
        plt.xlabel("Observation index (every 10 turns)")
        plt.ylabel("Bunch length [ns]")
        plt.title("PSB ramp: bunch length over 200 turns")


if __name__ == "__main__":  # pragma: no cover
    setup_backend("auto")
    main()
    plt.show()
