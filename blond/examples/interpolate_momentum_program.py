# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from blond import (
    Beam,
    DriftSimple,
    EmptyBeam,
    MagneticCycleByTime,
    ReferenceEnergyChange,
    Ring,
    Simulation,
    backend,
    proton,
)
from blond.handle_results.helpers import callers_relative_path


def main():
    filename, injection_time, momentum = from_file()

    start_time = 1500e-3
    injection_time, momentum = truncate_data(
        injection_time, momentum, start_time
    )

    cycle = MagneticCycleByTime(
        reference_particle=proton,
        base_time=injection_time,
        base_values=momentum,
        in_unit="momentum",
        interpolator=PchipInterpolator,
        # # alternatively
        # interpolator=scipy.interpolate.Akima1DInterpolator,
        # method="makima",
    )
    ring = Ring(circumference=6911.5038)
    momentum_compaction_factor = 1  # not required for `EmptyBeam`
    ring.add_elements(
        (
            DriftSimple(
                orbit_length=ring.circumference,
                momentum_compaction_factor=momentum_compaction_factor,
            ),
            ReferenceEnergyChange(),
        ),
    )

    simulation = Simulation(ring=ring, magnetic_cycle=cycle)
    beam = EmptyBeam(
        particle_type=cycle.reference_particle,
        reference_total_energy=cycle.get_total_energy_init(
            particle_type=cycle.reference_particle
        ),
        reference_time=0.0,
    )
    # todo better automatic of turn number until interpolation fails.
    N_TURNS = 326786
    result = np.empty((int(N_TURNS), 2), dtype=float)

    def my_callback(simulation: Simulation, beam: Beam) -> None:
        result[simulation.turn_i.value, 0] = beam.reference.time
        result[simulation.turn_i.value, 1] = beam.reference.total_energy

    backend.set_specials("python")

    simulation.run_simulation(
        beams=beam, n_turns=result.shape[0], callbacks=my_callback
    )

    np.savetxt(
        filename.replace(".csv", "_interpolated.csv"),
        result,
        header="time[s] total_energy[eV]",
    )
    ax = plt.subplot(2, 1, 1)
    plt.xlabel("Time [s]")
    plt.ylabel("Momentum [eV]")
    plt.plot(result[:, 0], result[:, 1], "o")
    plt.subplot(2, 1, 2, sharex=ax)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta$ Momentum [eV]")
    plt.plot(result[:-1, 0], np.diff(result[:, 1]), "o")
    plt.tight_layout()
    plt.show()


def truncate_data(injection_time, momentum, start_time):
    start_index = np.argmax(injection_time > start_time)
    injection_time = injection_time[start_index:-2]
    injection_time -= injection_time[0]
    momentum = momentum[start_index:-2]
    assert not np.any(np.isnan(injection_time))
    assert not np.any(np.isnan(momentum))
    return injection_time, momentum


def from_file():
    filename = "Momentum.csv"
    momentum_ramp_data_path = callers_relative_path(
        "resources/%s" % filename, stacklevel=1
    )
    momentum_data = np.genfromtxt(
        momentum_ramp_data_path, delimiter=",", skip_header=2
    ).transpose()
    injection_time = momentum_data[0] * 1e-3
    momentum = momentum_data[1] * 1e9
    return filename, injection_time, momentum


if __name__ == "__main__":  # pragma: no cover
    main()
