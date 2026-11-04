# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Simulate a cycle given as a JSON-style dict."""

from blond import Beam, BiGaussian, Simulation
from blond.interfaces.portable import CycleDefinition

n_turns = 1000
n_macroparticles = 1e5

cycle_dict = {
    "circumference": 26658.883,
    "magnetic_cycle": {
        "time": [0.0],
        "value": [450e9],
        "particle": {"mass": 938272089.4282573, "charge": 1},
        "in_unit": "momentum",
    },
    "rf_stations": [
        {
            "harmonics": [
                {
                    "harmonic": {"time": [0.0], "value": [35640]},
                    "voltage": {"time": [0.0], "value": [6e6]},
                    "phi_rf_design": {"time": [0.0], "value": [0.0]},
                }
            ],
            "name": "rf",
        }
    ],
    "drifts": [
        {
            "orbit_length": 26658.883,
            "momentum_compaction_factor": {
                "time": [0.0],
                "value": [3.216e-4],
            },
            "name": "drift",
        }
    ],
}


def main():
    cycle = CycleDefinition.model_validate(cycle_dict)

    ring = cycle.to_ring()
    magnetic_cycle = cycle.magnetic_cycle.to_magnetic_cycle()
    beam = Beam(
        intensity=1e9,
        particle_type=cycle.magnetic_cycle.particle.to_particle_type(),
    )

    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    sim.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            sigma_dt=0.1e-9, n_macroparticles=n_macroparticles
        ),
    )
    sim.run_simulation(beams=(beam,), n_turns=n_turns)


if __name__ == "__main__":  # pragma: no cover
    main()
