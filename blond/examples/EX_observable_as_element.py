# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover
import logging
import os

import numpy as np

from blond import (
    Beam,
    BeamObservationInRingElement,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.handle_results.helpers import callers_relative_path

logging.basicConfig(level=logging.INFO)


def main():
    try:
        os.mkdir(callers_relative_path("./results/", stacklevel=1))
    except:
        pass

    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf = 0

    N_TURNS = int(10)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
        momentum_compaction_factor=1 / (55.759505**2),
    )
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    beam_logger_element = BeamObservationInRingElement(
        name="logger",
        section_index=0,
        folder=callers_relative_path(
            "./results/",
            stacklevel=1,
        ),
    )

    one_turn_execution_order = (
        drift1,
        rf_station,
        beam_logger_element,
    )
    ring.add_elements(one_turn_execution_order)

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,
            reinsertion=True,
            seed=1,
            n_macroparticles=10,
        ),
        beam=beam1,
    )

    sim.print_one_turn_execution_order()

    sim.run_simulation(
        n_turns=10,
        beams=(beam1,),
    )

    beam_logger_element.to_disk()


if __name__ == "__main__":  # pragma: no cover
    main()
