# pragma: no cover
import logging
import os

import numpy as np

from blond import (
    Beam,
    BeamObservationInRingElement,
    BiGaussian,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.handle_results.helpers import callers_relative_path

logging.basicConfig(level=logging.INFO)


def main():
    try:
        os.mkdir(callers_relative_path("./results/", stacklevel=1))
    except:
        pass

    ring = Ring(26658.883)

    cavity1 = SingleHarmonicRfStation()
    cavity1.harmonic = 35640
    cavity1.voltage = 6e6
    cavity1.phi_rf = 0

    N_TURNS = int(10)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.transition_gamma = 55.759505
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
        cavity1,
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
