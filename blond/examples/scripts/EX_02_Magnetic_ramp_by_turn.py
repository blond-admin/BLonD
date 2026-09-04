# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    ResultsFormatError,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.experimental import (
    SemiEmpiricMatcher,
)
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

logging.basicConfig(level=logging.INFO)


def main(
    n_turns=int(1e3),
    n_macroparticles=int(1e3),
):
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf_design = 0

    energy_cycle = MagneticCyclePerTurn.init_from_linspace(
        values=np.linspace(450e9, 450e9, n_turns + 1),
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

    ring.add_elements((drift1, rf_station))
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    # or alternatively to automatically discover the variables
    # sim = Simulation.from_locals(locals())

    sim.print_one_turn_execution_order()
    BIGAUS = True
    if BIGAUS:
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=n_macroparticles,
            ),
        )
    else:  # pragma: no cover
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(0, 2.5e-9),
                n_macroparticles=n_macroparticles,
                seed=0,
                maxiter_intensity_effects=0,
                hamilton_to_density_kwargs=dict(
                    density_modifier=2.0,  # Controls density profile sharpness
                    hamilton_max=40.0,  # Hamiltonian cutoff [eV]
                ),
                animate=True,
            ),
        )

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)

    def animate_live(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_counter.value % 10 != 0:
            return

        plt.scatter(
            copy_to_cpu(beam.read_partial_dt()),
            copy_to_cpu(beam.read_partial_dE()),
        )

        sim.plot_separatrix(
            beam=beam,
            dt=np.linspace(beam.dt.min(), beam.dt.max(), 1000),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            beams=(beam1,),
            n_turns=(n_turns),
            observe=(phase_observation, bunch_observation),
        )
        print(
            f"Loaded {phase_observation.common_filepath}"
        )  # pragma: no cover
    except (FileNotFoundError, AssertionError, ResultsFormatError):
        sim.run_simulation(
            beams=(beam1,),
            n_turns=(n_turns),
            observe=(phase_observation, bunch_observation),
            callbacks=animate_live,
        )
    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(n_turns):
            plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                bins=256,
                range=[[0, 2.5e-9], [-4e8, 4e8]],
            )
            plt.draw()
            plt.pause(0.1)


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
