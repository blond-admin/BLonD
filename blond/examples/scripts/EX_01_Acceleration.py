# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
import matplotlib

matplotlib.use("Qt5Agg")

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
    MultiHarmonicRFStation,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.experimental import (
    SemiEmpiricMatcher,
)

logging.basicConfig(level=logging.INFO)


def main():
    ring = Ring(26658.883)

    rf_station = MultiHarmonicRFStation(n_harmonics=2, main_harmonic_idx=0)
    rf_station.harmonic = np.array([35640, 4 * 35640])
    rf_station.voltage = np.array([6e6, 6e6 / 2])
    rf_station.phi_rf_design = np.array([0, 0])

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn.init_from_linspace(
        values=np.linspace(450e9, 450e9, N_TURNS + 1),
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
    BIGAUS = True
    if BIGAUS:
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=1e3,
            ),
        )
    else:  # pragma: no cover
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=SemiEmpiricMatcher(
                time_limit=(0, 2.5e-9),
                n_macroparticles=1e6,
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

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value % 10 != 0:
            return

        dt = beam.read_partial_dt()
        plt.scatter(
            dt,
            beam.read_partial_dE(),
            s=1,
        )
        t0 = dt.min()
        t1 = dt.max()
        trange = t1 - t0

        sim.plot_separatrix(
            beam=beam,
            dt=np.linspace(t0 - trange, t1 + trange, 1000),
        )
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
            observe=(phase_observation, bunch_observation),
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


if __name__ == "__main__":  # pragma: no cover
    main()
