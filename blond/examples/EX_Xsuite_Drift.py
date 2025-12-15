import logging

import numpy as np
import xobjects as xo
import xtrack as xt
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    BunchObservation,
    CavityPhaseObservation,
    DriftXSuite,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    lead_ion,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.handle_results.helpers import callers_relative_path

logging.basicConfig(level=logging.INFO)


def main():
    xsuite_folder = callers_relative_path(
        "./resources/xsuite_lines/SPS_2021_Pb_nominal.json", stacklevel=1
    )
    ctx = xo.ContextCpu()
    line = xt.Line.from_json(f"{xsuite_folder}")
    line.cycle(
        name_first_element="actcse.31632"
    )  # make the start at the cavity

    voltage = 3e6
    line["actcse.31632"].voltage = 0  # xsuite cavity =0, only use drift

    ring = Ring(2 * np.pi * 1100.009)

    momentum = 1.4e12

    slip_factor = -0.017166999
    mom_compaction = 0.0015175

    cavity1 = SingleHarmonicCavity()
    cavity1.harmonic = 4620
    cavity1.voltage = voltage
    cavity1.phi_rf = 0

    N_TURNS = int(1e3)

    beam1 = Beam(
        intensity=1.5e11,
        particle_type=lead_ion,
    )

    energy_cycle = MagneticCyclePerTurn(
        value_init=momentum,
        values_after_turn=np.linspace(momentum, momentum, N_TURNS),
        reference_particle=lead_ion,
    )

    drift1 = DriftXSuite(
        line=line,
        beam=beam1,
        phi_s=0,
        omega_rf=200e6,
        energy0=momentum,
        beta0=0.6,  # TODO
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    BIGAUS = True
    if BIGAUS:
        sim.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=3e-9 / 2,
                sigma_dE=0.1e-8,
                reinsertion=False,
                seed=1,
                n_macroparticles=1e3,
            ),
        )
    phase_observation = CavityPhaseObservation(
        each_turn_i=1,
        cavity=cavity1,
    )
    bunch_observation = BunchObservation(each_turn_i=1, beam=beam1)

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value % 10 != 0:
            return

        plt.scatter(
            beam.read_partial_dt(),
            beam.read_partial_dE(),
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
        print(f"Loaded {phase_observation.common_name}")
    except (FileNotFoundError, AssertionError):
        sim.run_simulation(
            beams=(beam1,),
            turn_i_init=0,
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
            # callback=custom_action,
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
