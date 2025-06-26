import numpy as np
from matplotlib import pyplot as plt

from blond3._core.backends.backend import backend, Numpy32Bit
from blond3.cycles.energy_cycle import EnergyCyclePerTurn

backend.change_backend(Numpy32Bit)
backend.set_specials("numba")

from blond3 import (
    Beam,
    proton,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    BiGaussian,
    RfStationParams,
    CavityPhaseObservation,
)
import logging

logging.basicConfig(level=logging.INFO)
ring = Ring(circumference=26658.883)

cavity1 = SingleHarmonicCavity(
    rf_program=RfStationParams(harmonic=35640, voltage=6e6, phi_rf=0)
)

N_TURNS = int(1e3)
energy_cycle = EnergyCyclePerTurn(values_per_turn=np.linspace(450e9, 450e9, N_TURNS))

drift1 = DriftSimple(
    transition_gamma=55.759505,
    share_of_circumference=1.0,
)
beam1 = Beam(n_particles=1e9, particle_type=proton)


sim = Simulation.from_locals(locals())
sim.print_one_turn_execution_order()


sim.on_prepare_beam(
    preparation_routine=BiGaussian(
        sigma_dt=0.4e-9 / 4,
        sigma_dE=1e9 / 4,
        reinsertion=False,
        seed=1,
        n_macroparticles=1e3,
    )
)

# sim.beams[0].plot_hist2d()
# plt.show()
phase_observation = CavityPhaseObservation(each_turn_i=1, cavity=cavity1)


# bunch_observation = BunchObservation(each_turn_i=10, batch_size=) # todo
# batches
def my_callback(simulation: Simulation):
    if simulation.turn_i.value % 10 != 0:
        return

    plt.scatter(
        simulation.beams[0].read_partial_dt(), simulation.beams[0].read_partial_dE()
    )
    plt.draw()
    plt.pause(0.1)
    plt.clf()


# sim.profiling(turn_i_init=0, profile_start_turn_i=10, n_turns=10000)
# sys.exit(0)
try:
    sim.load_results(turn_i_init=0, n_turns=N_TURNS, observe=[phase_observation])
except FileNotFoundError as exc:
    sim.run_simulation(
        turn_i_init=0,
        n_turns=N_TURNS,
        observe=[phase_observation],
        # callback=my_callback,
    )
plt.plot(phase_observation.phases)
plt.show()
