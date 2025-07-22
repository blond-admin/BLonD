# pragma: no cover
import numpy as np
from matplotlib import pyplot as plt

from blond3._core.backends.backend import backend, Numpy32Bit
from blond3.beam_preparation.empiric_matcher import EmpiricMatcher
from blond3.cycles.magnetic_cycle import MagneticCyclePerTurn

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
    CavityPhaseObservation,
    BunchObservation,
)
import logging

logging.basicConfig(level=logging.INFO)
ring = Ring()

cavity1 = SingleHarmonicCavity()
cavity1.harmonic = 35640
cavity1.voltage = 6e6
cavity1.phi_rf = 0

N_TURNS = int(1e3)

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
    n_particles=1e9,
    particle_type=proton,
)


sim = Simulation.from_locals(locals())
sim.print_one_turn_execution_order()


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

sim.prepare_beam(
    beam=beam1,
    preparation_routine=EmpiricMatcher(
        grid_base_dt=np.linspace(0, 2.5e-9, 100),
        grid_base_dE=np.linspace(-(777538700.0 * 2), 777538700.0 * 2, 100),
        n_macroparticles=1e6,
        seed=0,
        maxiter_intensity_effects=0,
    ),
)

phase_observation = CavityPhaseObservation(
    each_turn_i=1,
    cavity=cavity1,
)
bunch_observation = BunchObservation(each_turn_i=1)


# bunch_observation = BunchObservation(each_turn_i=10, batch_size=) # todo
# batches
def custom_action(simulation: Simulation):
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
    sim.load_results(
        turn_i_init=0,
        n_turns=N_TURNS,
        observe=[phase_observation],
    )
except FileNotFoundError as exc:
    sim.run_simulation(
        beams=(beam1,),
        turn_i_init=0,
        n_turns=N_TURNS,
        observe=[phase_observation, bunch_observation],
        # callback=custom_action,
    )
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
