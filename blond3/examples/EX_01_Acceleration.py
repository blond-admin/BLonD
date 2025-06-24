import numpy as np
from matplotlib import pyplot as plt

from blond3 import (
    Beam,
    proton,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    BiGaussian,
    EnergyCycle,
    RfStationParams,
    CavityPhaseObservation,
    BunchObservation,
)

ring = Ring(circumference=26658.883)

cavity1 = SingleHarmonicCavity(
    rf_program=RfStationParams(harmonic=35640, voltage=6e6, phi_rf=0)
)

N_TURNS = int(1e6)
energy_cycle = EnergyCycle(synchronous_data=np.linspace(450e9, 460.005e9, N_TURNS))

drift1 = DriftSimple(
    transition_gamma=55.759505,
    share_of_circumference=1.0,
)
beam1 = Beam(n_particles=1e9, n_macroparticles=1e5, particle_type=proton)


sim = Simulation.from_locals(locals())
sim.on_prepare_beam(
    preparation_routine=BiGaussian(sigma_dt=0.4e-9 / 4, reinsertion=False, seed=1)
)
# sim.beams[0].plot_hist2d()
# plt.show()
phase_observation = CavityPhaseObservation(each_turn_i=1, cavity=cavity1)
#bunch_observation = BunchObservation(each_turn_i=10, batch_size=) # todo
# batches

try:
    sim.load_results(
        turn_i_init=0, n_turns=N_TURNS, observe=[phase_observation]
    )
except FileNotFoundError as exc:
    sim.run_simulation(
        turn_i_init=0, n_turns=N_TURNS, observe=[phase_observation]
    )
plt.plot(phase_observation.phases)
plt.show()
