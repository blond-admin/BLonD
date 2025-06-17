from matplotlib import pyplot as plt

from classes import *

ring = Ring(circumference=26658.883)

cavity1 = Cavity(
    harmonic=35640, rf_program=ConstantProgram(effective_voltage=6e6, phase=0)
)


energy_cycle = EnergyCycle(beam_energy_by_turn=np.linspace(450e9, 460.005e9, 2000))

drift1 = DriftSimple(
    transition_gamma=55.759505,
    share_of_circumference=1.0,
)
beam1 = Beam(n_particles=1e9, n_macroparticles=1001, particle_type=proton)


sim = Simulation(ring=ring, ring_attributes=locals())
sim.prepare_beam(
    preparation_routine=BiGaussian(rms_dt=0.4e-9 / 4, reinsertion=True, seed=1)
)

phase_observation = CavityPhaseObservation(each_turn_i=1, cavity=cavity1)
bunch_observation = BunchObservation(each_turn_i=10)

try:
    sim.load_results(
        turn_i_init=0, n_turns=100, observe=[phase_observation, bunch_observation]
    )
except FileNotFoundError as exc:
    sim.run_simulation(
        turn_i_init=0, n_turns=100, observe=[phase_observation, bunch_observation]
    )
plt.plot(phase_observation.phases)
plt.show()
