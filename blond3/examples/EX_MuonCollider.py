from blond3 import (
    BiGaussian,
    Beam,
    proton,
    Ring,
    Simulation,
    EnergyCycle,
    ConstantProgram,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
)
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.sovlers import MutliTurnResonatorSolver

ring = Ring(circumference=26658.883)
energy_cycle = EnergyCycle.from_linspace(450e9, 460.005e9, 2000)

cavity1 = SingleHarmonicCavity(
    harmonic=35640, rf_program=ConstantProgram(effective_voltage=6e6, phase=0)
)

drift1 = DriftSimple(
    transition_gamma=55.759505,
    share_of_circumference=1.0,
)

n_cavities = 7
one_turn_model = []
for cavity_i in range(n_cavities):
    one_turn_model.extend(
        [
            SingleHarmonicCavity(
                harmonic=35640,
                rf_program=ConstantProgram(effective_voltage=6e6, phase=0),
                local_wakefield=WakeField(
                    sources=(Resonators(),),
                    solver=MutliTurnResonatorSolver(),
                ),
            ),
            DriftSimple(
                transition_gamma=55.759505,
                share_of_circumference=1 / n_cavities,
            ),
        ]
    )
ring.add_elements(one_turn_model, reorder=False)
####################################################################
beam1 = Beam(
    n_particles=1e9,
    n_macroparticles=1001,
    particle_type=proton,
    is_counter_rotating=False,
)
beam2 = Beam(
    n_particles=1e9,
    n_macroparticles=1001,
    particle_type=proton,
    is_counter_rotating=True,
)
sim = Simulation(ring=ring, beams=(beam1, beam2), energy_cycle=energy_cycle)
sim.prepare_beam(
    preparation_routine=BiGaussian(rms_dt=0.4e-9 / 4, reinsertion=True, seed=1)
)

sim.run_simulation(turn_i_init=0, n_turns=100)
