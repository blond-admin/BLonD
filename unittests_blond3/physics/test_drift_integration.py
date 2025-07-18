import numpy as np

from blond3._core.backends.backend import backend, Numpy32Bit
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
)
import logging

circumference = 26658.883

logging.basicConfig(level=logging.INFO)
ring = Ring()

cavity1 = SingleHarmonicCavity(section_index=0)
cavity1.harmonic = 35640
cavity1.voltage = 6e6
cavity1.phi_rf = 0
cavity2 = SingleHarmonicCavity(section_index=1)
cavity2.harmonic = 35640
cavity2.voltage = 6e6
cavity2.phi_rf = 0

N_TURNS = int(1e3)
energy_cycle = MagneticCyclePerTurn(
    value_init=450e9,
    values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
    reference_particle=proton,
)

drift1 = DriftSimple(
    orbit_length=circumference / 3,
    section_index=0,
)

drift2 = DriftSimple(
    orbit_length=circumference / 3,
    section_index=1,
)

drift3 = DriftSimple(
    orbit_length=circumference / 3,
    section_index=1,
)
drift1.transition_gamma = 55.759505
drift2.transition_gamma = 55.759505
drift3.transition_gamma = 55.759505
beam1 = Beam(n_particles=1e9, particle_type=proton)


sim = Simulation.from_locals(locals())
sim.ring.assert_circumference(circumference=26658.883)
