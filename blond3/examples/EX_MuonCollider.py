from os import PathLike

import numpy as np

from blond3 import (
    Beam,
    mu_plus,
    mu_minus,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
    MagneticCyclePerTurn,
    BunchObservation,
    StaticProfile,
)
from blond3._core.beam.base import BeamBaseClass
from blond3.beam_preparation.base import BeamPreparationRoutine
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import MultiPassResonatorSolver, AnalyticSingleTurnResonatorSolver
from scipy.constants import pi


class LoadBeamDataCR(BeamPreparationRoutine):
    def __init__(
        self,
        filename: PathLike | str,
        ):
        self.dt = np.load(filename)["dt"]
        self.dE = np.load(filename)["dE"]

        self.dt_cr = np.load(filename)["dt"]
        self.dE_cr = np.load(filename)["dE"]

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass | list[BeamBaseClass],
    ) -> None:
        beam[0].setup_beam(
            dt=self.dt,
            dE=self.dE,
        )
        beam[1].setup_beam(
            dt=self.dt_cr,
            dE=self.dE_cr,
        )

phi_s = 128 * pi / 180  # deg
inj_energy = 63e9
ejection_energy = 313.83e9
n_turns = 17
energy_gain_per_turn = (ejection_energy - inj_energy) / n_turns
total_voltage = energy_gain_per_turn / np.sin(phi_s)
n_cavities = 20
Q_factor = 0.96e6
R_over_Q = 518
alpha_p = 4.68e-4
gamma_transition = 1 / np.sqrt(alpha_p)
circumference = 5990
harmonic = 25900

ring = Ring(circumference=circumference)
magnetic_cycle = MagneticCyclePerTurn(value_init=inj_energy,
                                    values_after_turn=np.linspace(inj_energy + energy_gain_per_turn, ejection_energy,
                                                                  n_turns),
                                    in_unit="kinetic energy",
                                    reference_particle=mu_plus)
profile = StaticProfile.from_rad(
    0,
    2 * np.pi,
    2 ** 10,
    magnetic_cycle.get_t_rev_init(
        ring.circumference,
        turn_i_init=0,
        t_init=0,
        particle_type=mu_plus,
    )
    / harmonic / n_cavities,
)
one_turn_model = []
for cavity_i in range(n_cavities):
    local_res = Resonators(center_frequencies=1.3e9, quality_factors=Q_factor, shunt_impedances=R_over_Q*Q_factor)  # FM only
    one_turn_model.extend(
        [
            SingleHarmonicCavity(
                voltage=total_voltage / n_cavities,
                phi_rf=0,
                harmonic=harmonic,
                local_wakefield=WakeField(
                    sources=(local_res,),
                    solver=AnalyticSingleTurnResonatorSolver(),
                    profile=profile,
                ),
                section_index=cavity_i,
            ),
            DriftSimple(
                transition_gamma=gamma_transition,
                orbit_length=circumference / n_cavities,
                section_index=cavity_i,
            ),
        ]
    )
ring.add_elements(one_turn_model, reorder=False)
####################################################################
beam = Beam(
    n_particles=2.7e12,
    particle_type=mu_plus,
    is_counter_rotating=False,
)
beam_CR = Beam(
    n_particles=2.7e12,
    particle_type=mu_minus,
    is_counter_rotating=True,
)
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
sim.prepare_beam(
    beam=[beam, beam_CR],
    preparation_routine=LoadBeamDataCR(
        "initial_beam.npz"
    )
)

bunch_observation = BunchObservation(each_turn_i=1)
sim.run_simulation(beams=(beam, beam_CR), turn_i_init=0, n_turns=n_turns, observe=[bunch_observation])
