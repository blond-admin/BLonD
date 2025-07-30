# pragma: no cover
import sys
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np

from blond3 import (
    Beam,
    proton,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
    MagneticCyclePerTurn,
    mu_plus,
    StaticProfile,
)
from blond3._core.beam.base import BeamBaseClass
from blond3.beam_preparation.base import BeamPreparationRoutine
from blond3.handle_results.helpers import callers_relative_path
from blond3.physics.impedances.solvers import MutliTurnResonatorSolver
from blond3.physics.impedances.sources import Resonators

if TYPE_CHECKING:
    pass


class LeonardsCounterrrotBeam(BeamPreparationRoutine):
    def __init__(
        self,
        filename_dt: PathLike | str,
        filename_dE: PathLike | str,
    ):
        self.dt = np.loadtxt(callers_relative_path(filename_dt, stacklevel=2))
        self.dE = np.loadtxt(callers_relative_path(filename_dE, stacklevel=2))

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        beam.setup_beam(
            dt=self.dt,
            dE=self.dE,
        )


ring = Ring(circumference=26_658.883)
energy_cycle = MagneticCyclePerTurn(
    reference_particle=mu_plus,
    value_init=450e9,
    values_after_turn=np.linspace(450e9, 460.005e9, 2000),
    in_unit="momentum",
)

# TODO implement MutliTurnResonatorSolver
"""local_wakefield=WakeField(
    sources=(
        Resonators(
            # TODO set some useful parameter
            shunt_impedances=np.ones(1, dtype=float),
            center_frequencies=np.ones(1, dtype=float),
            quality_factors=np.ones(1, dtype=float),
        ),
    ),
    solver=MutliTurnResonatorSolver(),
),"""
n_cavities = 7
one_turn_model = []
for cavity_i in range(n_cavities):
    cavity = SingleHarmonicCavity(
        section_index=cavity_i,
        # local_wakefield=local_wakefield, # todo
    )
    profile = StaticProfile(cut_left=0, cut_right=1, n_bins=256, section_index=cavity_i)
    cavity.voltage = 6e6
    cavity.phi_rf = 0
    cavity.harmonic = 35640
    one_turn_model.extend(
        [
            cavity,
            DriftSimple(
                transition_gamma=55.759505,
                orbit_length=ring.circumference / n_cavities,
                section_index=cavity_i,
            ),
            profile,
        ]
    )
ring.add_elements(one_turn_model, reorder=False)
sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
sim.print_one_turn_execution_order()
####################################################################
beam1 = Beam(
    n_particles=1e9,
    particle_type=proton,
    is_counter_rotating=False,
)
beam2 = Beam(
    n_particles=1e9,
    particle_type=proton,
    is_counter_rotating=True,
)
sim.prepare_beam(
    preparation_routine=LeonardsCounterrrotBeam(
        "resources/coordinates1.txt",
        "resources/coordinates2.txt",
    ),
    beam=beam1,
)
sim.prepare_beam(
    preparation_routine=LeonardsCounterrrotBeam(
        "resources/coordinates4.txt",
        "resources/coordinates3.txt",
    ),
    beam=beam2,
)
# TODO implement _run_simulation_counterrotating_beam
# sim.run_simulation(beams=(beam1, beam2), turn_i_init=0, n_turns=100)
