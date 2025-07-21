# pragma: no cover

from os import PathLike

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
)
from blond3._core.beam.base import BeamBaseClass
from blond3.beam_preparation.base import BeamPreparationRoutine
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import MutliTurnResonatorSolver

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type


class LeonardsCounterrrotBeam(BeamPreparationRoutine):
    def __init__(
        self,
        filename_dt: PathLike | str,
        filename_dE: PathLike | str,
        filename_dt_cr: PathLike | str,
        filename_dE_cr: PathLike | str,
    ):
        self.dt = np.loadtxt(filename_dt)
        self.dE = np.loadtxt(filename_dE)

        self.dt_cr = np.loadtxt(filename_dt_cr)
        self.dE_cr = np.loadtxt(filename_dE_cr)

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        beam.setup_beam(
            dt=self.dt,
            dE=self.dE,
        )
        beam.setup_beam(
            dt=self.dt_cr,
            dE=self.dE_cr,
        )


ring = Ring(circumference=26_658.883)
energy_cycle = MagneticCyclePerTurn(np.linspace(450e9, 460.005e9, 2000))


n_cavities = 7
one_turn_model = []
for cavity_i in range(n_cavities):
    one_turn_model.extend(
        [
            SingleHarmonicCavity(
                rf_program=RfStationParams(
                    voltage=6e6,
                    phi_rf=0,
                    harmonic=35640,
                ),
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
    particle_type=proton,
    is_counter_rotating=False,
)
beam2 = Beam(
    n_particles=1e9,
    particle_type=proton,
    is_counter_rotating=True,
)
sim = Simulation(ring=ring, beams=(beam1, beam2), magnetic_cycle=energy_cycle)
sim.prepare_beam(
    preparation_routine=LeonardsCounterrrotBeam(
        "coordinates1.npy", "coordinates2.npy", "coordinates3.npy", "coordinates4.npy"
    )
)

sim.run_simulation(turn_i_init=0, n_turns=100)
