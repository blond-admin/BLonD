# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


from copy import deepcopy

import numpy as np

from blond import Simulation
from blond.beam_preparation.base import MatchingRoutine
from blond.core.beam.base import BeamBaseClass


class BruteForceMatcher(MatchingRoutine):
    """
    Brute-force beam matching routine.

    This matcher initializes the beam using uniformly spaced macroparticles
    within specified time and energy limits and iteratively runs a simulation
    to obtain a matched beam distribution.

    Parameters
    ----------
    time_limit : tuple[float, float]
        Lower and upper limits for the time coordinate (dt).
    energy_limit : tuple[float, float]
        Lower and upper limits for the energy deviation (dE).
    n_macroparticles : int
        Number of macroparticles used to initialize the beam.
    n_iter : int
        Number of simulation iterations to perform.
    """

    def __init__(
        self,
        time_limit: tuple[float, float],
        energy_limit: tuple[float, float],
        n_macroparticles: int,
        n_iter: int,
    ) -> None:
        super().__init__()
        self.time_limit = time_limit
        self.energy_limit = energy_limit
        self.n_macroparticles = n_macroparticles
        self.n_iter = n_iter

    def prepare_beam(self, simulation: Simulation, beam: BeamBaseClass):
        """
        Prepare and match the beam using a brute-force approach.

        The beam is initialized with uniformly distributed macroparticles
        in time and energy. A copy of the simulation is then run multiple
        times to iteratively evolve the beam towards a matched state.

        Parameters
        ----------
        simulation : Simulation
            Simulation object used to track the beam.
        beam : BeamBaseClass
            Beam instance to be initialized and matched.
        """

        beam.setup_beam(
            simulation=simulation,
            dt=np.linspace(
                self.time_limit[0], self.time_limit[1], self.n_macroparticles
            ),
            dE=np.linspace(
                self.energy_limit[0],
                self.energy_limit[1],
                self.n_macroparticles,
            ),
        )

        sim_copy = deepcopy(simulation)
        sim_copy.run_simulation(beams=[beam])
        for i in range(self.n_iter - 1):
            sim_copy = deepcopy(simulation)
            sim_copy.run_simulation(beams=[beam])
