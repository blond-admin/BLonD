# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Implementation to match beam coordinates to an emittance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.beam_preparation.base import MatchingRoutine
from blond.core.beam.base import BeamBaseClass
from blond.core.simulation.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    pass


class EmittanceMatcher(MatchingRoutine):
    """Matches the beam coordinates to a given emittance."""

    def __init__(self, some_emittance: float, n_macroparticles: int):
        raise NotImplementedError("To be developed")  # TODO
        super().__init__()
        self.some_emittance = some_emittance

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles.

        Parameters
        ----------
        simulation
            `Simulation` context manager
        beam
            Simulation `Beam` object
        """
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
