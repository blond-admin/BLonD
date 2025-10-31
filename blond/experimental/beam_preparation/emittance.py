"""Implementation to match beam coordinates to an emittance. """
from __future__ import annotations

from typing import TYPE_CHECKING

from blond._core.beam.base import BeamBaseClass
from blond._core.simulation.simulation import Simulation
from blond.beam_preparation.base import MatchingRoutine

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
            Simulation context manager
        beam
            Simulation `Beam` object
        """
        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
