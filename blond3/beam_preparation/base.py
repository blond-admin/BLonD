from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .._core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from .._core.simulation.simulation import Simulation


class BeamPreparationRoutine(ABC):
    """Base class to write beam preparation routines"""

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        beam.reference_total_energy = simulation.magnetic_cycle.get_total_energy_init(
            turn_i_init=simulation.turn_i.value,
            t_init=beam.reference_time,  # FIXME
            particle_type=beam.particle_type,
        )
        beam.reference_time = 0  # FIXME
        print(beam.reference_gamma)


class MatchingRoutine(BeamPreparationRoutine, ABC):
    pass
