"""Base classes to define :class:`~blond.blond.beam_preparation.base.BeamPreparationRoutine` and :class:`~blond.blond.beam_preparation.base.MatchingRoutine`.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .._core.base import Schedulable
from .._core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from .._core.simulation.simulation import Simulation


class BeamPreparationRoutine(ABC):
    """Base class to write beam preparation routines.

    Notes
    -----
    These tier of routines is allowed to produce mismatched beam,
    whereas `MatchingRoutine` shall always provide for matched distributions.
    """

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
        """
        beam.reference_total_energy = (
            simulation.magnetic_cycle.get_total_energy_init(
                turn_i_init=simulation.turn_i.value,
                t_init=beam.reference_time,  # FIXME
                particle_type=beam.particle_type,
            )
        )
        beam.reference_time = 0  # FIXME
        # assign beams?

        schedulables = simulation.ring.elements.get_elements(Schedulable)
        for s in schedulables:
            s.apply_schedules(
                turn_i=simulation.turn_i.value,
                reference_time=beam.reference_time,
            )

        schedulables = simulation.ring.elements.get_elements(Schedulable)
        for s in schedulables:
            s.apply_schedules(
                turn_i=simulation.turn_i.value,
                reference_time=beam.reference_time,
            )


class MatchingRoutine(BeamPreparationRoutine, ABC):
    """Base class to define matching routines."""

    pass
