# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base classes to define :class:`~blond.blond.beam_preparation.base.BeamPreparationRoutine` and :class:`~blond.blond.beam_preparation.base.MatchingRoutine`.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from blond.core.base import Schedulable
from blond.core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.simulation.simulation import Simulation


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
            `Simulation` context manager
        beam : BeamBaseClass
            The `Beam` object which state will be updated by this element.
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


class MatchingRoutine(BeamPreparationRoutine, ABC):
    """Base class to define matching routines."""

    pass
