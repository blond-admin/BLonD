# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to calculate the synchrotron radiation.

Authors
-------
Simon Lauber
Lina Valle
"""

from __future__ import (
    annotations,  # pragma: no cover # TODO remove when SR is implemented
)

from typing import (
    TYPE_CHECKING,  # pragma: no cover # TODO remove when SR is implemented
)

from blond.core.base import (
    BeamPhysicsRelevant,  # pragma: no cover # TODO remove when SR is implemented
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.drifts import DriftSimple


class SynchrotronRadiation(BeamPhysicsRelevant):  # pragma: no cover
    """Synchrotron radiation module.

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name
        User given name of the element
    """

    def __init__(self, section_index: int = 0, name: str | None = None):
        super().__init__(
            section_index=section_index,
            name=name,
        )
        raise NotImplementedError("For Lina")
        # TODO remove # pragma: no cover if implemented
        self._simulation: DriftSimple | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when :func:`blond.core.simulation.simulation.Simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        self._simulation = simulation

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass
