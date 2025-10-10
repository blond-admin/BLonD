"""Collection of implementations to calculate the synchrotron radiation.

Authors
-------
Simon Lauber
Lina Valle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond import DriftSimple, Simulation
from blond._core.base import BeamPhysicsRelevant
from blond._core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any


class SynchrotronRadiation(BeamPhysicsRelevant):
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
        self._simulation: DriftSimple | None = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

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
            Simulation beam object
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
