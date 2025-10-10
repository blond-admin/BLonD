"""Collection of implementations to handle beam losses in synchrotrons..

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant
from .._core.beam.base import BeamBaseClass
from .._core.simulation.simulation import Simulation


class LossesBaseClass(BeamPhysicsRelevant):
    """Abstract class to group/implement losses."""

    def __init__(self) -> None:
        super().__init__()


class BoxLosses(LossesBaseClass):
    """Label particles that are outside a box.

    Parameters
    ----------
    t_min
        Minimum of the bounding box, in [s].
    t_max
        Maximum of the bounding box, in [s].
    e_min
        Minimum of the bounding box, in [eV].
    e_max
        Maximum of the bounding box, in [eV].
    """

    def __init__(
        self,
        t_min: backend.float | None = None,
        t_max: backend.float | None = None,
        e_min: backend.float | None = None,
        e_max: backend.float | None = None,
    ) -> None:
        super().__init__()

        self.t_min = backend.float(t_min)
        self.t_max = backend.float(t_max)
        self.e_min = backend.float(e_min)
        self.e_max = backend.float(e_max)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        pass

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
        backend.loss_box(
            beam.write_partial_flags(),
            self.t_min,
            self.t_max,
            self.e_min,
            self.e_max,
        )


class SeparatrixLosses(LossesBaseClass):
    """Label particles that are outside of a separatrix."""

    def __init__(self) -> None:
        super().__init__()
        self._simulation: Simulation | None = None

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

        Parameters
        ----------
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
        self._simulation.get_separatrix()  # TODO
