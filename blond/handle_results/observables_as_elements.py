import numpy as np

from .. import Simulation
from .._core.base import UserDefinedElement
from .._core.beam.base import BeamBaseClass


class BeamLoggerElement(UserDefinedElement):
    """
    Element that tracks the state of `beam.dt` and `beam.dE` at a specific location in the ring.

    Parameters:
        beam (BeamBaseClass): The beam object to be monitored.
        n_turns (int): Number of turns in simulation

    Example:
        >>> beam_monitor = BeamLoggerElement(beam=BeamBaseClass, n_turns=N_TURNS)
        >>> # Insert the element into the one-turn map
        >>> # Run simulation
        >>> data = beam_monitor.get_logged_data()  # Returns beam.dE and beam.dt for all turns
    """

    def __init__(
        self,
        beam: BeamBaseClass,
        n_turns: int,
        section_index: int = 0,
        name: str = None,
    ):
        super().__init__(section_index=section_index, name=name)
        self._beam = beam
        self._de_log = []
        self._dt_log = []
        self._active_index = 0
        self._n_turns = n_turns

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        super().__init__(simulation)

    def track(self, beam: BeamBaseClass) -> None:
        # Log a copy of the beam state at this turn

        if self._active_index >= self._n_turns:
            return  #
        if self._active_index == 0:
            self._de_log = np.empty(
                (self._n_turns, self._beam.common_array_size)
            )
            self._dt_log = np.empty(
                (self._n_turns, self._beam.common_array_size)
            )

        self._de_log[self._active_index] = np.copy(beam._dE)
        self._dt_log[self._active_index] = np.copy(beam._dt)
        self._active_index += 1

    def get_logged_data(self) -> dict[str, list[np.ndarray]]:
        """Return the full turn-by-turn beam data."""
        return {
            "de": self._de_log,
            "dt": self._dt_log,
        }

    def get_turn_data(self, turn_i: int) -> dict[str, np.ndarray]:
        """Get the beam state at a specific turn."""
        return {
            "de": self._de_log[turn_i],
            "dt": self._dt_log[turn_i],
        }
