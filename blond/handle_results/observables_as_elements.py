"""Logs energy and time."""

from __future__ import annotations

import numpy as np

from blond._core.base import BeamObservationElement
from blond._core.beam.base import BeamBaseClass
from blond._core.simulation.simulation import Simulation


class BeamEnergyTimeLogger(BeamObservationElement):
    """logs dE and dt."""

    def __init__(
        self,
        each_turn_i: int = 1,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(section_index=section_index, name=name)
        self.each_turn_i = each_turn_i
        self.log_turns: list[int] = []
        self.log_de: list[np.ndarray] = []
        self.log_dt: list[np.ndarray] = []

    def on_init_simulation(self, simulation: Simulation) -> None:
        """On init simulation."""
        self.log_turns.clear()
        self.log_de.clear()
        self.log_dt.clear()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Run simulation."""
        pass

    def observe(self, beam: BeamBaseClass) -> None:
        """Log arrays."""
        self.log_turns.append(getattr(beam, "turn_i", len(self.log_turns)))
        self.log_de.append(np.copy(beam._dE))
        self.log_dt.append(np.copy(beam._dt))

    def track(self, beam: BeamBaseClass) -> None:
        """On track do observe the beam only."""
        self.observe(beam)

    def get_logged_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return logged arrays."""
        return (
            np.array(self.log_turns, dtype=int),
            np.stack(self.log_de) if self.log_de else np.empty((0,)),
            np.stack(self.log_dt) if self.log_dt else np.empty((0,)),
        )
