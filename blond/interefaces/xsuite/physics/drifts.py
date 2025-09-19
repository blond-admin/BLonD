from __future__ import annotations

from blond import Simulation
from blond._core.beam.base import BeamBaseClass
from blond.physics.drifts import DriftBaseClass


class DriftXSuite(DriftBaseClass):
    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    pass
