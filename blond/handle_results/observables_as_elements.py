"""
Logs energy and time at some points around the simulation, is inserted like all other elements.

Cannot be used with from_locals.

"""

from __future__ import annotations

from blond._core.base import BeamObservationElement
from blond._core.beam.base import BeamBaseClass
from blond._core.simulation.simulation import Simulation
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.observables import ObservablesGeneralElement


class BeamObserver(BeamObservationElement, ObservablesGeneralElement):
    """Logs ΔE and Δt of the beam and stores them via DenseArrayRecorder."""

    def __init__(
        self,
        each_turn_i: int = 1,
        section_index: int = 0,
        n_turns: int = 1,
        n_macroparticles: int = 1,
        folder: str | None = None,
        name: str | None = None,
    ) -> None:
        BeamObservationElement.__init__(
            self, section_index=section_index, name=name
        )
        ObservablesGeneralElement.__init__(self, folder=folder or "")  # todo
        self.each_turn_i = each_turn_i
        self.n_turns = n_turns
        self.n_macroparticles = n_macroparticles

    def on_init_simulation(self, simulation: Simulation) -> None:
        """On init."""
        n_entries = self.n_turns // self.each_turn_i + 2

        self._dEs = DenseArrayRecorder(
            self.common_name + "_dEs", (n_entries, self.n_macroparticles)
        )
        self._dts = DenseArrayRecorder(
            self.common_name + "_dts", (n_entries, self.n_macroparticles)
        )
        self._reference_time = DenseArrayRecorder(
            self.common_name + "_reference_time", (n_entries,)
        )
        self._reference_total_energy = DenseArrayRecorder(
            self.common_name + "_reference_total_energy", (n_entries,)
        )
        self._flags = DenseArrayRecorder(
            self.common_name + "_flags", (n_entries, self.n_macroparticles)
        )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        turn_i_init: int,
        obs_per_turn: int = 1,
        **kwargs,
    ) -> None:
        """On run."""
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Record beam data without modifying it."""
        self._dEs.write(beam.read_partial_dE())
        self._dts.write(beam.read_partial_dt())
        self._reference_time.write(beam.reference_time)
        self._reference_total_energy.write(beam.reference_total_energy)
        self._flags.write(beam.read_partial_flags())

    @property
    def reference_time(self):
        """Returns reference_time."""
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """Returns reference_total_energy."""
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        """Returns dts."""
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        """Returns dEs."""
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        """Returns loss flags."""
        return self._flags.get_valid_entries()
