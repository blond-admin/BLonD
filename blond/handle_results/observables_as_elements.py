"""
Logs energy and time at some points around the simulation, is inserted like all other elements.

Cannot be used with from_locals.

"""

from __future__ import annotations

from typing import Any

from blond.core.base import BeamObservationElement
from blond.core.beam.base import BeamBaseClass
from blond.core.simulation.simulation import Simulation
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.handle_results.observables import ObservablesBaseClass


class BeamObservationInRingElement(
    BeamObservationElement, ObservablesBaseClass
):
    """Observation element placed in the ring, records beam data mid-turn.

    This element should be placed at a specific location in your pipeline. It
    cannot be used with .from_locals().

    Parameters
    ----------
    each_turn_i : int, optional
        Interval of turns at which to record data (e.g., `each_turn_i=1`
        records every turn). Defaults to 1.
    section_index : int, optional
        Index of the pipeline section where this observation element is placed.
        Defaults to 0.
    n_turns : int, optional
        Number of turns to record. Defaults to 1.
    folder : str or None, optional
        Directory path where observation data will be stored. If ``None``,
        data is kept in memory. Defaults to ``None``.
    name : str or None, optional
        Optional name for this observation element. Defaults to ``None``.
    """

    def __init__(
        self,
        each_turn_i: int = 1,
        section_index: int = 0,
        n_turns: int = 1,
        folder: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(section_index=section_index, name=name, folder=folder)
        self.each_turn_i = each_turn_i
        self.n_turns = n_turns

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        turn_i_init: int,
        obs_per_turn: int = 1,
        **kwargs: dict[
            str,
            Any,
        ],
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
        obs_per_turn
            Number of observations per turn
        """
        n_entries = n_turns // self.each_turn_i + 2

        self._dEs = DenseArrayRecorder(
            self.common_filepath + "_dEs", (n_entries, beam.common_array_size)
        )
        self._dts = DenseArrayRecorder(
            self.common_filepath + "_dts", (n_entries, beam.common_array_size)
        )
        self._reference_time = DenseArrayRecorder(
            self.common_filepath + "_reference_time", (n_entries,)
        )
        self._reference_total_energy = DenseArrayRecorder(
            self.common_filepath + "_reference_total_energy", (n_entries,)
        )
        self._flags = DenseArrayRecorder(
            self.common_filepath + "_flags",
            (n_entries, beam.common_array_size),
        )

    def track(self, beam: BeamBaseClass) -> None:
        """Record beam data without modifying it."""
        self._dEs.write(beam.read_partial_dE())
        self._dts.write(beam.read_partial_dt())
        self._reference_time.write(beam.reference_time)
        self._reference_total_energy.write(beam.reference_total_energy)
        self._flags.write(beam.read_partial_flags())

    @property  # as readonly attributes
    def reference_time(self):
        """Returns reference_time [s]."""
        return self._reference_time.get_valid_entries()

    @property
    def reference_total_energy(self):
        """Returns Total beam energy [eV].."""
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        """Returns dt coordinates of the beam [s]."""
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        """Returns dEs coordinates of the beam [eV]."""
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        """Returns flags-arrays."""
        return self._flags.get_valid_entries()
