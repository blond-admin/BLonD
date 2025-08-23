from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .array_recorders import DenseArrayRecorder
from .._core.base import MainLoopRelevant

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional as LateInit
    from .. import WakeField
    from ..physics.cavities import SingleHarmonicCavity
    from ..physics.profiles import (
        StaticProfile,
    )
    from .._core.beam.base import BeamBaseClass

    from .._core.simulation.simulation import Simulation


class Observables(MainLoopRelevant):
    def __init__(self, each_turn_i: int):
        """
        Base class to observe attributes during simulation

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.

        """
        super().__init__()
        self.each_turn_i = each_turn_i

        self._n_turns: LateInit[int] = None
        self._turn_i_init: LateInit[int] = None
        self._turns_array: LateInit[NumpyArray] = None
        self._hash: LateInit[str] = None

    @property  # as readonly attributes
    def turns_array(self):
        """
        Helper method to get x-axis array with turn-number
        """
        return self._turns_array

    @abstractmethod  # pragma: no cover
    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._hash = simulation.get_hash()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self._n_turns = n_turns
        self._turn_i_init = turn_i_init
        self._turns_array = np.arange(turn_i_init, turn_i_init + n_turns)

    @abstractmethod  # pragma: no cover
    def to_disk(self) -> None:
        """
        Save data to disk
        """
        pass

    @abstractmethod  # pragma: no cover
    def from_disk(self) -> None:
        """
        Load data from disk
        """
        pass


class BunchObservation(Observables):
    def __init__(self, each_turn_i: int):
        """
        Observe the bunch coordinates during simulation execution

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        """
        super().__init__(each_turn_i=each_turn_i)
        self._dts: LateInit[DenseArrayRecorder] = None
        self._dEs: LateInit[DenseArrayRecorder] = None
        self._flags: LateInit[DenseArrayRecorder] = None
        self._reference_time: LateInit[DenseArrayRecorder] = None
        self._reference_total_energy: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            beam=beam,
        )
        n_entries = int(n_turns // self.each_turn_i + 2)
        n_particles = beam.common_array_size
        shape = (n_entries, n_particles)

        self._dts = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dts",
            shape,
        )  # TODO
        self._dEs = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dEs",
            shape,
        )  # TODO
        self._flags = DenseArrayRecorder(
            f"{'simulation.get_hash'}_flags",
            shape,
        )  # TODO

        self._reference_time = DenseArrayRecorder(
            f"{'simulation.get_hash'}_reference_time",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{'simulation.get_hash'}_reference_total_energy",
            (n_entries,),
        )

    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        # TODO allow several bunches
        self._reference_time.write(beam.reference_time)
        self._reference_total_energy.write(beam.reference_total_energy)
        self._dts.write(beam._dt)
        self._dEs.write(beam._dE)
        self._flags.write(beam._flags)

    @property  # as readonly attributes
    def reference_time(self):
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        return self._flags.get_valid_entries()

    def to_disk(self) -> None:
        self._reference_time.to_disk()
        self._reference_total_energy.to_disk()
        self._dts.to_disk()
        self._dEs.to_disk()
        self._flags.to_disk()

    def from_disk(self) -> None:
        self._reference_time = DenseArrayRecorder.from_disk(
            self._reference_time.filepath,
        )
        self._reference_total_energy = DenseArrayRecorder.from_disk(
            self._reference_total_energy.filepath,
        )
        self._dts = DenseArrayRecorder.from_disk(
            self._dts.filepath,
        )
        self._dEs = DenseArrayRecorder.from_disk(
            self._dEs.filepath,
        )
        self._flags = DenseArrayRecorder.from_disk(
            self._flags.filepath,
        )


class CRBunchObservation(BunchObservation):
    def __init__(self, each_turn_i: int, obs_per_turn: int):
        super().__init__(each_turn_i=each_turn_i)

        self._dts_CR: LateInit[DenseArrayRecorder] = None
        self._dEs_CR: LateInit[DenseArrayRecorder] = None

        self._obs_per_turn = obs_per_turn

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        # super call is neglected here on purpose, as array sizes will be wrong otherwise
        self._n_turns = n_turns
        self._turn_i_init = turn_i_init

        # overwrite for array lengths with multiple section
        n_entries = int(n_turns * self._obs_per_turn + 1)
        n_particles = beam.common_array_size
        shape = (n_entries, n_particles)

        self._turns_array = np.linspace(turn_i_init, turn_i_init + n_turns, n_entries, endpoint=False)

        self._dts = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dts",
            shape,
        )  # TODO
        self._dEs = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dEs",
            shape,
        )
        self._dts_CR = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dts",
            shape,
        )  # TODO
        self._dEs_CR = DenseArrayRecorder(
            f"{'simulation.get_hash'}_dEs",
            shape,
        )
        # TODO
        self._flags = DenseArrayRecorder(
            f"{'simulation.get_hash'}_flags",
            shape,
        )  # TODO

        self._reference_time = DenseArrayRecorder(
            f"{'simulation.get_hash'}_reference_time",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{'simulation.get_hash'}_reference_total_energy",
            (n_entries,),
        )

    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        beam_cr: BeamBaseClass=None,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        if simulation.section_i.current_group % self._obs_per_turn == 0:
            super().update(simulation=simulation, beam=beam)
            if beam_cr is None:
                raise RuntimeError("Use Bunch Observation for single beam observations")
            self._dts_CR.write(beam_cr._dt)
            self._dEs_CR.write(beam_cr._dE)

    @property  # as readonly attributes
    def dts_CR(self):
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs_CR(self):
        return self._dEs.get_valid_entries()

    def to_disk(self) -> None:
        super().to_disk()
        self._dts_CR.to_disk()
        self._dEs_CR.to_disk()

    def from_disk(self) -> None:
        super().from_disk()
        self._dts_CR = DenseArrayRecorder.from_disk(
            self._dts_CR.filepath,
        )
        self._dEs_CR = DenseArrayRecorder.from_disk(
            self._dEs_CR.filepath,
        )

class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: SingleHarmonicCavity):
        """
        Observe the cavity rf parameters during simulation execution

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        cavity
            Class that implements beam-rf interactions in a synchrotron
        """
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[DenseArrayRecorder] = None
        self._omegas: LateInit[DenseArrayRecorder] = None
        self._voltages: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            beam=beam,
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_harmonics = self._cavity.n_rf
        self._phases = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_harmonics),
        )
        self._omegas = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_harmonics),
        )
        self._voltages = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_harmonics),
        )

    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        self._phases.write(
            None
            if self._cavity.phi_rf is None
            else (self._cavity.phi_rf + self._cavity.delta_phi_rf)
        )
        self._omegas.write(
            None
            if self._cavity._omega_rf is None
            else (self._cavity._omega_rf + self._cavity.delta_omega_rf)
        )
        self._voltages.write(
            self._cavity.voltage,
        )

    @property  # as readonly attributes
    def phases(self):
        return self._phases.get_valid_entries()

    @property  # as readonly attributes
    def omegas(self):
        return self._omegas.get_valid_entries()

    @property  # as readonly attributes
    def voltages(self):
        return self._voltages.get_valid_entries()

    def to_disk(self) -> None:
        """
        Save data to disk
        """
        self._phases.to_disk()
        self._omegas.to_disk()
        self._voltages.to_disk()

    def from_disk(self) -> None:
        """
        Load data from disk
        """
        self._phases = DenseArrayRecorder.from_disk(self._phases.filepath)
        self._omegas = DenseArrayRecorder.from_disk(self._omegas.filepath)
        self._voltages = DenseArrayRecorder.from_disk(self._voltages.filepath)


class StaticProfileObservation(Observables):
    def __init__(self, each_turn_i: int, profile: StaticProfile):
        """
        Observation of a static beam profile

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        profile
            Class for the calculation of beam profile
            that doesn't change its parameters
        """
        super().__init__(each_turn_i=each_turn_i)
        self._profile = profile
        self._hist_y: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            beam=beam,
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_bins = self._profile.n_bins
        self._hist_y = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        self._hist_y.write(
            self._profile._hist_y,
        )

    @property  # as readonly attributes
    def hist_y(self):
        """Histogram amplitude"""
        return self._hist_y.get_valid_entries()

    def to_disk(self) -> None:
        """
        Save data to disk
        """
        self._hist_y.to_disk()

    def from_disk(self) -> None:
        """
        Load data from disk
        """
        self._hist_y = DenseArrayRecorder.from_disk(self._hist_y.filepath)


class WakeFieldObservation(Observables):
    def __init__(self, each_turn_i: int, wakefield: WakeField):
        """
        Observe the calculation of wake-fields

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        wakefield
            Manager class to calculate wake-fields
        """
        super().__init__(each_turn_i=each_turn_i)
        self._wakefield = wakefield
        self._induced_voltage: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            beam=beam,
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_bins = self._wakefield._profile.n_bins
        self._induced_voltage = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Update memory with new values

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object

        """
        try:
            self._induced_voltage.write(
                self._wakefield.induced_voltage,
            )
        except AttributeError:
            self._induced_voltage.write(np.zeros(self._wakefield._profile.n_bins))

    @property  # as readonly attributes
    def induced_voltage(self):
        """
        Induced voltage, in [V] from given beam profile and sources

        Returns
        -------
        induced_voltage

        """
        return self._induced_voltage.get_valid_entries()

    def to_disk(self) -> None:
        """
        Save data to disk
        """
        self._induced_voltage.to_disk()

    def from_disk(self) -> None:
        """
        Load data from disk
        """
        self._induced_voltage = DenseArrayRecorder.from_disk(
            self._induced_voltage.filepath
        )
