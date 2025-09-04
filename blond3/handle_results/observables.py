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
    def __init__(self, each_turn_i: int, obs_per_turn: int=1):
        """
        Base class to observe attributes during simulation

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        obs_per_turn
            Number of observations per turn.

        """
        super().__init__()
        self.each_turn_i = each_turn_i
        self._obs_per_turn = obs_per_turn

        self._n_turns: LateInit[int] = None
        self._index_list: LateInit[int] = None
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
            obs_per_turn: int = 1,
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
        self._obs_per_turn = obs_per_turn
        self._turns_array = np.linspace(turn_i_init, turn_i_init + n_turns, int(n_turns * self._obs_per_turn + 1), endpoint=False)
        self._index_list = np.arange(0, simulation.ring.n_cavities,
                                     step=np.ceil(simulation.ring.n_cavities / self._obs_per_turn))

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
        n_entries = n_turns // self.each_turn_i + 2
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


class BunchObservation_meta_params(Observables):
    """
    Records mean and sigma of both energy and time bunch coordinates
    """

    def __init__(self, each_turn_i: int, obs_per_turn: int):
        super().__init__(each_turn_i=each_turn_i)

        self._sigma_dt: LateInit[DenseArrayRecorder] = None
        self._sigma_dE: LateInit[DenseArrayRecorder] = None
        self._mean_dt: LateInit[DenseArrayRecorder] = None
        self._mean_dE: LateInit[DenseArrayRecorder] = None
        self._emittance_stat: LateInit[DenseArrayRecorder] = None

        assert obs_per_turn >= 0, "obs per turn needs to be >= 0"
        self._obs_per_turn = obs_per_turn

    def on_run_simulation(
            self,
            simulation: Simulation,
            beam: BeamBaseClass,
            n_turns: int,
            turn_i_init: int,
            **kwargs,
    ) -> None:
        # super call is neglected here on purpose, as array sizes will be wrong otherwise
        self._n_turns = n_turns
        self._turn_i_init = turn_i_init

        # TODO: check if the obs_per_turn is larger than the number of sections --> not possible

        n_entries = int(n_turns * self._obs_per_turn + 1)
        shape = (n_entries)

        self._turns_array = np.linspace(turn_i_init, turn_i_init + n_turns, n_entries, endpoint=False)

        assert simulation.ring.n_cavities >= self._obs_per_turn, "more obervations than observation points"
        self._index_list = np.arange(0, simulation.ring.n_cavities,
                                     step=np.ceil(simulation.ring.n_cavities / self._obs_per_turn))

        self._mean_dt = DenseArrayRecorder(
            f"{'simulation.get_hash'}_mean_dt",
            shape,
        )
        self._mean_dE = DenseArrayRecorder(
            f"{'simulation.get_hash'}_mean_dE",
            shape,
        )
        self._sigma_dt = DenseArrayRecorder(
            f"{'simulation.get_hash'}_sigma_dt",
            shape,
        )
        self._sigma_dE = DenseArrayRecorder(
            f"{'simulation.get_hash'}_sigma_dE",
            shape,
        )
        self._emittance_stat = DenseArrayRecorder(
            f"{'simulation.get_hash'}_emittance_stat",
            shape,
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
        if simulation.section_i.current_group in self._index_list:
            super().update(simulation=simulation, beam=beam)
            self._sigma_dt.write(np.std(beam._dt))
            self._sigma_dE.write(np.std(beam._dE))
            self._mean_dt.write(np.mean(beam._dt))
            self._mean_dE.write(np.mean(beam._dE))
            self._emittance_stat.write(
                np.sqrt(np.average(beam._dE ** 2) * np.average(beam._dt ** 2) - np.average(beam._dE * beam._dt)))

    @property  # as readonly attributes
    def sigma_dt(self):
        return self._sigma_dt.get_valid_entries()

    @property  # as readonly attributes
    def sigma_dE(self):
        return self._sigma_dE.get_valid_entries()

    @property  # as readonly attributes
    def mean_dt(self):
        return self._mean_dt.get_valid_entries()

    @property  # as readonly attributes
    def mean_dE(self):
        return self._mean_dE.get_valid_entries()

    @property  # as readonly attributes
    def emittance_stat(self):
        return self._emittance_stat.get_valid_entries()

    def to_disk(self) -> None:
        super().to_disk()
        self._sigma_dt.to_disk()
        self._sigma_dE.to_disk()
        self._mean_dt.to_disk()
        self._mean_dE.to_disk()
        self._emittance_stat.to_disk()

    def from_disk(self) -> None:
        super().from_disk()

        self._sigma_dt = DenseArrayRecorder.from_disk(
            self._sigma_dt.filepath,
        )
        self._sigma_dE = DenseArrayRecorder.from_disk(
            self._sigma_dE.filepath,
        )

        self._mean_dt = DenseArrayRecorder.from_disk(
            self._mean_dt.filepath,
        )

        self._mean_dE = DenseArrayRecorder.from_disk(
            self._mean_dE.filepath,
        )

        self._emittance_stat = DenseArrayRecorder.from_disk(
            self._emittance_stat.filepath,
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
    def __init__(self, each_turn_i: int, profile: StaticProfile, obs_per_turn: int=1):
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
        obs_per_turn
            Number of observations per turn
        """
        super().__init__(each_turn_i=each_turn_i, obs_per_turn=obs_per_turn)
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
            obs_per_turn=self._obs_per_turn,
            beam=beam,
        )
        n_entries = len(self._turns_array)
        n_bins = self._profile.n_bins
        self._hist_y = DenseArrayRecorder(
            f"{'simulation.get_hash'}_hist_y",  # TODO
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
        if simulation.section_i.current_group in self._index_list:
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

class StaticAllProfileObservation(Observables):
    # get from simulation elements
    def __init__(self, each_turn_i: int, profile: StaticProfile, obs_per_turn: int=1):
        pass


class WakeFieldObservation(Observables):
    def __init__(self, each_turn_i: int, wakefield: WakeField,
                 obs_per_turn: int=1):
        """
        Observe the calculation of wake-fields

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        wakefield
            Manager class to calculate wake-fields
        obs_per_turn
            Number of observations per turn
        """
        super().__init__(each_turn_i=each_turn_i, obs_per_turn=obs_per_turn)
        self._obs_per_turn = obs_per_turn
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
            obs_per_turn=self._obs_per_turn,
            beam=beam,
        )
        n_entries = len(self._turns_array)
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
        if simulation.section_i.current_group in self._index_list:
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
