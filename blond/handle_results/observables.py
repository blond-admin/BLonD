from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .._core.base import MainLoopRelevant
from .array_recorders import DenseArrayRecorder

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict
    from typing import Optional as LateInit

    from .. import WakeField
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..physics.cavities import SingleHarmonicCavity
    from ..physics.profiles import StaticProfile


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
        self.common_name = (
            "last"  # will result in filenames like last_dE.npy etc.
        )

        self._n_turns: LateInit[int] = None
        self._turn_i_init: LateInit[int] = None
        self._turns_array: LateInit[NumpyArray] = None
        self._hash: LateInit[str] = None

    @property  # as readonly attributes
    def turns_array(self) -> NumpyArray | None:
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
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: Dict[str, Any],
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
        self._n_turns = int(n_turns)
        self._turn_i_init = int(turn_i_init)
        self._turns_array = np.arange(turn_i_init, turn_i_init + n_turns)
        # should be called by child class via super()

    def get_recorders(self) -> List[DenseArrayRecorder]:
        return [
            instance
            for _, instance in self.__dict__.items()
            if isinstance(instance, DenseArrayRecorder)
        ]

    def rename(self, common_name: str) -> None:
        """
        Change the common save name of all internal arrays

        Notes
        -----
        This has no effect on files that are already saved to the disk.

        Parameters
        ----------
        common_name
            The new common name of all internal arrays.

        """
        for instance in self.get_recorders():
            if self.common_name not in instance.filepath:
                raise NameError(
                    f"'{instance.filepath} does not include"
                    f" {self.common_name}' anymore. This might be caused"
                    f" by a manual override of the filename."
                )
            instance.filepath = instance.filepath.replace(
                self.common_name,
                common_name,
            )
        self.common_name = common_name

    def to_disk(self) -> None:
        """
        Save data to disk
        """
        for instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            print(f"Saved {array_recorder.filepath_array}")
            array_recorder.to_disk()

    def from_disk(self) -> None:
        """
        Load data from disk
        """
        for instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            print(f"Loaded {array_recorder.filepath_array}")
            array_recorder.from_disk(
                filepath=array_recorder.filepath,
            )


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
        **kwargs: Dict[str, Any],
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
        n_particles = int(beam.common_array_size)
        shape = (n_entries, n_particles)

        self._dts = DenseArrayRecorder(
            f"{self.common_name}_dts",
            shape,
        )  # TODO
        self._dEs = DenseArrayRecorder(
            f"{self.common_name}_dEs",
            shape,
        )  # TODO
        self._flags = DenseArrayRecorder(
            f"{self.common_name}_flags",
            shape,
        )  # TODO

        self._reference_time = DenseArrayRecorder(
            f"{self.common_name}_reference_time",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_name}_reference_total_energy",
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
        **kwargs: Dict[str, Any],
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
        n_harmonics = int(self._cavity.n_rf)
        self._phases = DenseArrayRecorder(
            f"{self.common_name}_phases",  # TODO
            (n_entries, n_harmonics),
        )
        self._omegas = DenseArrayRecorder(
            f"{self.common_name}_omegas",  # TODO
            (n_entries, n_harmonics),
        )
        self._voltages = DenseArrayRecorder(
            f"{self.common_name}_voltages",  # TODO
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
    def phases(self) -> NumpyArray:
        return self._phases.get_valid_entries()

    @property  # as readonly attributes
    def omegas(self) -> NumpyArray:
        return self._omegas.get_valid_entries()

    @property  # as readonly attributes
    def voltages(self) -> NumpyArray:
        return self._voltages.get_valid_entries()


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
        **kwargs: Dict[str, Any],
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
        n_bins = int(self._profile.n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_name}_hist_y",  # TODO
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
        **kwargs: Dict[str, Any],
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
        n_bins = int(self._wakefield._profile.n_bins)
        self._induced_voltage = DenseArrayRecorder(
            f"{self.common_name}_induced_voltage",  # TODO
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
            self._induced_voltage.write(
                np.zeros(self._wakefield._profile.n_bins)
            )

    @property  # as readonly attributes
    def induced_voltage(self):
        """
        Induced voltage, in [V] from given beam profile and sources

        Returns
        -------
        induced_voltage

        """
        return self._induced_voltage.get_valid_entries()
