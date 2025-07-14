from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .array_recorders import DenseArrayRecorder
from .._core.base import MainLoopRelevant

if TYPE_CHECKING:
    from typing import Optional as LateInit
    from .. import WakeField
    from ..physics.cavities import SingleHarmonicCavity
    from ..physics.profiles import (
        ProfileBaseClass,
        DynamicProfileConstNBins,
        StaticProfile,
    )

    from .._core.simulation.simulation import Simulation


class Observables(MainLoopRelevant):
    def __init__(self, each_turn_i: int):
        super().__init__()
        self.each_turn_i = each_turn_i

        self._n_turns: LateInit[int] = None
        self._turn_i_init: LateInit[int] = None
        self._turns_array: LateInit[NumpyArray] = None
        self._hash: LateInit[str] = None

    @property  # as readonly attributes
    def turns_array(self):
        return self._turns_array

    @abstractmethod
    def update(self, simulation: Simulation) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        self._hash = simulation.get_hash()

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self._n_turns = n_turns
        self._turn_i_init = turn_i_init
        self._turns_array = np.arange(turn_i_init, turn_i_init + n_turns)

    @abstractmethod
    def to_disk(self) -> None:
        pass

    @abstractmethod
    def from_disk(self) -> None:
        pass


class ProfileObservation(Observables):
    def __init__(self, each_turn_i: int, profile: LateInit[ProfileBaseClass] = None):
        super().__init__(each_turn_i=each_turn_i)
        self._profile = profile
        self._hist_ys: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 2

        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(ProfileBaseClass)
            if len(profiles) == 0:
                raise Exception("Please define a profile for your simulation!")
            elif len(profiles) > 1:
                raise Exception(
                    f"There are {len(profiles)} that can be observed."
                    f" Select one profile when initializing `ProfileObservation`."
                )
            profile = profiles[0]
            assert isinstance(profile, DynamicProfileConstNBins) or isinstance(
                profile, StaticProfile
            ), f"Only `DynamicProfileConstNBins` or `StaticProfile` allowed"
            self._profile = profile
        n_bins = len(self._profile._hist_y)

        self._hist_ys = DenseArrayRecorder(
            f"{simulation.get_hash}_hist_ys", (n_bins, n_entries)
        )

    def update(self, simulation: Simulation):
        self._hist_ys.write(self._profile._hist_y)

    @property  # as readonly attributes
    def hist_ys(self):
        return self._hist_ys.get_valid_entries()


class BunchObservation(Observables):
    def __init__(self, each_turn_i: int):
        super().__init__(each_turn_i=each_turn_i)
        self._dts: LateInit[DenseArrayRecorder] = None
        self._dEs: LateInit[DenseArrayRecorder] = None
        self._flags: LateInit[DenseArrayRecorder] = None
        self._reference_time: LateInit[DenseArrayRecorder] = None
        self._reference_total_energy: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_particles = simulation.beams[0].common_array_size
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

    def update(self, simulation: Simulation):
        self._reference_time.write(simulation.beams[0].reference_time)
        self._reference_total_energy.write(simulation.beams[0].reference_total_energy)
        self._dts.write(simulation.beams[0]._dt)
        self._dEs.write(simulation.beams[0]._dE)
        self._flags.write(simulation.beams[0]._flags)

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


class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: SingleHarmonicCavity):
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[DenseArrayRecorder] = None
        self._omegas: LateInit[DenseArrayRecorder] = None
        self._voltages: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
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

    def update(self, simulation: Simulation):
        self._phases.write(
            self._cavity.phi_rf,
        )
        self._omegas.write(
            self._cavity._omegas,
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
        self._phases.to_disk()
        self._omegas.to_disk()
        self._voltages.to_disk()

    def from_disk(self) -> None:
        self._phases = DenseArrayRecorder.from_disk(self._phases.filepath)
        self._omegas = DenseArrayRecorder.from_disk(self._omegas.filepath)
        self._voltages = DenseArrayRecorder.from_disk(self._voltages.filepath)


class StaticProfileObservation(Observables):
    def __init__(self, each_turn_i: int, profile: StaticProfile):
        super().__init__(each_turn_i=each_turn_i)
        self._profile = profile
        self._hist_y: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_bins = self._profile.n_bins
        self._hist_y = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_bins),
        )

    def update(self, simulation: Simulation):
        self._hist_y.write(
            self._profile._hist_y,
        )

    @property  # as readonly attributes
    def hist_y(self):
        return self._hist_y.get_valid_entries()

    def to_disk(self) -> None:
        self._hist_y.to_disk()

    def from_disk(self) -> None:
        self._hist_y = DenseArrayRecorder.from_disk(self._hist_y.filepath)


class WakeFieldObservation(Observables):
    def __init__(self, each_turn_i: int, wakefield: WakeField):
        super().__init__(each_turn_i=each_turn_i)
        self._wakefield = wakefield
        self._induced_voltage: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        n_turns: int,
        turn_i_init: int,
    ):
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_bins = self._wakefield._profile.n_bins
        self._induced_voltage = DenseArrayRecorder(
            f"{'simulation.get_hash'}_phases",  # TODO
            (n_entries, n_bins),
        )

    def update(self, simulation: Simulation):
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
        return self._induced_voltage.get_valid_entries()

    def to_disk(self) -> None:
        self._induced_voltage.to_disk()

    def from_disk(self) -> None:
        self._induced_voltage = DenseArrayRecorder.from_disk(self._hist_y.filepath)
