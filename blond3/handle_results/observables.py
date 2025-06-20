from __future__ import annotations

from abc import abstractmethod
from typing import Optional as LateInit

import numpy as np
from numpy.typing import NDArray as NumpyArray


from .array_recorders import DenseArrayRecorder
from ..core.base import MainLoopRelevant
from ..core.simulation.simulation import Simulation
from ..physics.cavities import SingleHarmonicCavity
from ..physics.profiles import ProfileBaseClass, DynamicProfileConstNBins, StaticProfile


class Observables(MainLoopRelevant):
    def __init__(self, each_turn_i: int):
        super().__init__()
        self.each_turn_i = each_turn_i

        self._n_turns: LateInit[int] = None
        self._turn_i_init: LateInit[int] = None
        self._turns_array: LateInit[NumpyArray] = None

    @property
    def turns_array(self):
        return self._turns_array

    @abstractmethod
    def update(self, simulation: Simulation) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self, simulation: Simulation, n_turns: int, turn_i_init: int
    ) -> None:
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

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i

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

    @property
    def hist_ys(self):
        return self._hist_ys.get_valid_entries()


class BunchObservation(Observables):
    def __init__(self, each_turn_i: int):
        super().__init__(each_turn_i=each_turn_i)
        self._dts: LateInit[DenseArrayRecorder] = None
        self._dEs: LateInit[DenseArrayRecorder] = None
        self._flags: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i
        n_particles = simulation.ring.beams[0].common_array_size
        shape = (n_particles, n_entries)
        self._dts = DenseArrayRecorder(f"{simulation.get_hash}_dts", shape)
        self._dEs = DenseArrayRecorder(f"{simulation.get_hash}_dEs", shape)
        self._flags = DenseArrayRecorder(f"{simulation.get_hash}_flags", shape)

    def update(self, simulation: Simulation):
        self._dts.write(simulation.ring.beams[0]._dt)
        self._dEs.write(simulation.ring.beams[0]._dE)
        self._flags.write(simulation.ring.beams[0]._flags)

    @property
    def dts(self):
        return self._dts.get_valid_entries()

    @property
    def dEs(self):
        return self._dEs.get_valid_entries()

    @property
    def flags(self):
        return self._flags.get_valid_entries()


class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: SingleHarmonicCavity):
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i
        self._phases = DenseArrayRecorder(f"{simulation.get_hash}_phases", n_entries)

    def update(self, simulation: Simulation):
        self._phases.write(self._cavity._rf_program.get_phase(simulation.turn_i.value))

    @property
    def phases(self):
        return self._phases.get_valid_entries()
