from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .array_recorders import DenseArrayRecorder
from .._core.base import MainLoopRelevant
from ..physics.cavities import SingleHarmonicCavity
from ..physics.profiles import ProfileBaseClass, DynamicProfileConstNBins, StaticProfile

if TYPE_CHECKING:
    from typing import Optional as LateInit

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
        self._hash = simulation.get_hash()

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
        n_entries = n_turns // self.each_turn_i + 1

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

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 1
        n_particles = simulation.beams[0].common_array_size
        shape = (n_entries, n_particles)
        self._dts = DenseArrayRecorder(f"{simulation.get_hash}_dts", shape)
        self._dEs = DenseArrayRecorder(f"{simulation.get_hash}_dEs", shape)
        self._flags = DenseArrayRecorder(f"{simulation.get_hash}_flags", shape)

    def update(self, simulation: Simulation):
        self._dts.write(simulation.beams[0]._dt)
        self._dEs.write(simulation.beams[0]._dE)
        self._flags.write(simulation.beams[0]._flags)

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
        key = self._hash
        np.save(f"BunchObservation_{key}_dEs.npy", self._dEs)
        np.save(f"BunchObservation_{key}_dts.npy", self._dts)
        np.save(f"BunchObservation_{key}_flags.npy", self._flags)

    def from_disk(self) -> None:
        key = self._hash
        self._dEs = np.load(f"BunchObservation_{key}_dEs.npy")
        self._dts = np.load(f"BunchObservation_{key}_dts.npy")
        self._flags = np.load(f"BunchObservation_{key}_flags.npy")


class CavityPhaseObservation(Observables):
    def __init__(self, each_turn_i: int, cavity: SingleHarmonicCavity):
        super().__init__(each_turn_i=each_turn_i)
        self._cavity = cavity
        self._phases: LateInit[DenseArrayRecorder] = None
        self._omegas: LateInit[DenseArrayRecorder] = None
        self._voltages: LateInit[DenseArrayRecorder] = None

    def on_run_simulation(self, simulation: Simulation, n_turns: int, turn_i_init: int):
        super().on_run_simulation(
            simulation=simulation, n_turns=n_turns, turn_i_init=turn_i_init
        )
        n_entries = n_turns // self.each_turn_i + 1
        n_harmonics = self._cavity.rf_program.phi_rf.shape[0]
        self._phases = DenseArrayRecorder(
            f"{simulation.get_hash}_phases",
            (n_entries, n_harmonics),
        )
        self._omegas = DenseArrayRecorder(
            f"{simulation.get_hash}_phases",
            (n_entries, n_harmonics),
        )
        self._voltages = DenseArrayRecorder(
            f"{simulation.get_hash}_phases",
            (n_entries, n_harmonics),
        )

    def update(self, simulation: Simulation):
        self._phases.write(
            self._cavity._rf_program.phi_rf[:, simulation.turn_i.value],
        )
        self._omegas.write(
            self._cavity._rf_program.omega_rf[:, simulation.turn_i.value],
        )
        self._voltages.write(
            self._cavity._rf_program.voltage[:, simulation.turn_i.value],
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
