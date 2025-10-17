from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from .._core.base import MainLoopRelevant
from .array_recorders import DenseArrayRecorder

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from .. import WakeField
    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation
    from ..physics.cavities import SingleHarmonicCavity
    from ..physics.profiles import DynamicProfileConstNBins, StaticProfile

logger = logging.getLogger(__name__)


class Observables(MainLoopRelevant):
    def __init__(
        self,
        each_turn_i: int,
        folder: str,
        obs_per_turn: int = 1,
    ):
        """Base class to observe attributes during simulation.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        obs_per_turn
            Number of observations per turn. Default is 1,
            cannot be higher than number of cavities in turn map
        folder
            Path to the target folder used for
            saving or loading files.

        """
        super().__init__()
        self.each_turn_i = each_turn_i
        self._obs_per_turn = obs_per_turn
        if len(folder) > 0:
            assert folder.endswith("/") or folder.endswith("\\")
        self.common_name = (
            folder + "last"  # will result in filenames like last_dE.npy etc.
        )
        logger.info(f"Will save {self} to {self.common_name}_,,,")

        self._n_turns: int | None = None
        self._section_indices_to_observe: NumpyArray | None = None
        self._turn_i_init: int | None = None
        self._turns_array: NumpyArray | None = None

        self._last_turn_i_observed = (
            -1
        )  # to avoid double recordings with multiple drifts in one section
        self._last_section_i_observed = -1

    @property  # as readonly attributes
    def turns_array(self) -> NumpyArray | None:
        """Helper method to get x-axis array with turn-number and decimal places in case
        observations are performed more than once per turn.
        """
        return self._turns_array

    @abstractmethod  # pragma: no cover
    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

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
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        obs_per_turn
            Number of observations per turn. Default is 1
        """
        self._n_turns = int(n_turns)
        self._turn_i_init = int(turn_i_init)
        if obs_per_turn >= 0:
            self._obs_per_turn = obs_per_turn
        else:
            self._obs_per_turn = 1
            warnings.warn(
                f"obs_per_turn must be greater than 0, got {obs_per_turn}, value was set to 1.",
                UserWarning,
            )
        if obs_per_turn > simulation.ring.n_cavities:
            self._obs_per_turn = simulation.ring.n_cavities
            warnings.warn(
                f"obs_per_turn must be smaller than n_cavities ({simulation.ring.n_cavities}), "
                f"got {obs_per_turn}, value was set to {simulation.ring.n_cavities}.",
                UserWarning,
            )

        self._section_indices_to_observe = np.arange(
            0,
            simulation.ring.n_cavities,
            step=np.ceil(simulation.ring.n_cavities / self._obs_per_turn),
            dtype=int,
        )
        # To get the decimal point for the turns array, the distances of the individual sections in the ring
        # need to be taken into account
        section_lengths = (
            np.array(
                [
                    np.sum(simulation.ring.section_lengths[0:ind])
                    for ind in self._section_indices_to_observe
                ]
            )
            / simulation.ring.circumference
        )
        self._turns_array = np.zeros(0)
        for turn in range(turn_i_init, turn_i_init + n_turns):
            self._turns_array = np.append(
                self._turns_array, turn + section_lengths
            )

    def assert_lateinit(self):
        for parameter, value in self.__dict__.items():
            if value is None:  # uninitialized
                assert value is not None, f"`{parameter}` was not initialized."

    def get_recorders(self) -> list[tuple[str, DenseArrayRecorder]]:
        self.assert_lateinit()
        recorders = [
            (attribute, instance)
            for attribute, instance in self.__dict__.items()
            if isinstance(instance, DenseArrayRecorder)  # initialized
        ]
        return recorders

    def rename(self, common_name: str) -> None:
        """Change the common save name of all internal arrays.

        Notes
        -----
        This has no effect on files that are already saved to the disk.

        Parameters
        ----------
        common_name
            The new common name of all internal arrays.

        """
        for _attribute_name, instance in self.get_recorders():
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
        logger.info(f"Changed save target of {self} to {self.common_name}_,,,")

    def to_disk(self) -> None:
        """Save data to disk."""
        for _attribute_name, instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            logger.info(f"Saved {array_recorder.filepath_array}")
            array_recorder.to_disk()

    def from_disk(self) -> None:
        """Load data from disk."""
        for attribute_name, instance in self.get_recorders():
            array_recorder: DenseArrayRecorder = instance
            logger.info(f"Loaded {array_recorder.filepath_array}")

            self.__setattr__(
                attribute_name,
                array_recorder.from_disk(
                    filepath=array_recorder.filepath,
                ),
            )


class BunchObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        beam: BeamBaseClass,
        folder: str = "",
    ):
        """Observe the bunch coordinates during simulation execution.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        beam
            Simulation beam object
        folder
            Path to the target folder used for
            saving or loading files.
        """
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam = beam
        self._dts: DenseArrayRecorder | None = None
        self._dEs: DenseArrayRecorder | None = None
        self._flags: DenseArrayRecorder | None = None
        self._reference_time: DenseArrayRecorder | None = None
        self._reference_total_energy: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

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
            beam=self._beam,
        )
        n_entries = n_turns // self.each_turn_i + 2
        n_macroparticles = int(beam.common_array_size)
        shape = (n_entries, n_macroparticles)

        self._dts = DenseArrayRecorder(
            f"{self.common_name}_dts",
            shape,
        )
        self._dEs = DenseArrayRecorder(
            f"{self.common_name}_dEs",
            shape,
        )
        self._flags = DenseArrayRecorder(
            f"{self.common_name}_flags",
            shape,
        )

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
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

        """
        # TODO allow several bunches
        self._reference_time.write(self._beam.reference_time)
        self._reference_total_energy.write(self._beam.reference_total_energy)
        self._dts.write(self._beam._dt)
        self._dEs.write(self._beam._dE)
        self._flags.write(self._beam._flags)

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


class BunchObservation_meta_params(Observables):
    def __init__(
        self,
        each_turn_i: int,
        beam: BeamBaseClass,
        folder: str = "",
        obs_per_turn: int = 1,
    ):
        """Records mean and standard deviation of both energy and time coordinates and estimates the bunch emittance.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        obs_per_turn
            Number of observations per turn. Default is 1,
            cannot be more than number of cavities in turn map
        beam
            Simulation beam object
        folder
            Path to the target folder used for
            saving or loading files.
        """
        super().__init__(
            each_turn_i=each_turn_i, obs_per_turn=obs_per_turn, folder=folder
        )
        self._beam = beam

        self._sigma_dt: DenseArrayRecorder | None = None
        self._sigma_dE: DenseArrayRecorder | None = None
        self._mean_dt: DenseArrayRecorder | None = None
        self._mean_dE: DenseArrayRecorder | None = None
        self._emittance_stat: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when :func:`blond._core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
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
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            obs_per_turn=self._obs_per_turn,
        )

        n_entries = int(n_turns * self._obs_per_turn + 1)
        shape = n_entries

        self._mean_dt = DenseArrayRecorder(
            f"{self.common_name}_mean_dt",
            shape,
        )
        self._mean_dE = DenseArrayRecorder(
            f"{self.common_name}_mean_dE",
            shape,
        )
        self._sigma_dt = DenseArrayRecorder(
            f"{self.common_name}_sigma_dt",
            shape,
        )
        self._sigma_dE = DenseArrayRecorder(
            f"{self.common_name}_sigma_dE",
            shape,
        )
        self._emittance_stat = DenseArrayRecorder(
            f"{self.common_name}_emittance_stat",
            shape,
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

        """
        if (
            self._last_section_i_observed == simulation.section_i.value
            and self._last_turn_i_observed == simulation.turn_i.value
        ):
            return
        self._last_turn_i_observed = simulation.turn_i.value
        self._last_section_i_observed = simulation.section_i.value
        if simulation.section_i.value in self._section_indices_to_observe:
            self._sigma_dt.write(np.std(self._beam._dt))
            self._sigma_dE.write(np.std(self._beam._dE))
            self._mean_dt.write(np.mean(self._beam._dt))
            self._mean_dE.write(np.mean(self._beam._dE))
            self._emittance_stat.write(
                np.sqrt(
                    np.average(self._beam._dE**2)
                    * np.average(self._beam._dt**2)
                    - np.average(self._beam._dE * self._beam._dt) ** 2
                )
            )

    @property  # as readonly attributes
    def sigma_dt(self):
        """Standard deviation of the time coordinate."""
        return self._sigma_dt.get_valid_entries()

    @property  # as readonly attributes
    def sigma_dE(self):
        """Standard deviation of the energy coordinate."""
        return self._sigma_dE.get_valid_entries()

    @property  # as readonly attributes
    def mean_dt(self):
        """Mean of the time coordinate."""
        return self._mean_dt.get_valid_entries()

    @property  # as readonly attributes
    def mean_dE(self):
        """Mean of the time coordinate."""
        return self._mean_dE.get_valid_entries()

    @property  # as readonly attributes
    def emittance_stat(self):
        """Statistical emittance calculated with

        .. math::
            \epsilon = \sqrt{\langle \Delta t^2 \\rangle \langle \Delta E^2 \\rangle - \langle \Delta t \Delta E \\rangle^2}

        """
        return self._emittance_stat.get_valid_entries()


class MultiCavityObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        cavities: list[SingleHarmonicCavity],
        folder: str = "",
    ):
        raise NotImplementedError("To be implemented")
        # super().__init__(each_turn_i=each_turn_i, folder=folder)
        # self._cavities = cavities
        # self._phases: DenseArrayRecorder | None = None
        # self._omegas: DenseArrayRecorder | None = None
        # self._voltages: DenseArrayRecorder | None = None


class CavityPhaseObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        cavity: SingleHarmonicCavity,
        folder: str = "",
    ):
        """Observe the cavity rf parameters during simulation execution.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        cavity
            Class that implements beam-rf interactions in a synchrotron
        folder
            Path to the target folder used for
            saving or loading files.
        """
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._cavity = cavity
        self._phases: DenseArrayRecorder | None = None
        self._omegas: DenseArrayRecorder | None = None
        self._voltages: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

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
            f"{self.common_name}_phases",
            (n_entries, n_harmonics),
        )
        self._omegas = DenseArrayRecorder(
            f"{self.common_name}_omegas",
            (n_entries, n_harmonics),
        )
        self._voltages = DenseArrayRecorder(
            f"{self.common_name}_voltages",
            (n_entries, n_harmonics),
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

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
    def __init__(
        self,
        each_turn_i: int,
        profile: StaticProfile,
        obs_per_turn: int = 1,
        folder: str = "",
    ):
        """Observation of a static beam profile.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        profile
            Class for the calculation of beam profile
            that doesn't change its parameters
        obs_per_turn
            Number of observations per turn, default is 1
        folder
            Path to the target folder used for
            saving or loading files.
        """
        super().__init__(
            each_turn_i=each_turn_i,
            obs_per_turn=obs_per_turn,
            folder=folder,
        )
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

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
        n_bins = int(self._profile.n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_name}_hist_y",
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

        """
        if simulation.section_i.value in self._section_indices_to_observe:
            if (
                self._last_turn_i_observed == simulation.turn_i.value
                and self._last_section_i_observed == simulation.section_i
            ):
                return
            self._last_turn_i_observed = simulation.turn_i.value
            self._last_section_i_observed = simulation.section_i
            self._hist_y.write(
                self._profile._hist_y,
            )
        # else return without recording

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """Histogram amplitude."""
        return self._hist_y.get_valid_entries()


class StaticMultiProfileObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        profiles: list[StaticProfile],
        folder: str = "",
        obs_per_turn: int = 1,
    ):
        """Observation of multiple profiles in one observation object. The profiles need to have the same n_bins.

        Parameters
        ----------
        each_turn_i
            Value to control that the element is
            callable each n-th turn.
        profiles
            List of class for the calculation of beam profile
            that doesn't change its parameters
        obs_per_turn
            Number of observations per turn, default is 1
        folder
            Path to the target folder used for
            saving or loading files.
        """
        super().__init__(
            each_turn_i=each_turn_i, obs_per_turn=obs_per_turn, folder=folder
        )

        self._profiles = profiles
        assert all(
            prof.n_bins == self._profiles[0].n_bins for prof in self._profiles
        ), "n_bins should be equal for all given profiles"

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        turn_i_init: int,
        obs_per_turn: int = 1,
        **kwargs,
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        obs_per_turn
            Number of observations per turn, default is 1
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
            obs_per_turn=obs_per_turn,
        )
        n_entries = len(self._turns_array) * len(self._profiles)
        n_bins = self._profiles[0].n_bins
        self._hist_y = DenseArrayRecorder(
            f"{self.common_name}_hist_y",
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Updates the data in case the function has not been called on the current section and turn already.

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        if (
            self._last_turn_i_observed == simulation.turn_i.value
            and self._last_section_i_observed == simulation.section_i
        ):
            return
        self._last_turn_i_observed = simulation.turn_i.value
        self._last_section_i_observed = simulation.section_i
        for prof in self._profiles:
            if simulation.section_i.value == prof.section_index:
                self._hist_y.write(prof.hist_y)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """Histogram of given profiles."""
        return self._hist_y.get_valid_entries()


class WakeFieldObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        wakefield: WakeField,
        obs_per_turn: int = 1,
        folder: str = "",
    ):
        """Observe the calculation of wake-fields.

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
        super().__init__(
            each_turn_i=each_turn_i,
            obs_per_turn=obs_per_turn,
            folder=folder,
        )
        self._wakefield = wakefield
        self._induced_voltage: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

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
        n_bins = int(self._wakefield._profile.n_bins)
        self._induced_voltage = DenseArrayRecorder(
            f"{self.common_name}_induced_voltage",
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

        """
        if simulation.section_i.value in self._section_indices_to_observe:
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
        """Induced voltage, in [V] from given beam profile and sources.

        Returns
        -------
        induced_voltage

        """
        return self._induced_voltage.get_valid_entries()


class DynamicProfileConstNBinsObservation(Observables):
    def __init__(
        self,
        each_turn_i: int,
        profile: DynamicProfileConstNBins,
        folder: str = "",
    ):
        """Observation of a dynamic beam profile with changing width, while keeping a constant bin number.

        Parameters
        -------
         each_turn_i
            Value to control that the element is
            callable each n-th turn
        profile
            Class for the calculation of beam profile
            with a change in width, but a constant bin number
        """

        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when :func:`blond._core.simulation.simulation.Simulation.run_simulation` is called.

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
            f"{self.common_name}_hist_y",
            (n_entries, n_bins),
        )
        self._hist_x = DenseArrayRecorder(
            f"{self.common_name}_hist_x",
            (n_entries, n_bins),
        )

    def update(
        self,
        simulation: Simulation,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager
        """
        self._hist_y.write(self._profile._hist_y)
        self._hist_x.write(self._profile._hist_x)

    @property  # as readonly attributes
    def hist_y(self):
        """Histogram amplitude."""
        return self._hist_y.get_valid_entries()

    @property  # as readonly attributes
    def hist_x(self):
        """x-axis of histogram, in [s], i.e. `bin_centers`."""
        return self._hist_x.get_valid_entries()
