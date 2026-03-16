# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module holding all observables for the simulation."""

from __future__ import annotations

import logging
import math
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from blond.core.base import MainLoopRelevant
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.handle_results.array_recorders import DenseArrayRecorder
from blond.physics.drifts import DriftSimple

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond import WakeField
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import (
        SingleHarmonicRFStation,
    )
    from blond.physics.profiles import DynamicProfileConstNBins, StaticProfile

logger = logging.getLogger(__name__)

# DEV NOTE
# The main reason to have so much boilerplate code
# is providing an interface that allows autocompletion
# and allow testing beforehand.


class ObservablesBaseClass(MainLoopRelevant):
    """
    Base class to define observations.

    Parameters
    ----------
    folder
        Target folder to save the data at.
        Use `rename` to change the destination.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(self, folder: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if len(folder) > 0:
            assert folder.endswith("/") or folder.endswith("\\")
        self.common_filepath = folder + "last"
        logger.info(f"Will save {self} to {self.common_filepath}_,,,")

    def get_recorders(self) -> list[tuple[str, DenseArrayRecorder]]:
        """
        Get all `DenseArrayRecorder` inside the current instance.

        Returns
        -------
        recorders
            List of ((attribute name, attribute), ...).
        """
        self.assert_lateinit()
        recorders = [
            (attribute, instance)
            for attribute, instance in self.__dict__.items()
            if isinstance(instance, DenseArrayRecorder)  # initialized
        ]
        return recorders

    def rename(self, new_common_filepath: str) -> None:
        """
        Change the common save name of all internal arrays.

        Parameters
        ----------
        new_common_filepath
            The new common name of all internal arrays.

        Notes
        -----
        This has no effect on files that are already saved to the disk.
        """
        old_common_filepath = self.common_filepath
        for _attribute_name, instance in self.get_recorders():
            if old_common_filepath not in instance.filepath:
                # it would not make sense to replace the old filepath
                raise NameError(
                    f"{instance.filepath} does not include"
                    f" {old_common_filepath} anymore. This might be caused"
                    f" by a manual override of the filename."
                )
            instance.filepath = instance.filepath.replace(
                old_common_filepath,
                new_common_filepath,
            )
        self.common_filepath = new_common_filepath
        logger.info(
            f"Changed save target of {self} to {self.common_filepath}."
        )

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

    def assert_lateinit(self):
        """Check that DenseArrays are already initialized."""
        for parameter, value in self.__dict__.items():
            if value is None:  # uninitialized
                assert value is not None, f"`{parameter}` was not initialized."


class ObservablesOncePerTurnBase(ObservablesBaseClass):
    """
    Observe attributes during simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
        **kwargs,
    ):
        super().__init__(folder=folder, **kwargs)
        self.each_turn_i = each_turn_i

        self._n_turns: int | None = None
        self._turns_array: NumpyArray | None = None

        self._last_turn_i_observed = (
            -1
        )  # to avoid double recordings with multiple drifts in one section
        self._last_section_i_observed = -1

        self._simulation: Simulation | None = None

    def _calc_n_entries(self, n_turns: int) -> int:
        """
        Calculate the number of entries considering `each_turn_i`.

        Parameters
        ----------
        n_turns
            Number of turns that the simulation is foreseen to run.

        Returns
        -------
        n_entries
            The number of observations during the simulation.
        """
        return int(math.ceil(n_turns / self.each_turn_i))

    @property  # as readonly attributes
    def turns_array(self) -> NumpyArray | None:
        """
        Helper method to get x-axis array with turn-number of shape ``(n_observations, )``.

        Helper method to get x-axis array with turn-number for which the
        observations are performed.

        Returns
        -------
        turns_array
            Array with turn numbers for observations.
        """
        return self._turns_array

    @abstractmethod  # pragma: no cover
    def _update(self) -> None:
        """Update memory with new values."""
        pass

    def update(self) -> None:
        """Update memory with new values."""
        if self._last_turn_i_observed != self._simulation.turn_i.value:
            self._update()
            self._last_turn_i_observed = self._simulation.turn_i.value
        else:
            raise RuntimeError(
                f"{self} already called update in this turn for turn {self._last_turn_i_observed}."
                f" Was this observation added twice?",
            )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        self._n_turns = int(n_turns)

        self._turns_array = np.arange(0, n_turns, self.each_turn_i, dtype=int)
        assert len(self._turns_array) == self._calc_n_entries(n_turns=n_turns)

        self._simulation = simulation


class BeamObservationOncePerTurn(ObservablesOncePerTurnBase):
    """
    Observe the bunch coordinates during simulation execution after a drift element.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> from blond import BeamObservationOncePerTurn
    >>>
    >>> sim = Simulation(...)
    >>> bunch_observation = BeamObservationOncePerTurn(each_turn_i=2)
    >>>
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(bunch_observation,),
    ... )
    >>> turn_0 = 0 # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.hist2d(
    ...         bunch_observation.dts[index, :],
    ...         bunch_observation.dEs[index, :],
    ...         bins=256,
    ...         range=[[0, 2.5e-9], [-4e8, 4e8]],
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam: BeamBaseClass | None = None
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
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        from blond.generals.distributed.helpers import mpi_is_distributed

        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        self._beam = beam
        n_entries = self._calc_n_entries(n_turns)
        n_macroparticles = int(beam._dt.local_size)
        if mpi_is_distributed():
            warnings.warn(
                "Saving beam with `BeamObservationOncePerTurn` only from "
                "MPI-rank 0.",
                UserWarning,
                stacklevel=2,
            )
        shape = (n_entries, n_macroparticles)

        self._dts = DenseArrayRecorder(
            f"{self.common_filepath}_dts",
            shape,
        )
        self._dEs = DenseArrayRecorder(
            f"{self.common_filepath}_dEs",
            shape,
        )
        self._flags = DenseArrayRecorder(
            f"{self.common_filepath}_flags",
            shape,
        )

        self._reference_time = DenseArrayRecorder(
            f"{self.common_filepath}_reference_time",
            (n_entries,),
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_filepath}_reference_total_energy",
            (n_entries,),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        # TODO allow several bunches
        self._reference_time.write(self._beam.reference.time)
        self._reference_total_energy.write(self._beam.reference.total_energy)
        self._dts.write(self._beam.read_partial_dt())
        self._dEs.write(self._beam.read_partial_dE())
        self._flags.write(self._beam.read_partial_flags())

    @property  # as readonly attributes
    def reference_time(self):
        """
        Return reference time of shape ``(n_observations, n_bins)``.

        Returns
        -------
        reference_time
            Reference time array.
        """
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """
        Return total energy of shape ``(n_observations, n_bins)``.

        Returns
        -------
        reference_total_energy
            Total energy array.
        """
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        """
        Return array of dts of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        dts
            Time coordinate array.
        """
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        """
        Return array of dEs of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        dEs
            Energy coordinate array.
        """
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        """
        Return flags of particles, eg if lost or not of shape ``(n_observations, n_macroparticles)``.

        Returns
        -------
        flags
            Particle flags array.
        """
        return self._flags.get_valid_entries()


class BeamStatisticsOncePerTurn(ObservablesOncePerTurnBase):
    """
    Observe the beam statistics during simulation execution after a drift element.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> bunch_statistics = BeamStatisticsOncePerTurn(each_turn_i=2, beam=...)
    >>>
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(bunch_statistics,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2)
    ...     plt.plot(
    ...         bunch_statistics.bunch_position()[index, :],
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._beam: BeamBaseClass | None = None
        self._bunch_position: DenseArrayRecorder | None = None
        self._energy_spread: DenseArrayRecorder | None = None
        self._bunch_length: DenseArrayRecorder | None = None
        self._reference_time: DenseArrayRecorder | None = None
        self._reference_total_energy: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond._cycles_core.beam.beam.Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        self._beam = beam
        n_entries = self._calc_n_entries(n_turns)

        self._bunch_position = DenseArrayRecorder(
            f"{self.common_filepath}_bunch_position",
            n_entries,
        )
        self._energy_spread = DenseArrayRecorder(
            f"{self.common_filepath}_energy_spread",
            n_entries,
        )
        self._bunch_length = DenseArrayRecorder(
            f"{self.common_filepath}_bunch_length",
            n_entries,
        )
        self._reference_time = DenseArrayRecorder(
            f"{self.common_filepath}_reference_time",
            n_entries,
        )
        self._reference_total_energy = DenseArrayRecorder(
            f"{self.common_filepath}_reference_total_energy",
            n_entries,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        # TODO allow several bunches

        self._bunch_position.write(np.average(self._beam.read_partial_dt()))
        self._energy_spread.write(np.std(self._beam.read_partial_dE()))
        self._bunch_length.write(np.std(self._beam.read_partial_dt()))

        self._reference_time.write(self._beam.reference.time)
        self._reference_total_energy.write(self._beam.reference.total_energy)

    @property  # as readonly attributes
    def bunch_position(self):
        """
        Return array of bunch_position of shape (n_observations,).

        Returns
        -------
        bunch_position
            Bunch position array.
        """
        return self._bunch_position.get_valid_entries()

    @property  # as readonly attributes
    def energy_spread(self):
        """
        Return array of energy spread of shape (n_observations,).

        Returns
        -------
        energy_spread
            Energy spread array.
        """
        return self._energy_spread.get_valid_entries()

    @property  # as readonly attributes
    def bunch_length(self):
        """
        Return array of bunch_length of shape (n_observations,).

        Returns
        -------
        bunch_length
            Bunch length array.
        """
        return self._bunch_length.get_valid_entries()

    @property  # as readonly attributes
    def reference_time(self):
        """
        Return reference time of shape (n_observations,).

        Returns
        -------
        reference_time
            Reference time array.
        """
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """
        Return reference total energy of shape (n_observations,).

        Returns
        -------
        reference_total_energy
            Total energy array.
        """
        return self._reference_total_energy.get_valid_entries()


class RFStationPhaseObservation(ObservablesOncePerTurnBase):
    """
    Observe the RF station parameters during the execution of the simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    rf_station
        Class that implements beam-RF interactions in a synchrotron.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation( ... )
    >>> rf_station_observation = RFStationPhaseObservation(each_turn_i=2, rf_station=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(rf_station_observation,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> plt.scatter(
    ...     rf_station_observation.turns_array[[turn_0, turn_2]],
    ...     rf_station_observation.phases[[turn_0, turn_2]],
    ... )
    >>> plt.plot(
    ...     rf_station_observation.turns_array[:], rf_station_observation.phases[:]
    ... )
    """

    def __init__(
        self,
        each_turn_i: int,
        rf_station: SingleHarmonicRFStation,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._rf_station = rf_station
        self._phases: DenseArrayRecorder | None = None
        self._omegas: DenseArrayRecorder | None = None
        self._voltages: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_harmonics = int(self._rf_station.n_rf)
        shape = (n_entries, n_harmonics)
        self._phases = DenseArrayRecorder(
            f"{self.common_filepath}_phases",
            shape,
        )
        self._omegas = DenseArrayRecorder(
            f"{self.common_filepath}_omegas",
            shape,
        )
        self._voltages = DenseArrayRecorder(
            f"{self.common_filepath}_voltages",
            shape,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._phases.write(self._rf_station.phi_rf)
        self._omegas.write(self._rf_station.omega_rf)
        self._voltages.write(self._rf_station.voltage)

    @property  # as readonly attributes
    def phases(self) -> NumpyArray:
        """
        RF station's effective phase of shape ``(n_observations, )``, in [rad].

        Returns
        -------
        phases
            Array of RF phases.
        """
        return self._phases.get_valid_entries()

    @property  # as readonly attributes
    def omegas(self) -> NumpyArray:
        """
        RF station's angular frequency of shape ``(n_observations, )``, in [Hz].

        Returns
        -------
        omegas
            Array of RF angular frequencies.
        """
        return self._omegas.get_valid_entries()

    @property  # as readonly attributes
    def voltages(self) -> NumpyArray:
        """
        RF station's effective voltage of shape ``(n_observations, )``, in [V].

        Returns
        -------
        voltages
            Array of RF voltages.
        """
        return self._voltages.get_valid_entries()


class StaticProfileObservation(ObservablesOncePerTurnBase):
    """
    Observation of a static beam profile.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profile
        Class for the calculation of beam profile
        that doesn't change its parameters.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = StaticProfileObservation(each_turn_i=2, profile=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(
    ...         profile_obs.hist_x, profile_obs.hist_y[index, :]
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        profile: StaticProfile,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            n_turns=n_turns,
            beam=beam,
        )
        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._profile.n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            (n_entries, n_bins),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._hist_y.write(
            copy_to_cpu(self._profile.hist_y),
        )

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray:
        """
        Histogram x axis, always the same.

        Returns
        -------
        hist_x
            Histogram x-axis array.
        """
        return copy_to_cpu(self._profile.hist_x)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram amplitude for each observed turn.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()


class StaticMultiProfileObservation(ObservablesOncePerTurnBase):
    """
    Observation of multiple profiles in one observation object. The profiles need to have the same n_bins.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profiles
        List of class for the calculation of beam profile
        that doesn't change its parameters.
    folder
        Path to the target folder used for
        saving or loading files.
    sort_profiles_by_section
        Whether to sort profiles by section index.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = StaticMultiProfileObservation(each_turn_i=2, profiles=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> # This example assumes that two profiles are in `profile_obs`
    >>> turn_0_profile0 = 0  # turn_0 simulation
    >>> turn_0_profile1 = 1  # turn_0 simulation
    >>> turn_2_profile0 = 2  # after 2 turns, because `each_turn_i = 2`
    >>> turn_2_profile1 = 3  # after 2 turns, because `each_turn_i = 2`
    >>> plt.plot(
    >>>     profile_obs.hist_x[0], profile_obs.hist_y[0][0]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[1], profile_obs.hist_y[0][1]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[0], profile_obs.hist_y[1][0]
    >>> )
    >>> plt.plot(
    >>>     profile_obs.hist_x[1], profile_obs.hist_y[1][1]
    >>> )
    """

    def __init__(
        self,
        each_turn_i: int,
        profiles: list[StaticProfile],
        folder: str = "",
        sort_profiles_by_section=True,
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)

        if sort_profiles_by_section:
            profiles = sorted(profiles, key=lambda prof: prof.section_index)
        self._profiles = profiles

        assert all(
            prof.n_bins == self._profiles[0].n_bins for prof in self._profiles
        ), "n_bins should be equal for all given profiles"

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # this is not used in this context
        n_turns: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )
        n_turns_observation = int(len(self._turns_array) // self.each_turn_i)
        n_bins = self._profiles[0].n_bins
        shape = (n_turns_observation, len(self._profiles), n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            shape,
        )

    def _update(self) -> None:
        """Update the data."""
        self._hist_y.write(
            [copy_to_cpu(prof.hist_y) for prof in self._profiles]
        )

    @property  # as readonly attributes
    def hist_x(self) -> list[NumpyArray]:
        """
        Histogram x axis, always the same of shape ``((n_bins, ), ..)``.

        Returns
        -------
        hist_x
            List of histogram x-axis arrays.
        """
        return [
            copy_to_cpu(self._profiles[i].hist_x)
            for i in range(len(self._profiles))
        ]

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram of given profiles of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()


class WakeFieldObservation(ObservablesOncePerTurnBase):
    """
    Observe the calculation of wake-fields.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    wakefield
        Manager class to calculate wake-fields.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> wake_obs = WakeFieldObservation(wakefield=..., each_turn_i=2)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(wake_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(wake_obs.induced_voltage[index, :])
    """

    def __init__(
        self,
        each_turn_i: int,
        wakefield: WakeField,
        folder: str = "",
    ):
        super().__init__(
            each_turn_i=each_turn_i,
            folder=folder,
        )
        self._wakefield = wakefield
        self._induced_voltage: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,  # not used in this context
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._wakefield._profile.n_bins)
        self._induced_voltage = DenseArrayRecorder(
            f"{self.common_filepath}_induced_voltage",
            (n_entries, n_bins),
        )

    def _update(self) -> None:
        """Update memory with new values."""
        try:
            self._induced_voltage.write(
                self._wakefield.induced_voltage,
            )
        except AttributeError:
            self._induced_voltage.write(
                np.zeros(self._wakefield._profile.n_bins)
            )

    @property  # as readonly attributes
    def induced_voltage(self) -> NumpyArray:
        """
        Induced voltage, in [V] from given beam profile and sources  of shape ``(n_observations, n_bins)``.

        Returns
        -------
        induced_voltage
            Array of induced voltages.
        """
        return self._induced_voltage.get_valid_entries()


class DynamicProfileConstNBinsObservation(ObservablesOncePerTurnBase):
    """
    Observation of a dynamic beam profile with changing width, while keeping a constant bin number.

    Parameters
    ----------
     each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profile
        Class for the calculation of beam profile
        with a change in width, but a constant bin number.
    folder
        Path to the target folder used for
        saving or loading files.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from blond import Simulation
    >>> sim = Simulation(...)
    >>> profile_obs = DynamicProfileConstNBinsObservation(each_turn_i=2, profile=...)
    >>> sim.run_simulation(
    ...     beams=...,
    ...     observe=(profile_obs,),
    ... )
    >>> turn_0 = 0  # first turn
    >>> turn_2 = 1  # after 2 turns, because `each_turn_i = 2`
    >>> for index in (turn_0, turn_2):
    ...     plt.plot(
    ...         profile_obs.hist_x[index, :], profile_obs.hist_y[index, :]
    ...     )
    """

    def __init__(
        self,
        each_turn_i: int,
        profile: DynamicProfileConstNBins,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._profile = profile
        self._hist_y: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :func:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns)
        n_bins = int(self._profile.n_bins)
        shape = (n_entries, n_bins)
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
            shape,
        )
        self._hist_x = DenseArrayRecorder(
            f"{self.common_filepath}_hist_x",
            shape,
        )

    def _update(self) -> None:
        """Update memory with new values."""
        self._hist_y.write(self._profile.hist_y)
        self._hist_x.write(self._profile.hist_x)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """
        Histogram amplitude of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_y
            Histogram amplitude array.
        """
        return self._hist_y.get_valid_entries()

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray:
        """
        Get x-axis of histogram, in [s], i.e. `bin_centers` of shape ``(n_observations, n_bins)``.

        Returns
        -------
        hist_x
            Histogram x-axis array.
        """
        return self._hist_x.get_valid_entries()


class SimulationObservation(ObservablesOncePerTurnBase):
    """
    Observation of the `Simulation` object itself.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    folder
        Path to the target folder used for
        saving or loading files.
    """

    def __init__(
        self,
        each_turn_i: int,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._simulation: Simulation | None = None

        self._t_revs: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :func:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        n_entries = self._calc_n_entries(n_turns=n_turns)
        shape = n_entries
        self._t_revs = DenseArrayRecorder(
            f"{self.common_filepath}_t_revs",
            shape,
        )
        self._simulation = simulation

    def _update(
        self,
    ) -> None:
        """Update memory with new values."""
        self._t_revs.write(self._simulation.current_t_rev)

    @property  # as readonly attributes
    def t_revs(self) -> NumpyArray:
        """
        Revolution time, in [s] of shape ``(n_observations)``.

        Returns
        -------
        t_rev
            Revolution time, in [s] of shape ``(n_observations)``.
        """
        return self._t_revs.get_valid_entries()


class DriftObservation(ObservablesOncePerTurnBase):
    """
    Observation of `eta_0` of the `DriftSimple` object.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    drift
        `DriftSimple` object.
    folder
        Path to the target folder used for
        saving or loading files.
    """

    def __init__(
        self,
        each_turn_i: int,
        drift: DriftSimple,
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)
        self._drift: DriftSimple = drift

        self._eta_0s: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when :func:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

        self._eta_0s = DenseArrayRecorder(
            f"{self.common_filepath}_eta_0s",
            (self._calc_n_entries(n_turns=n_turns)),
        )

    def _update(
        self,
    ) -> None:
        """Update memory with new values."""
        self._eta_0s.write(float(self._drift._last_eta_0))

    @property  # as readonly attributes
    def eta_0s(self) -> NumpyArray:
        """
        Drift in arc parameter eta of shape ``(n_observations)``.

        Returns
        -------
        eta_0
            Drift in arc parameter eta of shape ``(n_observations)``.
        """
        return self._eta_0s.get_valid_entries()
