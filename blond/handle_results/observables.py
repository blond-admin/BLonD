# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module holding all observables for the simulation.

Author
------
Simon Lauber
Leonard Thiele
Elleanor Lamb
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NumpyArray

from blond.core.base import MainLoopRelevant
from blond.handle_results.array_recorders import DenseArrayRecorder

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond import WakeField
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import SingleHarmonicRfStation
    from blond.physics.profiles import DynamicProfileConstNBins, StaticProfile

logger = logging.getLogger(__name__)


class ObservablesBaseClass(MainLoopRelevant):
    """Base class to define observations.

    Parameters
    ----------
    folder
        Target folder to save the data at.
        Use `rename` to change the ddestination.
    """

    def __init__(self, folder: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if len(folder) > 0:
            assert folder.endswith("/") or folder.endswith("\\")
        self.common_filepath = folder + "last"
        logger.info(f"Will save {self} to {self.common_filepath}_,,,")

    def get_recorders(self) -> list[tuple[str, DenseArrayRecorder]]:
        """Get all `DenseArrayRecorder` inside the current instance.

        Returns
        -------
        recorders
            List of ((attribute name, attribute), ...)
        """
        self.assert_lateinit()
        recorders = [
            (attribute, instance)
            for attribute, instance in self.__dict__.items()
            if isinstance(instance, DenseArrayRecorder)  # initialized
        ]
        return recorders

    def rename(self, new_common_filepath: str) -> None:
        """Change the common save name of all internal arrays.

        Notes
        -----
        This has no effect on files that are already saved to the disk.

        Parameters
        ----------
        new_common_filepath
            The new common name of all internal arrays.

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
        """Checks that DenseArrays are already initialized."""
        for parameter, value in self.__dict__.items():
            if value is None:  # uninitialized
                assert value is not None, f"`{parameter}` was not initialized."


class ObservablesOncePerTurnBase(ObservablesBaseClass):
    """Base class to observe attributes during simulation.

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
        **kwargs,
    ):
        super().__init__(folder=folder, **kwargs)
        self.each_turn_i = each_turn_i

        self._n_turns: int | None = None
        self._turn_i_init: int | None = None
        self._turns_array: NumpyArray | None = None

        self._last_turn_i_observed = (
            -1
        )  # to avoid double recordings with multiple drifts in one section
        self._last_section_i_observed = -1

    @property  # as readonly attributes
    def turns_array(self) -> NumpyArray | None:
        """Helper method to get x-axis array with turn-number.

        Helper method to get x-axis array with turn-number for which the
        observations are performed.
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
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self._n_turns = int(n_turns)
        self._turn_i_init = int(turn_i_init)

        self._turns_array = np.linspace(
            0, n_turns, num=n_turns // self.each_turn_i + 1, dtype=int
        )
        self._turns_array = np.append(
            np.array([0]), self._turns_array
        )  # prepend 0 for pre-running


class BeamObservationOncePerTurn(ObservablesOncePerTurnBase):
    """Observe the bunch coordinates during simulation execution after a drift element.

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

    def __init__(
        self,
        each_turn_i: int,
        beam: BeamBaseClass,
        folder: str = "",
    ):
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
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation :class:`~blond._cycles_core.beam.beam.Beam` object
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
        )

        n_entries = n_turns // self.each_turn_i + 2
        n_macroparticles = int(beam.common_array_size)
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
        """Returns reference time."""
        return self._reference_time.get_valid_entries()

    @property  # as readonly attributes
    def reference_total_energy(self):
        """Returns total energy."""
        return self._reference_total_energy.get_valid_entries()

    @property  # as readonly attributes
    def dts(self):
        """Returns array of dts."""
        return self._dts.get_valid_entries()

    @property  # as readonly attributes
    def dEs(self):
        """Returns array of dEs."""
        return self._dEs.get_valid_entries()

    @property  # as readonly attributes
    def flags(self):
        """Returns flags of particles, eg if lost or not."""
        return self._flags.get_valid_entries()


class CavityPhaseObservation(ObservablesOncePerTurnBase):
    """Observe the RF cavity parameters during the execution of the simulation.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    cavity
        Class that implements beam-RF interactions in a synchrotron
    folder
        Path to the target folder used for
        saving or loading files.
    """

    def __init__(
        self,
        each_turn_i: int,
        cavity: SingleHarmonicRfStation,
        folder: str = "",
    ):
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
            Simulation `Beam` object
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
        )

        n_entries = n_turns // self.each_turn_i + 2
        n_harmonics = int(self._cavity.n_rf)
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
            # TODO: should be property call instead of private member
        )
        self._voltages.write(
            self._cavity.voltage,
        )

    @property  # as readonly attributes
    def phases(self) -> NumpyArray:
        """Cavity's effective phase, in [rad]."""
        return self._phases.get_valid_entries()

    @property  # as readonly attributes
    def omegas(self) -> NumpyArray:
        """Cavity's angular frequency, in [Hz]."""
        return self._omegas.get_valid_entries()

    @property  # as readonly attributes
    def voltages(self) -> NumpyArray:
        """Cavity's effective voltage, in [V]."""
        return self._voltages.get_valid_entries()


class StaticProfileObservation(ObservablesOncePerTurnBase):
    """Observation of a static beam profile.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profile
        Class for the calculation of beam profile
        that doesn't change its parameters
    folder
        Path to the target folder used for
        saving or loading files.
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
        turn_i_init: int,
        **kwargs: dict[str, Any],
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
            f"{self.common_filepath}_hist_y",
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
        if (
            self._last_turn_i_observed == simulation.turn_i.value
            and self._last_section_i_observed == simulation.section_i.value
        ):
            return
        self._last_turn_i_observed = simulation.turn_i.value
        self._last_section_i_observed = simulation.section_i.value
        self._hist_y.write(
            self._profile.hist_y,
        )
        # else return without recording

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """Histogram amplitude."""
        return self._hist_y.get_valid_entries()


class StaticMultiProfileObservation(ObservablesOncePerTurnBase):
    """Observation of multiple profiles in one observation object. The profiles need to have the same n_bins.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    profiles
        List of class for the calculation of beam profile
        that doesn't change its parameters
    folder
        Path to the target folder used for
        saving or loading files.
    """

    def __init__(
        self,
        each_turn_i: int,
        profiles: list[StaticProfile],
        folder: str = "",
    ):
        super().__init__(each_turn_i=each_turn_i, folder=folder)

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
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )
        n_entries = int(
            (len(self._turns_array) * len(self._profiles)) // self.each_turn_i
            + 2 * len(self._profiles)
        )
        n_bins = self._profiles[0].n_bins
        self._hist_y = DenseArrayRecorder(
            f"{self.common_filepath}_hist_y",
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
            and self._last_section_i_observed == simulation.section_i.value
        ):
            return
        self._last_turn_i_observed = simulation.turn_i.value
        self._last_section_i_observed = simulation.section_i.value
        for prof in self._profiles:
            if simulation.section_i.value == prof.section_index:
                self._hist_y.write(prof.hist_y)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """Histogram of given profiles."""
        return self._hist_y.get_valid_entries()


class WakeFieldObservation(ObservablesOncePerTurnBase):
    """Observe the calculation of wake-fields.

    Parameters
    ----------
    each_turn_i
        Value to control that the element is
        callable each n-th turn.
    wakefield
        Manager class to calculate wake-fields
    folder
        Path to the target folder used for
        saving or loading files.
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
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
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
        )

        n_entries = n_turns // self.each_turn_i + 2
        n_bins = int(self._wakefield._profile.n_bins)
        self._induced_voltage = DenseArrayRecorder(
            f"{self.common_filepath}_induced_voltage",
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
        """Induced voltage, in [V] from given beam profile and sources.

        Returns
        -------
        induced_voltage

        """
        return self._induced_voltage.get_valid_entries()


class DynamicProfileConstNBinsObservation(ObservablesOncePerTurnBase):
    """Observation of a dynamic beam profile with changing width, while keeping a constant bin number.

    Parameters
    ----------
     each_turn_i
        Value to control that the element is
        callable each n-th turn
    profile
        Class for the calculation of beam profile
        with a change in width, but a constant bin number
    folder
        Path to the target folder used for
        saving or loading files.

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
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when :func:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

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
        )

        n_entries = n_turns // self.each_turn_i + 2
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
        self._hist_y.write(self._profile.hist_y)
        self._hist_x.write(self._profile.hist_x)

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray:
        """Histogram amplitude."""
        return self._hist_y.get_valid_entries()

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray:
        """x-axis of histogram, in [s], i.e. `bin_centers`."""
        return self._hist_x.get_valid_entries()
