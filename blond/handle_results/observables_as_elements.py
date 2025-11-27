# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Logs energy and time at some points around the simulation, is inserted like all other elements.

Cannot be used with from_locals.

"""

from __future__ import annotations

from typing import Any

import numpy as np

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
            `Simulation` context manager
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
            `Simulation` context manager
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


class BunchObservationMetaParams(BeamObservationElement, ObservablesBaseClass):
    """Records mean and standard deviation of both energy and time coordinates and estimates the bunch emittance.

    The observation object needs to be placed in a section. Only one recording will be performed per section.

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
        super().__init__(folder=folder)

        self.each_turn_i = each_turn_i
        self._beam = beam

        self._sigma_dt: DenseArrayRecorder | None = None
        self._sigma_dE: DenseArrayRecorder | None = None
        self._mean_dt: DenseArrayRecorder | None = None
        self._mean_dE: DenseArrayRecorder | None = None
        self._rms_emittance: DenseArrayRecorder | None = None

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """Lateinit method when :func:`blond.core.simulation.simulation.Simulation.run_simulation` is called.

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

        count = sum([el == self for el in simulation.ring.elements.elements])

        n_entries = int(n_turns * count // self.each_turn_i)
        shape = n_entries

        self._mean_dt = DenseArrayRecorder(
            f"{self.common_filepath}_mean_dt",
            shape,
        )
        self._mean_dE = DenseArrayRecorder(
            f"{self.common_filepath}_mean_dE",
            shape,
        )
        self._sigma_dt = DenseArrayRecorder(
            f"{self.common_filepath}_sigma_dt",
            shape,
        )
        self._sigma_dE = DenseArrayRecorder(
            f"{self.common_filepath}_sigma_dE",
            shape,
        )
        self._rms_emittance = DenseArrayRecorder(
            f"{self.common_filepath}_emittance_stat",
            shape,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        pass

    def track(
        self,
        beam: BeamBaseClass,
    ) -> None:
        """Update memory with new values.

        Parameters
        ----------
        simulation
            Simulation context manager

        """
        if self._beam is not beam:
            return
        self._sigma_dt.write(np.std(self._beam._dt))
        self._sigma_dE.write(np.std(self._beam._dE))
        self._mean_dt.write(np.mean(self._beam._dt))
        self._mean_dE.write(np.mean(self._beam._dE))
        self._rms_emittance.write(
            np.sqrt(
                np.average(self._beam._dE**2) * np.average(self._beam._dt**2)
                - np.average(self._beam._dE * self._beam._dt) ** 2
            )
        )

    @property  # as readonly attributes
    def sigma_dt(self):
        """Standard deviation of the time coordinate."""
        return self._sigma_dt.get_valid_entries()

    @property  # as readonly attributes
    def sigma_dE(self):
        """Standard deviation of the energy coordinate, in [eV]."""
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
    def rms_emittance(self):
        r"""Root-mean-square emittance.

        The statistical emittance is calculated with

        .. math::
            \epsilon = \sqrt{\langle \Delta t^2 \\rangle \langle \Delta E^2 \\rangle - \langle \Delta t \Delta E \\rangle^2}
        """
        return self._rms_emittance.get_valid_entries()
