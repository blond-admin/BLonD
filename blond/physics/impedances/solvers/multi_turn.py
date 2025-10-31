"""Solvers to calculate the wake potential from impedance sources.

This module considers multi-turn wake fields.

Authors
-------
Leonard Thiele
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import elementary_charge as e

from blond import Simulation, WakeField
from blond._core.beam.base import BeamBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

    from ...profiles import (
        ProfileBaseClass,
    )
    from ..base import TimeDomain, WakeFieldSolver


class ProfileHistoryHelper:
    """Saves several profiles temporarily.

    Parameters
    ----------
    n_last_profile
        Number of past beam profiles to be considered
        for calumniation of the wake field.

    """

    def __init__(
        self,
        n_last_profile: int,
    ) -> None:
        self.n_last_profile = n_last_profile

        self._history_hist_x = []
        self._history_hist_y = []
        self._history_hist_t = []

    def add_to_history(
        self,
        beam_time: float,
        profile: ProfileBaseClass,
    ) -> None:
        """Add a profile observation to the history.

        Parameters
        ----------
        beam_time
            Time of the observation, in [s].
        profile
            Corresponding beam profilee.
        """
        # todo make this more efficient memory-wise
        self._history_hist_x.append(profile.hist_x)
        self._history_hist_y.append(profile.hist_y)
        self._history_hist_t.append(beam_time)

        self._history_hist_x = self._history_hist_x[-self.n_last_profile]
        self._history_hist_y = self._history_hist_y[-self.n_last_profile]
        self._history_hist_t = self._history_hist_t[-self.n_last_profile]

    def __iter__(self):
        """Iterate past/current hist_x, hist_y and t."""
        return zip(
            self._history_hist_x,
            self._history_hist_y,
            self._history_hist_t,
            strict=False,
        )


class MultiTurnTimeDomainFftSolver(WakeFieldSolver):
    """Wakefield solver that considers the beam in several turns.

    This solver keeps several observations of the beam profile
    in memory. The beam profile is transferred to the history
    when `calc_induced_voltage` is executed.
    Internally, the `fftconvolve` is used to convolute
    the beam profile with the wake filed kernel function of the source.

    Parameters
    ----------
    n_last_profile
        Number of past beam profiles to be considered
        for calumniation of the wake field.
    """

    def __init__(
        self,
        n_last_profile: int,
    ):
        self._profile_history_helper = ProfileHistoryHelper(
            n_last_profile=n_last_profile,
        )
        self._simulation: Simulation | None = None
        self._parent_wakefield: WakeField | None = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            Simulation context manager
        parent_wakefield
            Wakefield that this solver affiliated to
        """
        if parent_wakefield.profile is None:
            raise ValueError("Parent wakefield needs to have a profile.")
        self._parent_wakefield = parent_wakefield
        self._simulation = simulation

        for source in self._parent_wakefield.sources:
            if source.is_dynamic:
                raise NotImplementedError(
                    "All sources must be static for multi-turn wakes."
                )

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """Calculate the effect from each past profile to the current."""
        self._profile_history_helper.add_to_history(
            beam_time=beam.reference_time,  # todo does the profile offer this variable already?
            profile=self._parent_wakefield.profile,
        )

        # For each past histogram
        induced_voltage = 0  # will be casted to ndarray
        for hist_x, _hist_y, t in self._profile_history_helper:
            t_diff = beam.reference_time - t  # gets bigger

            # For each source
            wake_impedance = 0  # will be casted to ndarray
            for source in self._parent_wakefield.sources:
                source: TimeDomain
                wake_impedance += source.get_wake_impedance(
                    time=hist_x + t_diff,
                    simulation=self._simulation,
                    beam=beam,
                    n_fft=(2 * len(self._parent_wakefield.profile.hist_x)),
                )
            # TODO Use the
            _factor = (-1 * beam.particle_type.charge * e) * (
                # TODO this might be a problem with MPI
                beam.ratio
            )
            # Calculate the convolution of the wake and the beam
            # Usually this would be np.convolve(wake, beam).
            # This can be also done via fftconvolve.
            # Using ifft(fft(wake) * fft(beam)).
            # fft(wake)  is already precalculated in the memory.
            induced_voltage += _factor * np.fft.irfft(
                wake_impedance
                * self._parent_wakefield.profile.beam_spectrum(
                    n_fft=(2 * len(self._parent_wakefield.profile.hist_x))
                ),
            )

        return induced_voltage
