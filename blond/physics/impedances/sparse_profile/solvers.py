# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds the `MultiTurnSparseProfileSolver` to be used with `EquidistantMultiProfile`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mkl_fft
import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.constants import e

from blond import backend
from blond.core.simulation.simulation import Simulation
from blond.physics.impedances.base import (
    WakeField,
    WakeFieldSolver,
)
from blond.physics.profiles_sparse import EquidistantMultiProfile

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore

    from blond.core.base import DynamicParameter
    from blond.core.beam.base import BeamBaseClass


class MultiTurnSparseProfileSolver(WakeFieldSolver):
    """
    Wakefield solver that considers evely separated profiles.

    Parameters
    ----------
    n_turns
        Number of turns to accumulate for the wakefields.
    """

    def __init__(self, n_turns: int):
        super().__init__()
        self._n_turns = n_turns
        self._beam: BeamBaseClass | None = None

        self._turn_i: DynamicParameter | None = None
        self._parent_wakefield: WakeField | None = None
        self._simulation: Simulation | None = None

        self._kernel_multiturn: NumpyArray | CupyArray = None
        self._time_multiturn: NumpyArray | CupyArray = None
        self._mask_multiturn: NumpyArray | CupyArray = None

        self._previous_induced_voltage_multiturn: NumpyArray | CupyArray = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ):
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        self._parent_wakefield = parent_wakefield
        assert isinstance(parent_wakefield.profile, EquidistantMultiProfile)
        self._t_rev = simulation.get_t_rev_init()

        # Pre-allocate profile histogram to full FFT size to avoid zero-padding overhead

    def _resize_profile_for_fft(self) -> None:
        """
        Resize profile's continuous memory histogram to full FFT size.

        This eliminates zero-padding overhead in rfft by pre-allocating
        the histogram array to the size needed for n_turns convolution.
        The extra space is filled with zeros and never modified, so rfft
        can operate directly on the array without padding.
        """
        profile: EquidistantMultiProfile = self._parent_wakefield.profile

        # Calculate required FFT size
        original_size = len(profile._continuous_memory_hist_y)
        fft_size = original_size * self._n_turns

        # Create new larger array with zeros
        new_hist_y = backend.zeros(fft_size, dtype=backend.float)

        # Copy existing data to the beginning
        new_hist_y[:original_size] = profile._continuous_memory_hist_y

        # Replace profile's histogram with the larger pre-padded array
        profile._continuous_memory_hist_y = new_hist_y

        # Update profile views to point to the correct slice
        # (individual profile histograms are views into the continuous array)
        n = profile._bins_per_profile
        for i, prof in enumerate(profile.profiles):
            start = 2 * i * n
            stop = start + n
            prof._hist_y = profile._continuous_memory_hist_y[start:stop]

        # Store original size for masking
        self._original_profile_size = original_size

    def _update_kernel_multiturn(self) -> None:
        """Update the wakefield kernel, that represents a single particle wake."""
        profile: EquidistantMultiProfile = self._parent_wakefield.profile

        continuous_memory_mask = profile._continuous_memory_mask
        continuous_memory_hist_x = profile._continuous_memory_hist_x

        time_multiturn = np.concatenate(
            [
                continuous_memory_hist_x + i * self._t_rev
                for i in range(self._n_turns)
            ]
        )
        mask_multiturn = np.concatenate(
            [continuous_memory_mask for _ in range(self._n_turns)]
        )
        time_multiturn -= continuous_memory_hist_x[
            continuous_memory_mask
        ].min()
        kernel_size = len(profile._continuous_memory_hist_x) * self._n_turns
        kernel_multiturn = backend.zeros(kernel_size, dtype=backend.float)

        for source in self._parent_wakefield.sources:
            # calculate wake, skipping the
            # (intentionally) empty entries in between the profiles
            kernel_multiturn[mask_multiturn] = source.get_wake(
                time_multiturn[mask_multiturn]
            )

        self._time_multiturn = time_multiturn
        self._mask_multiturn = mask_multiturn
        self._kernel_multiturn = kernel_multiturn
        self._rfft_kernel_multiturn = backend.fft.rfft(
            self._kernel_multiturn, n=(len(self._kernel_multiturn))
        )
        self._resize_profile_for_fft()

    def calc_induced_voltage(self, beam: BeamBaseClass) -> np.ndarray:
        """
        Calculate the induced voltage.

        Parameters
        ----------
        beam
            Beam class to interact with this element.

        Returns
        -------
        induced_voltage
            The induced voltage in teh current turn.
        """
        if self._kernel_multiturn is None:
            _factor = -(beam.particle_type.charge * e) * (
                beam.intensity / beam.common_array_size
            )
            self._update_kernel_multiturn()
            self._rfft_kernel_multiturn *= _factor
            self.H = None
            self.induced_voltage_multiturn = None

        profile: EquidistantMultiProfile = self._parent_wakefield.profile
        hist = profile._continuous_memory_hist_y
        mask = profile._continuous_memory_mask_prof

        # -------------------------------------------------
        # OPTIMIZATION: No zero-padding needed!
        # hist is pre-allocated to full FFT size, with zeros at the end
        # -------------------------------------------------
        # Direct FFT without padding (hist is already the right size)
        if self.H is None:
            H = mkl_fft.rfft(hist)
            self.H = H
        else:
            mkl_fft.rfft(hist, out=self.H)
            H = self.H

        # frequency-domain multiply
        H *= self._rfft_kernel_multiturn

        # inverse FFT (output is same size as hist)
        if self.induced_voltage_multiturn is None:
            self.induced_voltage_multiturn = mkl_fft.irfft(H)
            induced_voltage_multiturn = self.induced_voltage_multiturn
        else:
            mkl_fft.irfft(H, out=self.induced_voltage_multiturn)
            induced_voltage_multiturn = self.induced_voltage_multiturn

        # -------------------------------------------------
        # MULTITURN ACCUMULATION
        # -------------------------------------------------
        n_single = self._original_profile_size  # Use original size, not padded

        if self._previous_induced_voltage_multiturn is None:
            self._previous_induced_voltage_multiturn = (
                induced_voltage_multiturn.copy()
            )
        else:
            induced_voltage_multiturn[:-n_single] += (
                self._previous_induced_voltage_multiturn[n_single:]
            )
            self._previous_induced_voltage_multiturn[:] = (
                induced_voltage_multiturn
            )

        induced_voltage = induced_voltage_multiturn[:n_single][mask]

        # Note: The padded region (hist[n_single:]) remains zero because:
        # - C++ sparse_histogram_strided only writes to first n_single bins
        # - The padding is never modified, stays zero for next FFT

        return induced_voltage
