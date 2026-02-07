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

    def _update_kernel_multiturn(self) -> None:
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

    def calc_induced_voltage(self, beam: BeamBaseClass) -> np.ndarray:
        if self._kernel_multiturn is None:
            self._update_kernel_multiturn()

        profile: EquidistantMultiProfile = self._parent_wakefield.profile
        hist = profile._continuous_memory_hist_y
        mask = profile._continuous_memory_mask_prof

        n_fft = len(self._kernel_multiturn)

        # -------------------------------------------------
        # FFT (MKL, multithreaded)
        # -------------------------------------------------
        # zero-padded rfft
        H = mkl_fft.rfft(hist, n=n_fft)

        # frequency-domain multiply
        H *= self._rfft_kernel_multiturn

        # inverse FFT
        induced_voltage_multiturn = mkl_fft.irfft(H, n=n_fft)

        # -------------------------------------------------
        # MULTITURN ACCUMULATION
        # -------------------------------------------------
        n_single = len(hist)

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

        _factor = -(beam.particle_type.charge * e) * (
            beam.intensity / beam.common_array_size
        )

        return _factor * induced_voltage
