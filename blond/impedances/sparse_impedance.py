# coding: utf8
# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Induced voltage from a pole-residue impedance model on sparse profiles,
with multi-turn wake memory**

The solver ports the pole-residue (vector-fitted) wake evaluation of the
BLonD ``spare-profile-and-pole-wakes`` development branch
(``blond/physics/impedances/solvers.py::MultiPoleSparseSolve`` and the
``wake_from_pole_residue`` backend kernel) onto the SparseBatch /
SparseBucket profiles of this tree.

Each impedance source contributes complex pole/residue pairs
(``Resonators.get_vectorfit``). The wake of one pole is
:math:`W(t) = 2\\,\\Re(A_k e^{s_k t})`, so the induced voltage is a
recursion with one complex state per pole: within a window the state
decays by ``exp(pole * bin_size)`` per bin, and across the empty buckets
between windows (and across turns) it decays analytically in a single
step. This is what makes the scheme both sparse-friendly and naturally
multi-turn: the wake memory is the pole states, not a stored wake array.

:Authors: **Lina Valle**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import e

from ..beam.sparse_profiles import SparseProfileBaseClass
from ..utils import bmath as bm

if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable, Optional

    from numpy.typing import NDArray as NumpyArray

    from ..beam.beam import Beam
    from ..input_parameters.rf_parameters import RFStation
    from .impedance_sources import Resonators


def wake_from_pole_residue(
    profile: NumpyArray,
    profile_dts: NumpyArray,
    poles: NumpyArray,
    residues: NumpyArray,
    update_on_bin: NumpyArray,
    factor: float,
    states: NumpyArray,
    voltage: NumpyArray,
) -> None:
    """Accumulate the pole-residue wake of `profile` into `voltage`.

    Parameters
    ----------
    profile
        Beam profile histogram on the (possibly sparse) memory grid.
    profile_dts
        Bin centers belonging to `profile`, monotonically increasing, in [s].
    poles
        Complex poles of the equivalent circuit model, in [1/s].
    residues
        Complex residues of the equivalent circuit model, in [Ohm/s].
    update_on_bin
        Memory indices where a new profile window starts and the time step
        must be re-evaluated. Must start with 0.
    factor
        Charge per histogram count, in [C] (converts `profile` to charge
        per bin).
    states
        Complex state vector of length ``len(poles) + 1``; entry ``-1``
        holds the reference time of the left edge of the first bin
        (edge-based state semantics). Persist it between calls — shifted
        back by the elapsed time — to accumulate a multi-turn wake.
    voltage
        Output induced voltage per memory bin, in [V]. Overwritten.
    """
    n_poles = len(poles)
    two_factor = 2 * factor
    n_bins = len(profile)

    voltage[:] = 0

    t_start = states[-1]

    for pole_i in range(n_poles):
        i_update = 0
        update_on_bin_i = update_on_bin[i_update]

        pole = complex(poles[pole_i])
        residue = complex(residues[pole_i])
        state = complex(states[pole_i])

        decay = 0.0 + 0j
        for bin_i in range(n_bins):
            profile_i_half = 0.5 * profile[bin_i] * two_factor

            if bin_i == update_on_bin_i:
                if bin_i == 0:
                    t_jump = profile_dts[0] - t_start + 0j
                else:
                    t_jump = profile_dts[bin_i] - profile_dts[bin_i - 1] + 0j
                state *= np.exp(pole * t_jump)
                dt = profile_dts[bin_i + 1] - profile_dts[bin_i]
                decay = np.exp(pole * dt)

                i_update += 1
                if i_update < len(update_on_bin):
                    update_on_bin_i = update_on_bin[i_update]
            else:
                state *= decay
            state += profile_i_half
            voltage[bin_i] += float(np.real(residue * state))
            state += profile_i_half
        states[pole_i] = state

    states[-1] = profile_dts[-1]


class InducedVoltageSparseMTW:
    """Induced voltage on a sparse profile with multi-turn wake memory.

    Parameters
    ----------
    beam
        Beam object.
    profile
        SparseBatch / SparseBucket (any SparseProfileBaseClass), or a
        standard Profile-like object exposing ``bin_centers`` and
        ``n_macroparticles`` on a contiguous grid.
    resonators
        The Resonators impedance source(s); every source must implement
        ``get_vectorfit()``.
    rf_station
        RFStation used to advance the wake memory by one revolution
        period per call. Without it, only single-turn wakes are computed.

    Attributes
    ----------
    induced_voltage : float array
        Induced voltage per memory bin, in [V], on the time-ordered
        concatenated grid of the sparse profile.
    """

    def __init__(
        self,
        beam: Beam,
        profile,
        resonators: Resonators | Iterable[Resonators],
        rf_station: Optional[RFStation] = None,
    ) -> None:
        self.beam = beam
        self.profile = profile
        self.rf_station = rf_station

        try:
            resonators = list(resonators)
        except TypeError:
            resonators = [resonators]
        poles = []
        residues = []
        for source in resonators:
            if not hasattr(source, "get_vectorfit"):
                raise AttributeError(
                    f"The source {source!r} should implement `get_vectorfit`."
                )
            poles_, residues_ = source.get_vectorfit()
            poles.extend(poles_)
            residues.extend(residues_)
        self.poles = np.array(poles, dtype=complex)
        self.residues = np.array(residues, dtype=complex)

        self.process()

    @property
    def _is_sparse(self) -> bool:
        return isinstance(self.profile, SparseProfileBaseClass)

    def _memory_grid(self) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
        """Time-ordered concatenated bin centers, histogram and window
        start indices of the profile."""
        if self._is_sparse:
            return (
                self.profile.continuous_bin_centers,
                self.profile.continuous_n_macroparticles,
                self.profile.update_on_bin,
            )
        return (
            self.profile.bin_centers,
            self.profile.n_macroparticles,
            np.zeros(1, dtype=np.int64),
        )

    def process(self) -> None:
        """(Re-)initialise the wake memory, e.g. after the profile changed.

        Discards the accumulated multi-turn wake.
        """
        bin_centers, _, _ = self._memory_grid()
        self._states = np.zeros(len(self.poles) + 1, dtype=complex)
        # Left EDGE of the first bin, so that the first window's time jump
        # is zero on the first call (edge-based state semantics).
        bin_size = self.profile.bin_size
        self._states[-1] = bin_centers[0] - bin_size / 2.0
        self.induced_voltage = np.zeros(
            len(bin_centers), dtype=bm.precision.real_t
        )

    def induced_voltage_generation(self) -> None:
        """Compute the induced voltage of the current profile, adding the
        decayed wake of the previous turns."""
        bin_centers, hist, update_on_bin = self._memory_grid()

        if len(self.induced_voltage) != len(bin_centers):
            # multi-turn injection changed the number of windows: keep the
            # pole states (the wake memory is per pole, not per bin), only
            # the output buffer needs the new size
            self.induced_voltage = np.zeros(
                len(bin_centers), dtype=bm.precision.real_t
            )

        if self.rf_station is not None:
            # Advance the wake memory to this turn: shift the reference
            # time back by one revolution period
            turn = self.rf_station.counter[0]
            self._states[-1] -= self.rf_station.t_rev[turn]
            if self._states[-1].real > bin_centers[0]:
                raise RuntimeError(
                    "The wake memory reference time is ahead of the first "
                    "profile bin; process() must be called after a profile "
                    "change that moves the window starts backwards in time."
                )

        factor = -self.beam.particle.charge * e * self.beam.ratio
        wake_from_pole_residue(
            profile=np.ascontiguousarray(hist, dtype=float),
            profile_dts=np.ascontiguousarray(bin_centers, dtype=float),
            poles=self.poles,
            residues=self.residues,
            update_on_bin=update_on_bin,
            factor=factor,
            states=self._states,
            voltage=self.induced_voltage,
        )

    def track(self) -> None:
        """Compute the induced voltage and apply the kick to the beam."""
        self.induced_voltage_generation()

        charge = self.beam.particle.charge
        if self._is_sparse:
            n_p = self.profile.number_of_slices_per_profile
            for k, p in enumerate(self.profile.memory_time_order):
                profile = self.profile.profiles_list[p]
                bm.linear_interp_kick(
                    dt=self.beam.dt,
                    dE=self.beam.dE,
                    voltage=np.ascontiguousarray(
                        self.induced_voltage[k * n_p : (k + 1) * n_p]
                    ),
                    bin_centers=profile.bin_centers,
                    charge=charge,
                    acceleration_kick=0.0,
                )
        else:
            bm.linear_interp_kick(
                dt=self.beam.dt,
                dE=self.beam.dE,
                voltage=self.induced_voltage,
                bin_centers=self.profile.bin_centers,
                charge=charge,
                acceleration_kick=0.0,
            )
