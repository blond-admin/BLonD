# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The pole-residue model of `MultiPoleSparseSolve` on the bin grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.wake_per_bin.bin_average import (
    triple_box_average_pole,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


@dataclass(frozen=True, eq=False)
class _PoleModel:
    """
    The pole-residue model on the profile's bin grid, fixed for a run.

    Built once, at the first call of `MultiPoleSparseSolve`, from the
    sources and the profile; nothing in it changes afterwards.
    """

    # Poles p, in [rad/s]; -Re(p) > 0 is the decay rate, Im(p) the angular
    # frequency. A complex pole stands for its conjugate too.
    poles: NumpyArray | CupyArray
    # Residues, one per pole: the wake of a pole is rho * exp(p t).
    residues: NumpyArray | CupyArray
    # Per pole, +1 or -1: how a counter-rotating beam sees this pole.
    counterrotating_pole_signs: NumpyArray | CupyArray
    # Per pole, 2 for a complex pole (it stands for its conjugate as well)
    # and 1 for a real one.
    pair_factors: NumpyArray | CupyArray
    # Per pole, the residue of the bin-averaged wake past its onset at 1.5
    # bins, where it is a pure exponential; read two bins behind, so
    # wake(t) = tail_residue * exp(p (t - 2 bin_dt)).
    tail_residues: NumpyArray | CupyArray
    # Near field: the bin-averaged wake summed over all poles at lags of
    # +1, 0 and -1 bin, in [V/C]; applied to a bin's neighbours.
    taps: NumpyArray | CupyArray
    # Per pair of consecutive bins, True if they are one bin apart (so the
    # near taps apply); False across a gap in the profile.
    adjacent: NumpyArray | CupyArray
    # Profile bin width, in [s].
    bin_dt: float
    # Time from the first to the last bin centre of the profile, in [s].
    span_dt: float


@dataclass(frozen=True, eq=False)
class _FarField:
    """
    The poles the far-field recursion keeps, fixed for a run.

    Those whose wake from two bins on is above the tolerance. The others
    live only in the near taps and in the carried charge's first-bin term.
    """

    # The kept poles, in [rad/s].
    poles: NumpyArray | CupyArray
    # Their `_PoleModel.tail_residues`, handed to the kernel.
    residues: NumpyArray | CupyArray
    # Their `_PoleModel.counterrotating_pole_signs`.
    pole_signs: NumpyArray | CupyArray
    # The plan of `backend.specials.no_parity.wake_from_pole_residue_gemm`
    # on this grid, which then replaces `wake_from_pole_residue`; None
    # where the backend has none or it would not pay off.
    gemm_plan: Any = None


@dataclass(eq=False)
class _Carry:
    """What one call of `MultiPoleSparseSolve` hands to the next."""

    # The last bin as `factor * hist_y`: its charge in [C], negated as the
    # induced voltage needs. A length-1 array, overwritten in place.
    charge: NumpyArray | CupyArray
    # Whether the beam of that call was counter-rotating.
    is_counter_rotating: bool = False


def _is_far_field_pole(model: _PoleModel, tolerance: float) -> np.ndarray:
    """
    Which poles have a far field above the tolerance.

    Compares a pole's far-field wake, summed over every lag from two bins
    on, with the peak magnitude of its three near taps.

    Parameters
    ----------
    model
        The pole-residue model on the bin grid.
    tolerance
        Largest share of the near-field peak a dropped pole's far field
        may have.

    Returns
    -------
    is_far_field_pole
        Host boolean array, one entry per pole.
    """
    bin_dt = model.bin_dt
    # The three near-tap lags, in [s], along axis 0; the poles along axis 1,
    # so every pole is evaluated in one call.
    lags = backend.array([bin_dt, 0.0, -bin_dt], dtype=backend.float)[:, None]
    poles = model.poles[None, :]
    residues = model.residues[None, :]
    # The taps are the real part; with the residue turned by -90 degrees
    # they are the imaginary part, so together the magnitude, whatever the
    # residue's phase.
    in_phase = triple_box_average_pole(lags, poles, residues, bin_dt)
    quadrature = triple_box_average_pole(lags, poles, -1j * residues, bin_dt)
    # Per pole, the largest magnitude of its three near taps.
    near_field_peaks = np.max(
        np.hypot(copy_to_cpu(in_phase), copy_to_cpu(quadrature)), axis=0
    )
    # The far field from two bins on is a geometric series, so its sum is
    # tail_at_two_bins / (1 - decay_per_bin); compared below without that
    # division, which is by zero for an undamped pole.
    # |z| = |exp(p bin_dt)|, the factor by which the wake shrinks per bin.
    decay_per_bin = copy_to_cpu(backend.exp(model.poles.real * bin_dt))
    # Per pole, the magnitude of the far-field wake at a lag of 2 bins.
    tail_at_two_bins = copy_to_cpu(
        model.pair_factors * backend.abs(model.tail_residues)
    )
    return tail_at_two_bins >= (
        tolerance * near_field_peaks * (1.0 - decay_per_bin)
    )
