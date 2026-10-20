# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Validation of the profile grid and the poles of `MultiPoleSparseSolve`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.generals.cupy_.no_cupy_import import copy_to_cpu

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

# Slack on the whole-bin check of the profile's own gaps, as a fraction of a
# bin; no looser than the kernel's 1e-6 bin, which decides whether a bin is
# two bins behind the read-out.
_BIN_GRID_TOLERANCE = 1e-6


def _check_whole_bin_gaps(
    hist_x: NumpyArray | CupyArray, bin_dt: float
) -> None:
    """
    Check that consecutive bins lie a whole number of bins apart.

    The near field takes the neighbours at exactly one bin, the far field
    every bin at two bins and more, so a neighbour between one and two bins
    away would be counted at the wrong lag or not at all.

    Parameters
    ----------
    hist_x
        Bin centres, in [s], increasing.
    bin_dt
        Profile bin width, in [s].

    Raises
    ------
    ValueError
        If two consecutive bins are not a whole number (at least one) of
        bin widths apart.
    """
    spacing_bins = copy_to_cpu(hist_x[1:] - hist_x[:-1]) / bin_dt
    whole_bins = np.round(spacing_bins)
    is_off_grid = (whole_bins < 1) | (
        np.abs(spacing_bins - whole_bins) > _BIN_GRID_TOLERANCE
    )
    if np.any(is_off_grid):
        first = int(np.argmax(is_off_grid))
        raise ValueError(
            "MultiPoleSparseSolve: the bins of the profile must lie a whole "
            f"number of bin widths ({bin_dt} s) apart, but bins {first} and "
            f"{first + 1} are {spacing_bins[first]} bin widths apart."
        )


def _check_decaying_poles(poles: NumpyArray | CupyArray) -> None:
    """
    Check that every pole's wake decays, Re(p) < 0.

    A pole with Re(p) >= 0 has a wake that grows, or never fades: the
    far-field state then grows without bound, and the GEMM far field's
    tabulated powers of exp(p bin_dt) overflow to inf.

    Parameters
    ----------
    poles
        Poles of the model, in [rad/s].

    Raises
    ------
    ValueError
        If a pole has a non-negative real part.
    """
    is_not_decaying = copy_to_cpu(poles).real >= 0
    if np.any(is_not_decaying):
        first = int(np.argmax(is_not_decaying))
        raise ValueError(
            "MultiPoleSparseSolve: every pole must have a negative real "
            f"part (a decaying wake), but pole {first} is "
            f"{complex(copy_to_cpu(poles)[first])} rad/s."
        )
