# coding: utf8
# Copyright 2014-2026 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Compiled kernels for the trackers.**

They follow :mod:`blond.llrf.cavity_loop_kernels`: they are used when numba
is available, and the environment variable ``BLOND_DISABLE_NUMBA_KERNELS``
forces the original code paths.

:Authors: **Lina Valle**
"""

import numpy as np

from ..llrf.cavity_loop_kernels import njit


@njit(cache=True)
def sparse_linear_interp_kick(
    dt,
    dE,
    voltage,
    bin_centers,
    n_slices,
    window_start,
    window,
    inv_bin_width,
    charge,
    acceleration_kick,
):
    r"""Interpolated kick with a sparse profile, in one pass over the
    particles instead of one pass per window.

    The windows of the profile have n_slices bins each; window p uses
    voltage[p * n_slices : (p + 1) * n_slices] and the same elements of
    bin_centers, and inv_bin_width[p] is the inverse of its bin width.
    window_start holds the first bin centre of the windows sorted in time and
    window the corresponding window numbers. The windows must not overlap.

    Each particle is kicked by the voltage interpolated linearly between the
    two bin centres around it, plus the acceleration kick, as
    linear_interp_kick does with the bins of one window. Particles outside
    all the windows are not kicked."""

    n_windows = len(window_start)
    k = 0
    for i in range(len(dt)):
        # Last window starting before the particle. The particles of a bunch
        # usually follow each other, so try the window of the previous
        # particle before searching
        if not (
            window_start[k] <= dt[i]
            and (k + 1 == n_windows or dt[i] < window_start[k + 1])
        ):
            k = np.searchsorted(window_start, dt[i], side="right") - 1
            if k < 0:
                k = 0
                continue
        first = window[k] * n_slices
        fbin = int(
            np.floor((dt[i] - bin_centers[first]) * inv_bin_width[window[k]])
        )
        if (fbin >= 0) and (fbin < n_slices - 1):
            fbin += first
            slope = (
                charge
                * (voltage[fbin + 1] - voltage[fbin])
                * inv_bin_width[window[k]]
            )
            dE[i] += (
                slope * (dt[i] - bin_centers[fbin])
                + charge * voltage[fbin]
                + acceleration_kick
            )
