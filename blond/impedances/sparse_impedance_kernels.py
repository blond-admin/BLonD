# coding: utf8
# Copyright 2014-2026 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Compiled kernel for the multi-pass resonator induced voltage on sparse
profiles.**

The kernel computes the direct double sum
:math:`V[m] = \\sum_j f_j \\sum_k q_j[k] \\, W(t_m - t_{j,k})`
over all source windows (current and remembered passes), evaluating the
analytic resonator wake inline with the same half-weight-at-zero
convention as :meth:`blond.impedances.impedance_sources.Resonators.wake_calc`.
Set the environment variable ``BLOND_DISABLE_NUMBA_KERNELS`` to any
non-empty value to force the pure-numpy code path (e.g. for A/B
validation).

:Authors: **Lina Valle**
"""

import os

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = not os.environ.get("BLOND_DISABLE_NUMBA_KERNELS")
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return wrap

    prange = range


@njit(cache=True, parallel=True)
def multipass_induced_voltage(
    target_centers,
    source_centers,
    source_hists,
    source_bounds,
    source_factors,
    R_S,
    alpha,
    omega_bar,
    voltage,
):
    """Direct double sum of the resonator wake over all source windows.

    Parameters
    ----------
    target_centers
        Bin centers where the voltage is evaluated, ascending, in [s].
    source_centers
        Bin centers of all source windows, concatenated; ascending
        within each window, in [s].
    source_hists
        Histograms of all source windows, concatenated (same layout as
        `source_centers`).
    source_bounds
        Start index of each source window in the concatenated arrays,
        with a trailing total length (length ``n_windows + 1``).
    source_factors
        Charge per histogram count of each source window, in [C].
    R_S
        Shunt impedances of all resonators, in [Ohm].
    alpha
        Decay rates ``omega_R / (2 Q)`` of all resonators, in [1/s].
    omega_bar
        Oscillation frequencies ``sqrt(omega_R^2 - alpha^2)`` of all
        resonators, in [rad/s].
    voltage
        Output induced voltage per target bin, in [V]. Overwritten.
    """
    n_targets = len(target_centers)
    n_windows = len(source_bounds) - 1
    n_resonators = len(R_S)

    for m in prange(n_targets):
        t_m = target_centers[m]
        acc = 0.0
        for j in range(n_windows):
            start = source_bounds[j]
            stop = source_bounds[j + 1]
            # whole window after the target bin: no causal contribution
            if source_centers[start] > t_m:
                continue
            factor = source_factors[j]
            for k in range(start, stop):
                t = t_m - source_centers[k]
                if t < 0.0:
                    break  # centers ascending: later bins are non-causal
                q = source_hists[k]
                if q == 0.0:
                    continue
                # (sign(t) + 1): half weight exactly at t = 0
                weight = 1.0 if t == 0.0 else 2.0
                wake = 0.0
                for r in range(n_resonators):
                    wake += (
                        R_S[r]
                        * alpha[r]
                        * np.exp(-alpha[r] * t)
                        * (
                            np.cos(omega_bar[r] * t)
                            - alpha[r]
                            / omega_bar[r]
                            * np.sin(omega_bar[r] * t)
                        )
                    )
                acc += factor * q * weight * wake
        voltage[m] = acc
