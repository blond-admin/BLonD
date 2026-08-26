# coding: utf8
# Copyright 2014-2026 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Compiled kernels for the LHC/FCC cavity-loop coarse-grid recursion and the
ACS cavity response.**

The per-sample recursions are identical to the pure-Python implementations in
:mod:`blond.llrf.cavity_feedback` and :mod:`blond.llrf.impulse_response`;
numba only removes the interpreter overhead. Set the environment variable
``BLOND_DISABLE_NUMBA_KERNELS`` to any non-empty value to force the original
pure-Python/scipy code paths (e.g. for A/B validation).

:Authors: **Lina Valle**
Co-Authored-By: Claude Sonnet 5 noreply@anthropic.com
"""

import os

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = not os.environ.get("BLOND_DISABLE_NUMBA_KERNELS")
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True)
def coarse_loop_one_turn(
    n_coarse,
    samples,
    R_over_Q,
    ant_coeff,
    enable_klystron,
    n_delay,
    open_loop,
    ac_coeff,
    alpha,
    go_one_minus_alpha,
    n_otfb,
    fir_coeff,
    open_otfb,
    exc_coeff,
    an_coeff,
    G_a,
    di_decay,
    di_coeff,
    open_rffb,
    clamping,
    v_swap_thres,
    G_gen,
    open_drive,
    drive_offset,
    klystron_fir,
    V_ANT_COARSE,
    I_GEN_COARSE,
    I_BEAM_COARSE,
    V_SET,
    V_FB_IN,
    V_AC_IN,
    V_AN_IN,
    V_AN_OUT,
    V_DI_OUT,
    V_OTFB,
    V_OTFB_INT,
    V_FIR_OUT,
    V_FB_OUT,
    V_SWAP_OUT,
    I_TEST,
    I_GEN_GAIN,
    TUNER_INPUT,
    TUNER_INTEGRATED,
    V_EXC,
):
    r"""One coarse-grid turn of the LHCCavityLoop recursion
    (cavity_response -> rf_feedback -> swap -> generator_current ->
    tuner_input), sample by sample, operating in place on the 2*n_coarse
    state arrays. Expression order matches the pure-Python methods so the
    result is identical to round-off."""

    for i in range(n_coarse):
        ind = i + n_coarse

        # cavity_response
        V_ANT_COARSE[ind] = (
            I_GEN_COARSE[ind - 1] * R_over_Q * samples
            + V_ANT_COARSE[ind - 1] * ant_coeff
            - I_BEAM_COARSE[ind - 1] * 0.5 * R_over_Q * samples
        )

        # rf_feedback
        if enable_klystron:
            V_FB_IN[ind] = V_SET[ind] - open_loop * V_ANT_COARSE[ind]
        else:
            V_FB_IN[ind] = (
                V_SET[ind - n_delay] - open_loop * V_ANT_COARSE[ind - n_delay]
            )
        V_AC_IN[ind] = (
            ac_coeff * V_AC_IN[ind - 1] + V_FB_IN[ind] - V_FB_IN[ind - 1]
        )

        # one_turn_feedback
        V_OTFB_INT[ind] = (
            alpha * V_OTFB_INT[ind - n_coarse]
            + go_one_minus_alpha * V_AC_IN[ind - n_coarse + n_otfb]
        )
        acc = fir_coeff[0] * V_OTFB_INT[ind]
        for k in range(1, len(fir_coeff)):
            acc += fir_coeff[k] * V_OTFB_INT[ind - k]
        V_FIR_OUT[ind] = acc
        V_OTFB[ind] = (
            ac_coeff * V_OTFB[ind - 1] + V_FIR_OUT[ind] - V_FIR_OUT[ind - 1]
        )

        V_AN_IN[ind] = (
            V_FB_IN[ind] + open_otfb * V_OTFB[ind] + exc_coeff * V_EXC[ind]
        )
        V_AN_OUT[ind] = V_AN_OUT[ind - 1] * an_coeff + G_a * (
            V_AN_IN[ind] - V_AN_IN[ind - 1]
        )
        V_DI_OUT[ind] = (
            V_DI_OUT[ind - 1] * di_decay + di_coeff * V_FB_IN[ind - 1]
        )
        V_FB_OUT[ind] = open_rffb * (V_AN_OUT[ind] + V_DI_OUT[ind])

        # swap (smooth_step with N=0 reduces to a clamp of |V|/threshold)
        if clamping:
            x = abs(V_FB_OUT[ind]) / v_swap_thres
            if x > 1.0:
                x = 1.0
            V_SWAP_OUT[ind] = (
                v_swap_thres * x * np.exp(1j * np.angle(V_FB_OUT[ind]))
            )
        else:
            V_SWAP_OUT[ind] = V_FB_OUT[ind]

        # generator_current
        I_TEST[ind] = G_gen * V_SWAP_OUT[ind]
        I_GEN_GAIN[ind] = open_drive * I_TEST[ind] + drive_offset
        if enable_klystron:
            acc2 = klystron_fir[0] * I_GEN_GAIN[ind]
            for k in range(1, len(klystron_fir)):
                acc2 += klystron_fir[k] * I_GEN_GAIN[ind - k]
            I_GEN_COARSE[ind] = acc2
        else:
            I_GEN_COARSE[ind] = I_GEN_GAIN[ind]

        # tuner_input
        TUNER_INPUT[ind] = I_GEN_COARSE[ind] * np.conj(V_ANT_COARSE[ind])
        TUNER_INTEGRATED[ind] = (
            (1 / 64)
            * (
                TUNER_INPUT[ind]
                - 2 * TUNER_INPUT[ind - 8]
                + TUNER_INPUT[ind - 16]
            )
            + 2 * TUNER_INTEGRATED[ind - 1]
            - TUNER_INTEGRATED[ind - 2]
        )


@njit(cache=True)
def cavity_response_forward(b, B):
    r"""Forward substitution for the bidiagonal ACS cavity-response system:
    V[0] = b[0], V[n] = B * V[n-1] + b[n]. Mathematically identical to
    spsolve on the lower-bidiagonal B_matrix of
    :func:`blond.llrf.impulse_response.cavity_response_sparse_matrix`."""

    V = np.empty_like(b)
    V[0] = b[0]
    for n in range(1, len(b)):
        V[n] = B * V[n - 1] + b[n]
    return V
