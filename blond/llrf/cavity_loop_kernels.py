# coding: utf8
# Copyright 2014-2026 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Compiled kernels for the LHC/FCC cavity-loop coarse-grid recursion, the
ACS cavity response and the RF beam current.**

The per-sample recursions are identical to the pure-Python implementations in
:mod:`blond.llrf.cavity_feedback`, :mod:`blond.llrf.impulse_response` and
:mod:`blond.llrf.signal_processing`;
numba only removes the interpreter overhead. Set the environment variable
``BLOND_DISABLE_NUMBA_KERNELS`` to any non-empty value to force the original
pure-Python/scipy code paths (e.g. for A/B validation).

The exception is :func:`cavity_response_no_beam_gap`, which both code paths
use to carry the fine-grid antenna voltage of a sparse profile from one
window to the next without sampling the gap in between.

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


@njit(cache=True)
def cavity_response_n_steps(B, n):
    r"""Coefficients of n steps of the ACS cavity-response recursion
    V[j+1] = B * V[j] + d[j] for a drive that is linear in the sample number,
    d[j] = d[0] + slope * j:

    V[n] = V[0] + excess * V[0] + unit * d[0] + ramp * slope,

    with excess = B^n - 1, unit = sum_{j<n} B^(n-1-j) and
    ramp = sum_{j<n} j * B^(n-1-j).

    Two consecutive blocks of n1 and n2 steps combine as
    excess = excess1 + excess2 + excess1 * excess2,
    unit = unit1 + unit2 + excess2 * unit1 and
    ramp = ramp1 + ramp2 + excess2 * ramp1 + n1 * unit2, so the coefficients
    are built by doubling in O(log n) operations.

    B is close to 1, which makes the usual expressions lose digits: the
    closed forms of these geometric sums divide by the small 1 - B, and the
    round-off of B^n grows like n when B is squared repeatedly. Doubling the
    excess B^n - 1 instead (B - 1 is exact in floating point) has neither
    problem, and the coefficients are accurate to round-off."""

    excess = 0.0j
    unit = 0.0j
    ramp = 0.0j
    steps = 0
    block_excess = B - 1.0
    block_unit = 1.0 + 0.0j
    block_ramp = 0.0j
    block_steps = 1
    while n > 0:
        if n & 1:
            ramp = ramp + block_ramp + block_excess * ramp + steps * block_unit
            unit = unit + block_unit + block_excess * unit
            excess = excess + block_excess + block_excess * excess
            steps += block_steps
        block_ramp = (
            2.0 * block_ramp
            + block_excess * block_ramp
            + block_steps * block_unit
        )
        block_unit = 2.0 * block_unit + block_excess * block_unit
        block_excess = 2.0 * block_excess + block_excess * block_excess
        block_steps += block_steps
        n >>= 1
    return excess, unit, ramp


@njit(cache=True)
def cavity_response_no_beam_gap(
    V,
    I_gen_init,
    I_beam_init,
    n_samples,
    t_init,
    bin_size,
    coarse_time,
    I_gen_coarse,
    A,
    B,
):
    r"""Advance the ACS cavity-response recursion
    V[j+1] = B * V[j] + A * (2 * I_gen[j] - I_beam[j]) through a gap of
    n_samples samples without beam, the generator current being interpolated
    linearly from the coarse grid as np.interp does. V, I_gen_init and
    I_beam_init are the values at the last sample before the gap, at time
    t_init. Returns the antenna voltage and the generator current at the
    last sample of the gap, without computing the samples in between: the
    recursion is summed analytically over each coarse-grid interval, on
    which the drive is linear."""

    # The first sample of the gap is driven by the currents before the gap
    V = B * V + A * (2 * I_gen_init - I_beam_init)

    # Sample j+1 is driven by the generator current at sample j, for
    # j = 1 ... n_samples-1; "knot" is the first coarse-grid sample after
    # sample "first"
    n_knots = len(coarse_time)
    knot = np.searchsorted(coarse_time, t_init + bin_size, side="right")
    first = 1
    n_cached = 0
    excess = 0.0j
    unit = 0.0j
    ramp = 0.0j
    while first < n_samples:
        # Interpolated generator current I_gen_0 + I_gen_slope * (t - t_0)
        # up to the next coarse-grid sample, constant outside the coarse grid
        if knot == 0:
            stop = int(np.ceil((coarse_time[0] - t_init) / bin_size))
            t_0 = coarse_time[0]
            I_gen_0 = I_gen_coarse[0]
            I_gen_slope = 0.0j
        elif knot == n_knots:
            stop = n_samples
            t_0 = coarse_time[-1]
            I_gen_0 = I_gen_coarse[-1]
            I_gen_slope = 0.0j
        else:
            stop = int(np.ceil((coarse_time[knot] - t_init) / bin_size))
            t_0 = coarse_time[knot - 1]
            I_gen_0 = I_gen_coarse[knot - 1]
            I_gen_slope = (I_gen_coarse[knot] - I_gen_0) / (
                coarse_time[knot] - t_0
            )
        knot += 1
        stop = min(stop, n_samples)
        n = stop - first
        if n < 1:
            continue
        if n != n_cached:
            excess, unit, ramp = cavity_response_n_steps(B, n)
            n_cached = n
        drive = (
            2 * A * (I_gen_slope * (t_init + first * bin_size - t_0) + I_gen_0)
        )
        drive_slope = 2 * A * (I_gen_slope * bin_size)
        V = V + (excess * V + unit * drive + ramp * drive_slope)
        first = stop

    I_gen_end = np.interp(
        t_init + n_samples * bin_size, coarse_time, I_gen_coarse
    )
    return V, I_gen_end


@njit(cache=True)
def cavity_response_sparse_windows(
    V_ANT_FINE,
    I_BEAM_FINE,
    I_GEN_FINE,
    t_first,
    t_last,
    order,
    n_slices,
    bin_size,
    coarse_time,
    I_gen_coarse,
    V_ant_init,
    I_gen_init,
    A,
    B,
):
    r"""ACS cavity response on the fine grid of a sparse profile: the
    recursion V[j+1] = B * V[j] + A * (2 * I_gen[j] - I_beam[j]) over the
    windows of n_slices samples taken in time order, bridged by
    :func:`cavity_response_no_beam_gap` where they are not adjacent.
    Window p spans t_first[p] to t_last[p], uses I_BEAM_FINE[p * n_slices :
    (p + 1) * n_slices] and the elements one further in I_GEN_FINE, and
    fills the same elements of V_ANT_FINE as I_GEN_FINE; element 0 of these
    is one sample before window 0. V_ant_init and I_gen_init are the values
    one sample before the earliest window."""

    V = V_ant_init
    I_gen = I_gen_init
    I_beam = 0.0j
    for k in range(len(order)):
        p = order[k]
        if k > 0:
            t_init = t_last[order[k - 1]]
            # Number of fine bins strictly between the two windows
            n_gap = int(np.rint((t_first[p] - t_init) / bin_size)) - 1
            if n_gap > 0:
                V, I_gen = cavity_response_no_beam_gap(
                    V,
                    I_gen,
                    I_beam,
                    n_gap,
                    t_init,
                    bin_size,
                    coarse_time,
                    I_gen_coarse,
                    A,
                    B,
                )
                I_beam = 0.0j
        if p == 0:
            V_ANT_FINE[0] = V
        for n in range(p * n_slices, (p + 1) * n_slices):
            V = B * V + A * (2 * I_gen - I_beam)
            V_ANT_FINE[n + 1] = V
            I_gen = I_GEN_FINE[n + 1]
            I_beam = I_BEAM_FINE[n]


@njit(cache=True)
def rf_beam_charge(n_macroparticles, bin_centers, charge, omega_c):
    r"""RF beam charge on the fine grid: the charge of each bin, charge per
    macro-particle times n_macroparticles, demodulated at omega_c (factor 2
    included). Same expressions, bin by bin, as the numpy code of
    :func:`blond.llrf.signal_processing.rf_beam_current`, in a single pass."""

    charges_fine = np.empty(len(bin_centers), dtype=np.complex128)
    for i in range(len(bin_centers)):
        charges = charge * n_macroparticles[i]
        phase = omega_c * bin_centers[i]
        charges_fine[i] = complex(
            2.0 * charges * np.cos(phase), -2.0 * charges * np.sin(phase)
        )
    return charges_fine


@njit(cache=True)
def charges_from_fine_to_coarse(
    charges_fine, bin_centers, dT, half_period, T_s, n_points
):
    r"""Sum the fine-grid RF beam charge onto the coarse grid: the bin at
    time t goes to the coarse sample round((t - dT - half_period) / T_s),
    modulo n_points. Same result as the np.bincount scatter-add of
    :func:`blond.llrf.signal_processing.charges_from_fine_to_coarse`."""

    charges_coarse = np.zeros(n_points, dtype=np.complex128)
    for i in range(len(charges_fine)):
        ind_fine = (bin_centers[i] - dT - half_period) / T_s
        charges_coarse[int(np.rint(ind_fine)) % n_points] += charges_fine[i]
    return charges_coarse


@njit(cache=True)
def interp(x, xp, fp):
    r"""np.interp, compiled"""

    return np.interp(x, xp, fp)
