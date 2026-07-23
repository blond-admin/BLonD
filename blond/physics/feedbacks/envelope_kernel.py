# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Numba host kernel for the coarse-grid cavity-envelope recursion.

The per-cell antenna-voltage recursion of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
is inherently sequential (each cell reads the previous cell's voltage *and*
generator current, and -- with an active PI controller -- the generator current
of a cell depends on the voltage just computed for it). It therefore cannot be
vectorised, and in pure Python the ~10^5 per-turn cells are dominated by
interpreter and method-call overhead. :func:`envelope_pi_scan` compiles the
whole scan to a single numba call.

The kernel is deliberately *solver-agnostic*: the per-cell voltage multiplier
``B`` (``1 + L`` for forward Euler, ``e^L`` for the exponential propagator) and
the drive weight ``W`` (``1`` or ``(e^L - 1) / L``) depend only on the step
size and detuning, not on the recursion state, so they are precomputed on the
host (see ``_circuit_track_cells_kernel``) and passed in. The kernel then only
carries the state-dependent update ``V = V_prev * B + drive * W`` and the PI
controller, which keeps it identical -- byte-for-byte on complex128 -- to both
the Euler and the exponential Python paths without numba ever evaluating
``exp``/``expm1``.

The PI delay line is passed as a circular buffer (``delay_buffer`` + a head
index) rather than a :class:`collections.deque`; ``_circuit_track_cells_kernel``
marshals the deque in and out. Writing ``error`` at the head, advancing the
head and then reading the (new) head reproduces ``deque.append``/``[0]``
exactly, including the ``n_delay``-sample delay.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np


@nb.njit(cache=True)  # pragma: no cover
def envelope_pi_scan(
    voltage_multiplier,
    drive_weight,
    omega_times_dt,
    beam_current,
    voltage_out,
    generator_current_out,
    voltage_init,
    generator_current_init,
    r_over_q,
    controller_active,
    pi_setpoint,
    omega_input,
    gain_proportional,
    gain_integral,
    generator_current_bias,
    delay_buffer,
    delay_head,
    integral,
    max_output,
):
    r"""
    Run the coarse-grid antenna-voltage recursion (+ optional PI) over a span.

    Advances, for each coarse cell ``c``,

    .. math::
        V_c = V_{c-1}\,B_c + \mathrm{drive}_c\,W_c, \quad
        \mathrm{drive}_c = (R/Q)\,\omega\Delta t_c\,
            (I_{\mathrm{gen},c-1} - \tfrac12 I_{\mathrm{beam},c}),

    and, when ``controller_active``, updates the generator current from the
    antenna-voltage error with a saturating PI controller (conditional
    anti-windup, magnitude clamp). ``max_output = inf`` disables the clamp and
    the saturation check, matching an unlimited controller.

    Parameters
    ----------
    voltage_multiplier
        Per-cell voltage multiplier ``B`` (complex128, length ``N``).
    drive_weight
        Per-cell drive weight ``W`` (complex128, length ``N``).
    omega_times_dt
        Per-cell ``omega * dt`` (float64, length ``N``).
    beam_current
        Per-cell beam current (complex128, length ``N``); zero for a no-beam
        segment.
    voltage_out
        Output antenna voltage, written in place (complex128, length ``N``).
    generator_current_out
        Generator current (complex128, length ``N``), in/out: pre-filled by the
        caller with the current generator grid (the drive source for the
        inactive path and for cell ``c``'s read of cell ``c-1``); when
        ``controller_active`` each cell's PI output is written over it.
    voltage_init
        Antenna voltage seeding the first cell (previous cell's value).
    generator_current_init
        Generator current driving the first cell (the carried ``last_val`` at a
        segment starting at grid index 0, else the previous grid cell).
    r_over_q
        Cavity ``R/Q`` [Ohm].
    controller_active
        Whether to run the PI controller; if False each cell's drive uses the
        pre-filled generator grid (cell 0 uses ``generator_current_init``) and
        the grid is left unchanged.
    pi_setpoint
        PI voltage setpoint in the IQ frame.
    omega_input
        Segment angular frequency, used to recover ``dt = omega*dt / omega``.
    gain_proportional
        PI proportional gain.
    gain_integral
        PI integral gain.
    generator_current_bias
        PI generator-current bias ``I_0``.
    delay_buffer
        Circular buffer of the ``n_delay + 1`` most recent errors
        (complex128), modified in place.
    delay_head
        Current head index into ``delay_buffer``.
    integral
        Committed error integral entering the span.
    max_output
        Klystron current-magnitude limit, or ``inf`` for no limit.

    Returns
    -------
    delay_head
        The head index after the span.
    integral
        The committed error integral after the span.
    saturation_possible
        True if any cell's generator current reached (within a guard band) the
        klystron limit. numpy's scalar complex ``abs`` used by the reference
        clamp cannot be reproduced bit-for-bit in numba, so the caller reruns
        such (rare) segments on the exact Python path. When no cell nears the
        limit the clamp is never applied and the kernel output is identical.
    """
    n = omega_times_dt.shape[0]
    buffer_len = delay_buffer.shape[0]
    voltage_prev = voltage_init
    # Guard band below the limit: comfortably wider than a double-precision ULP
    # (~2e-16) yet negligible physically, so it can never miss a cell the
    # reference would clamp while almost never firing spuriously.
    saturation_guard = max_output * (1.0 - 1.0e-9)
    saturation_possible = False
    for cell in range(n):
        # Drive current: the carried value for the first cell, otherwise the
        # generator current currently in the grid at the previous cell -- the PI
        # output the controller just wrote (active), or the untouched static
        # grid value the caller pre-filled (inactive / constant current / no
        # beam). This mirrors cavity_response, which reads
        # generator_current_coarse_grid[idx-1] for idx>=1 and only uses the
        # carried last_val at idx==0.
        if cell == 0:
            generator_current_drive = generator_current_init
        else:
            generator_current_drive = generator_current_out[cell - 1]
        drive = (
            r_over_q
            * omega_times_dt[cell]
            * (generator_current_drive - 0.5 * beam_current[cell])
        )
        voltage = voltage_prev * voltage_multiplier[cell] + (
            drive * drive_weight[cell]
        )
        voltage_out[cell] = voltage
        if controller_active:
            error = pi_setpoint - voltage
            delta_t = omega_times_dt[cell] / omega_input
            delay_buffer[delay_head] = error
            delay_head = (delay_head + 1) % buffer_len
            delayed_error = delay_buffer[delay_head]
            candidate_integral = integral + delayed_error * delta_t
            output = (
                generator_current_bias
                + gain_proportional * delayed_error
                + gain_integral * candidate_integral
            )
            magnitude = np.abs(output)
            if magnitude > saturation_guard:
                # At/near the klystron limit: the reference clamp's numpy
                # magnitude cannot be reproduced bit-for-bit here, so flag the
                # segment for the exact Python rerun. The clamp below still runs
                # to keep the recursion self-consistent, but its result is
                # discarded by the caller when this flag is set.
                saturation_possible = True
            if magnitude > max_output:
                # Saturated: freeze the integral (anti-windup) and clamp.
                generator_current_out[cell] = output * (max_output / magnitude)
            else:
                integral = candidate_integral
                generator_current_out[cell] = output
        # Inactive: leave generator_current_out[cell] at its pre-filled static
        # grid value -- the constant-current / no-beam path never rewrites it.
        voltage_prev = voltage
    return delay_head, integral, saturation_possible
