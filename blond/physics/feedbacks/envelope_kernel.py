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

The envelope ODE is linear, so the recursion is run as TWO independent state
components through the same propagator -- superposition is exact:

- the *beam-sourced* component, driven by ``-I_beam / 2`` alone (the former
  single state with the generator current pinned to zero; bit-identical to
  it for an undriven feedback), anchored to the demodulation frame;
- the *generator-sourced* component, driven by ``I_gen`` alone, natively
  anchored to the piecewise design clock the coarse grid samples.

Each cell also composes the demodulation-frame sum
``V = V_beam + V_gen * generator_frame_rotation[c]`` (the rotation
``exp(-i (delta_phi_rf + carrier slip gap + accumulated phase))`` of cell
``c`` takes the design-anchored component into the demodulation frame; see
``IQCavityFeedbackTimingClass._update_frame_rotations``), which is what the
PI regulates -- in the *kick frame*,
``error = (V_set - V * kick_frame_rotation[c]) * pi_error_frame_rotation``.
Both rotations are per cell because a backfill span replays an interval
over which the accumulated phase is still growing; over a forward span
every entry holds the same per-passage value.

The kernel does not evaluate the propagator itself: the per-cell voltage
multiplier ``B = e^L`` and drive weight ``W = (e^L - 1) / L`` of the exact
exponential step depend only on the step size and detuning, not on the
recursion state, so they are precomputed on the host (see
``_circuit_track_cells_kernel``) and passed in, as are the per-cell frame
rotations. The kernel then only carries the state-dependent update
``V = V_prev * B + drive * W`` and the PI controller, which keeps it
identical -- byte-for-byte on complex128 -- to the Python reference path
without numba ever evaluating ``exp``/``expm1``. The derivation of ``B`` and
``W``, and why the retired forward-Euler ``B = 1 + L``, ``W = 1`` was only
their first-order truncation, is in the Notes of
``IQCavityFeedbackTimingClass._advance_coarse_voltage``.

The PI delay line is passed as a circular buffer (``delay_buffer`` + a head
index) rather than a :class:`collections.deque`; ``_circuit_track_cells_kernel``
marshals the deque in and out. Writing ``error`` at the head, advancing the
head and then reading the (new) head reproduces ``deque.append``/``[0]``
exactly, including the ``n_delay``-sample delay.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np


def inactive_controller_scan_state() -> tuple:
    """
    Neutral control arguments for a span with no regulation.

    A span with no controller attached still runs the same scan, with
    ``controller_active`` False. These placeholders satisfy the kernel's
    signature and are never read, so the generator current simply stays
    constant. (With a controller attached it runs on every span, the
    no-beam backfill segments included.)

    Returns
    -------
    state
        Control arguments in the order :func:`envelope_pi_scan` expects them.
    """
    return (
        0.0,
        0.0,
        0.0 + 0.0j,
        np.zeros(1, dtype=np.complex128),
        0,
        0.0 + 0.0j,
        np.inf,
    )


@nb.njit(cache=True)  # pragma: no cover
def envelope_pi_scan(
    voltage_multiplier,
    drive_weight,
    omega_times_dt,
    beam_current,
    voltage_gen_out,
    voltage_beam_out,
    voltage_out,
    generator_current_out,
    voltage_gen_init,
    voltage_beam_init,
    generator_current_init,
    r_over_q,
    generator_frame_rotation,
    kick_frame_rotation,
    pi_error_frame_rotation,
    controller_active,
    controller_update_interval,
    controller_update_phase,
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

    Advances, for each coarse cell ``c``, the two source-split components of
    the (linear) envelope ODE through the same propagator,

    .. math::
        V_{\mathrm{beam},c} = V_{\mathrm{beam},c-1}\,B_c
            + (R/Q)\,\omega\Delta t_c\,
              (0 - \tfrac12 I_{\mathrm{beam},c})\,W_c, \quad
        V_{\mathrm{gen},c} = V_{\mathrm{gen},c-1}\,B_c
            + (R/Q)\,\omega\Delta t_c\,
              I_{\mathrm{gen},c-1}\,W_c,

    composes the demodulation-frame sum
    ``V_c = V_beam,c + V_gen,c * generator_frame_rotation[c]`` and, when
    ``controller_active``, updates the generator current from the kick-frame
    antenna-voltage error ``V_set - V_c * kick_frame_rotation[c]`` with a
    saturating PI controller (conditional anti-windup, magnitude clamp).
    ``max_output = inf`` disables the clamp and the saturation check,
    matching an unlimited controller.

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
    voltage_gen_out
        Output generator-sourced antenna voltage, written in place
        (complex128, length ``N``).
    voltage_beam_out
        Output beam-sourced antenna voltage, written in place (complex128,
        length ``N``).
    voltage_out
        Output demodulation-frame sum, written in place (complex128,
        length ``N``).
    generator_current_out
        Generator current (complex128, length ``N``), in/out: pre-filled by the
        caller with the current generator grid (the drive source for the
        inactive path and for cell ``c``'s read of cell ``c-1``); when
        ``controller_active`` each cell's PI output is written over it.
    voltage_gen_init
        Generator-sourced voltage seeding the first cell.
    voltage_beam_init
        Beam-sourced voltage seeding the first cell.
    generator_current_init
        Generator current driving the first cell (the carried ``last_val`` at a
        segment starting at grid index 0, else the previous grid cell).
    r_over_q
        Cavity ``R/Q`` [Ohm].
    generator_frame_rotation
        Per-cell rotation ``exp(-i (delta_phi_rf + carrier slip gap))``
        (complex128, length ``N``) taking the design-anchored generator
        component of that cell into the demodulation frame of the beam
        component; on a backfill cell the carrier slip gap carries the
        phase accumulated up to it (unity without an RF-frequency offset
        and without multi-section acceleration).
    kick_frame_rotation
        Per-cell rotation ``exp(+i * carrier slip gap)`` (complex128,
        length ``N``), with the same gap as ``generator_frame_rotation``,
        taking the demodulation-frame sum of that cell into the frame of
        the applied kick, in which the PI error is formed.
    pi_error_frame_rotation
        Per-passage scalar ``exp(+i * delta_phi_rf)`` rotating the kick-frame
        error into the actuator (design) frame the generator current acts in,
        cancelling the ``exp(-i * delta_phi_rf)`` the composition applies to
        the generator component (unity without an RF-frequency offset).
    controller_active
        Whether to run the PI controller; if False each cell's drive uses the
        pre-filled generator grid (cell 0 uses ``generator_current_init``) and
        the grid is left unchanged.
    controller_update_interval
        Coarse cells between controller updates. 1 regulates every cell;
        ``x`` samples the loop ``x`` times slower than the cavity model
        steps and holds the command (zero order) over the cells in
        between, which is what a digital LLRF does between its own
        samples. The cavity recursion itself always steps every cell.
    controller_update_phase
        Phase of this span's first cell on that update clock: cell ``c``
        updates when ``(c + phase) % interval == 0``. The caller carries
        it across spans and turns, so the clock free-runs rather than
        re-phasing at every segment boundary.
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
    delay_buffer
        The delay buffer after the span, returned so the caller can hand the
        controller its state back without knowing the buffer's layout.
    delay_head
        The head index after the span.
    integral
        The committed error integral after the span.
    """
    n = omega_times_dt.shape[0]
    buffer_len = delay_buffer.shape[0]
    voltage_gen_prev = voltage_gen_init
    voltage_beam_prev = voltage_beam_init
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
        # Beam-sourced component: the former recursion with the generator
        # current pinned to (0 + 0j) -- bit-identical to the single state
        # for an undriven feedback (whose generator grid is exactly zero).
        drive_beam = (
            r_over_q
            * omega_times_dt[cell]
            * ((0.0 + 0.0j) - 0.5 * beam_current[cell])
        )
        voltage_beam = voltage_beam_prev * voltage_multiplier[cell] + (
            drive_beam * drive_weight[cell]
        )
        voltage_beam_out[cell] = voltage_beam
        voltage_beam_prev = voltage_beam
        # Generator-sourced component: same propagator, beam current
        # pinned to (0 + 0j). With nothing driving the generator this
        # component stays exactly zero and the composition below adds an
        # exact zero, which is why an undriven feedback needs no special
        # case to stay bit-identical to the former single state.
        drive_gen = (
            r_over_q
            * omega_times_dt[cell]
            * (generator_current_drive - 0.5 * (0.0 + 0.0j))
        )
        voltage_gen = voltage_gen_prev * voltage_multiplier[cell] + (
            drive_gen * drive_weight[cell]
        )
        voltage_gen_out[cell] = voltage_gen
        voltage_gen_prev = voltage_gen
        voltage = voltage_beam + voltage_gen * generator_frame_rotation[cell]
        voltage_out[cell] = voltage
        if not controller_active:
            continue
        if (cell + controller_update_phase) % controller_update_interval != 0:
            # Between samples the loop holds its last command (zero order),
            # so this cell drives the cavity with the command the previous
            # one carried -- the value the next cell reads back as its own
            # drive. Writing it keeps the grid, and the fine-grid solve
            # that interpolates it, a faithful staircase.
            generator_current_out[cell] = generator_current_drive
        else:
            error = (
                pi_setpoint - voltage * kick_frame_rotation[cell]
            ) * pi_error_frame_rotation
            # The command is held for the whole update interval, so the
            # integrator credits it with that much time. Exact inside a
            # segment, where the coarse steps are uniform.
            delta_t = (
                omega_times_dt[cell] / omega_input * controller_update_interval
            )
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
            if magnitude > max_output:
                # Saturated: freeze the integral (anti-windup) and clamp.
                generator_current_out[cell] = output * (max_output / magnitude)
            else:
                integral = candidate_integral
                generator_current_out[cell] = output
        # No controller: generator_current_out[cell] keeps its pre-filled
        # static grid value -- the constant-current / no-beam path never
        # rewrites it.
    return delay_buffer, delay_head, integral
