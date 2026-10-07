# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Numba kernels of the generator-current control laws, closed around the cavity.

The cavity model lives in :mod:`~blond.physics.feedbacks.envelope_kernel`
and knows no controller. This module holds what a digital LLRF adds on top
of it, in three layers kept apart on purpose:

- the **measurement**, law-independent: :func:`regulation_error` forms the
  error in the kick frame and rotates it into the actuator frame, and
  :func:`delay_line_push` delays it by the loop latency;
- the **laws**, one compiled function each and nothing shared between
  them: :func:`pi_law_step` (proportional-integral, conditional or
  directional anti-windup, magnitude clamp) and :func:`p_law_step`
  (proportional, magnitude clamp, no state);
- the **feedforward**, law-independent and computed elsewhere: a per-cell
  addition to the setpoint, which enters the measurement, and a per-cell
  addition to the bias, which a scan hands its law *as* the bias of that
  sample, so every law clamps the sum without knowing the term exists;
- the **klystron**, law-independent and optional: :func:`klystron_cell`
  relaxes the generator current towards the held command through one
  real pole on every cell, so the cavity is driven by what a klystron of
  finite bandwidth delivers rather than by the command itself;
- the **closed-loop scans**, one per law: :func:`envelope_pi_scan` and
  :func:`envelope_p_scan` step the cavity model cell by cell through
  :func:`~blond.physics.feedbacks.envelope_kernel.propagate_envelope_cell`,
  sample the loop every ``controller_update_interval`` cells, hold the
  command in between (zero order), and call their own law on each sample.
  With a klystron pole the generator grid carries the klystron's output
  and the held command travels as law state (``klystron_command``);
  without one (``klystron_time_constant`` zero) the grid is the held
  command, exactly as before the pole existed.

The two scans share their first positional arguments -- the cavity and the
sampler -- and differ only in the law state that follows, so the feedback
passes a controller's
:meth:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController.envelope_scan_state`
straight through and hands whatever the scan returns back to
:meth:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentController.absorb_envelope_scan_state`
without looking inside either. Each scan reproduces its controller's
``update_generator_current`` byte-for-byte on complex128 while the clamp is
idle; once it fires, numba's complex ``abs`` may differ from numpy's by an
ULP.

The scans' loop skeleton -- drive selection, the sample clock, the hold --
is written twice, because numba cannot take the law as an argument and
still cache the compiled scan. The cavity physics is not: both call the one
cavity model, and a test pins that both reduce to its open-loop scan
exactly when their gains are zero.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np

from blond.physics.feedbacks.envelope_kernel import propagate_envelope_cell


@nb.njit(cache=True)  # pragma: no cover
def regulation_error(
    voltage,
    voltage_setpoint,
    setpoint_feedforward,
    kick_frame_rotation,
    error_frame_rotation,
):
    """
    Voltage error of one sample, in the frame the generator current acts in.

    Formed in the kick frame -- the voltage the station actually applies --
    against the setpoint plus its per-cell feedforward, then rotated into
    the actuator frame so the open-loop gain stays real under an RF
    frequency or phase-loop offset.

    Parameters
    ----------
    voltage
        Demodulation-frame antenna voltage of the sampled cell.
    voltage_setpoint
        Voltage setpoint in the IQ frame.
    setpoint_feedforward
        Per-cell addition to the setpoint [V]; zero is an exact no-op.
    kick_frame_rotation
        Rotation of the demodulation-frame voltage into the kick frame.
    error_frame_rotation
        Rotation of the kick-frame error into the actuator frame.

    Returns
    -------
    error
        ``(V_set + feedforward - V * kick) * error_frame``.
    """
    return (
        voltage_setpoint + setpoint_feedforward - voltage * kick_frame_rotation
    ) * error_frame_rotation


@nb.njit(cache=True)  # pragma: no cover
def delay_line_push(delay_buffer, delay_head, error):
    """
    Push one error into the loop's delay line and read the delayed one.

    Writing at the head, advancing the head and then reading the new head
    reproduces ``deque.append`` followed by ``[0]``: with a buffer of
    ``n_delay + 1`` slots the value read is the error from ``n_delay``
    samples ago.

    Parameters
    ----------
    delay_buffer
        Circular buffer of the ``n_delay + 1`` most recent errors,
        modified in place.
    delay_head
        Current head index; its slot holds the oldest error.
    error
        Error of this sample.

    Returns
    -------
    delayed_error, delay_head
        The error from ``n_delay`` samples ago and the advanced head.
    """
    delay_buffer[delay_head] = error
    delay_head = (delay_head + 1) % delay_buffer.shape[0]
    return delay_buffer[delay_head], delay_head


@nb.njit(cache=True)  # pragma: no cover
def klystron_cell(previous_output, command, delta_t, time_constant):
    """
    Advance the klystron's one-pole output by one coarse cell.

    The output relaxes exactly towards the command held over the cell,
    ``u + (y - u) exp(-dt / tau)``, and the result drives the NEXT cell,
    as a command does without the pole. Holding it over that cell instead
    of averaging the exponential over it lags the continuous pole by half
    a cell, an RF period at most against a klystron time constant of tens
    of them. The compiled twin of
    :func:`~blond.physics.feedbacks.generator_current_controller.klystron_relax`.

    Parameters
    ----------
    previous_output
        Klystron output at the end of the previous cell [A].
    command
        Command held over this cell [A].
    delta_t
        Length of this cell [s].
    time_constant
        The pole's time constant [s], positive.

    Returns
    -------
    output
        Klystron output at the end of this cell [A].
    """
    return command + (previous_output - command) * np.exp(
        -delta_t / time_constant
    )


#: Integer code of the directional anti-windup, its index in
#: :data:`~blond.physics.feedbacks.generator_current_controller.ANTI_WINDUP_SCHEMES`.
DIRECTIONAL_ANTI_WINDUP = 1


@nb.njit(cache=True)  # pragma: no cover
def pi_law_step(
    delayed_error,
    delta_t,
    integral,
    gain_proportional,
    gain_integral,
    generator_current_bias,
    max_output,
    anti_windup,
):
    """
    One sample of the saturating PI law.

    ``I = clamp(I_0 + K_p e + K_i (integral + e dt))``. The whole
    increment ``e dt`` is committed only while the output is not clamped;
    on a clamped sample the conditional anti-windup (code 0) commits none
    of it, the directional one (:data:`DIRECTIONAL_ANTI_WINDUP`) all but
    its outward part. The compiled twin of
    ``GeneratorCurrentPIController.update_generator_current``.

    Parameters
    ----------
    delayed_error
        Error the law acts on this sample (already delayed).
    delta_t
        Time the command will be held for [s].
    integral
        Committed error integral before this sample.
    gain_proportional
        Proportional gain ``K_p`` [A/V].
    gain_integral
        Integral gain ``K_i`` [A/(V s)].
    generator_current_bias
        Generator current bias ``I_0`` [A].
    max_output
        Klystron current-magnitude limit, or ``inf`` for none.
    anti_windup
        Anti-windup code, the scheme's index in ``ANTI_WINDUP_SCHEMES``.

    Returns
    -------
    generator_current, integral
        The (clamped) command and the committed integral after it.
    """
    increment = delayed_error * delta_t
    candidate_integral = integral + increment
    output = (
        generator_current_bias
        + gain_proportional * delayed_error
        + gain_integral * candidate_integral
    )
    magnitude = np.abs(output)
    if magnitude > max_output:
        if anti_windup == DIRECTIONAL_ANTI_WINDUP:
            # The clamp leaves the phase free: drop only the part of the
            # increment that would push the command further out.
            unit = output / magnitude
            radial = (increment * np.conj(unit)).real
            if gain_integral * radial > 0.0:
                increment = increment - radial * unit
            return output * (max_output / magnitude), integral + increment
        # Saturated: freeze the integral (anti-windup) and clamp.
        return output * (max_output / magnitude), integral
    return output, candidate_integral


@nb.njit(cache=True)  # pragma: no cover
def p_law_step(
    delayed_error, gain_proportional, generator_current_bias, max_output
):
    """
    One sample of the saturating proportional law.

    ``I = clamp(I_0 + K_p e)``. No integral, so no anti-windup and no state:
    the command depends on this sample's delayed error alone. The compiled
    twin of ``GeneratorCurrentPController.update_generator_current``.

    Parameters
    ----------
    delayed_error
        Error the law acts on this sample (already delayed).
    gain_proportional
        Proportional gain ``K_p`` [A/V].
    generator_current_bias
        Generator current bias ``I_0`` [A].
    max_output
        Klystron current-magnitude limit, or ``inf`` for none.

    Returns
    -------
    generator_current
        The (clamped) command.
    """
    output = generator_current_bias + gain_proportional * delayed_error
    magnitude = np.abs(output)
    if magnitude > max_output:
        return output * (max_output / magnitude)
    return output


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
    error_frame_rotation,
    beam_step_rotation,
    controller_update_interval,
    controller_update_phase,
    voltage_setpoint,
    setpoint_feedforward,
    generator_current_feedforward,
    omega_input,
    gain_proportional,
    gain_integral,
    generator_current_bias,
    delay_buffer,
    delay_head,
    integral,
    max_output,
    anti_windup=0,
    klystron_time_constant=0.0,
    klystron_command=0.0 + 0.0j,
):
    """
    The cavity model closed through the PI law, over one span.

    Parameters
    ----------
    voltage_multiplier, drive_weight, omega_times_dt, beam_current
        Per-cell cavity step inputs (see
        :func:`~blond.physics.feedbacks.envelope_kernel.envelope_open_loop_scan`).
    voltage_gen_out, voltage_beam_out, voltage_out
        Output voltages, written in place.
    generator_current_out
        Generator current (complex128, length ``N``), in/out: pre-filled
        with the grid, overwritten with each sample's command and with the
        held command between samples -- or, with a klystron pole, with the
        klystron's output on every cell.
    voltage_gen_init, voltage_beam_init, generator_current_init
        State seeding the first cell.
    r_over_q
        Cavity ``R/Q`` [Ohm].
    generator_frame_rotation, kick_frame_rotation, error_frame_rotation
        Per-cell frame rotations (see :func:`regulation_error`).
    beam_step_rotation
        Per-cell counter-rotation of the carried beam component.
    controller_update_interval
        Cells between controller samples; the command is held between.
    controller_update_phase
        Phase of the first cell on that clock: cell ``c`` samples when
        ``(c + phase) % interval == 0``.
    voltage_setpoint
        Setpoint in the IQ frame.
    setpoint_feedforward
        Per-cell addition to the setpoint [V].
    generator_current_feedforward
        Per-cell addition to the bias [A], read on controller samples:
        the law commands ``clamp(I_0 + I_ff + correction)``, so the clamp
        and the anti-windup act on the sum. Zero is a no-op.
    omega_input
        Segment angular frequency, to recover the sample time.
    gain_proportional, gain_integral, generator_current_bias
        PI tuning.
    delay_buffer, delay_head
        The loop's delay line, advanced in place.
    integral
        Committed error integral entering the span.
    max_output
        Klystron current-magnitude limit, or ``inf``.
    anti_windup
        Anti-windup code (see :func:`pi_law_step`); 0, the conditional
        freeze, if omitted.
    klystron_time_constant
        Time constant of the klystron pole [s]; 0, no pole, if omitted.
    klystron_command
        Command the klystron is chasing as the span starts: the last
        command of the previous span.

    Returns
    -------
    delay_buffer, delay_head, integral, klystron_command
        The PI state after the span, for
        ``GeneratorCurrentPIController.absorb_envelope_scan_state``.
    """
    voltage_gen_previous = voltage_gen_init
    voltage_beam_previous = voltage_beam_init
    for cell in range(omega_times_dt.shape[0]):
        if cell == 0:
            generator_current_drive = generator_current_init
        else:
            generator_current_drive = generator_current_out[cell - 1]
        voltage_beam, voltage_gen, voltage = propagate_envelope_cell(
            voltage_beam_previous,
            voltage_gen_previous,
            generator_current_drive,
            beam_current[cell],
            omega_times_dt[cell],
            voltage_multiplier[cell],
            drive_weight[cell],
            r_over_q,
            beam_step_rotation[cell],
            generator_frame_rotation[cell],
        )
        voltage_beam_out[cell] = voltage_beam
        voltage_gen_out[cell] = voltage_gen
        voltage_out[cell] = voltage
        voltage_beam_previous = voltage_beam
        voltage_gen_previous = voltage_gen
        if (cell + controller_update_phase) % controller_update_interval != 0:
            if klystron_time_constant > 0.0:
                # The klystron keeps moving between samples.
                generator_current_out[cell] = klystron_cell(
                    generator_current_drive,
                    klystron_command,
                    omega_times_dt[cell] / omega_input,
                    klystron_time_constant,
                )
            else:
                # Between samples the loop holds its last command.
                generator_current_out[cell] = generator_current_drive
            continue
        error = regulation_error(
            voltage,
            voltage_setpoint,
            setpoint_feedforward[cell],
            kick_frame_rotation[cell],
            error_frame_rotation[cell],
        )
        # The command is held for the whole update interval, so the
        # integrator credits it with that much time.
        delta_t = (
            omega_times_dt[cell] / omega_input * controller_update_interval
        )
        delayed_error, delay_head = delay_line_push(
            delay_buffer, delay_head, error
        )
        klystron_command, integral = pi_law_step(
            delayed_error,
            delta_t,
            integral,
            gain_proportional,
            gain_integral,
            generator_current_bias + generator_current_feedforward[cell],
            max_output,
            anti_windup,
        )
        if klystron_time_constant > 0.0:
            generator_current_out[cell] = klystron_cell(
                generator_current_drive,
                klystron_command,
                omega_times_dt[cell] / omega_input,
                klystron_time_constant,
            )
        else:
            generator_current_out[cell] = klystron_command
    return delay_buffer, delay_head, integral, klystron_command


@nb.njit(cache=True)  # pragma: no cover
def envelope_p_scan(
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
    error_frame_rotation,
    beam_step_rotation,
    controller_update_interval,
    controller_update_phase,
    voltage_setpoint,
    setpoint_feedforward,
    generator_current_feedforward,
    omega_input,
    gain_proportional,
    generator_current_bias,
    delay_buffer,
    delay_head,
    max_output,
    klystron_time_constant=0.0,
    klystron_command=0.0 + 0.0j,
):
    """
    The cavity model closed through the proportional law, over one span.

    The same cavity and sampler arguments as :func:`envelope_pi_scan`, then
    the P controller's own state, which has no integral.

    Parameters
    ----------
    voltage_multiplier, drive_weight, omega_times_dt, beam_current
        Per-cell cavity step inputs.
    voltage_gen_out, voltage_beam_out, voltage_out
        Output voltages, written in place.
    generator_current_out
        Generator current, in/out as for :func:`envelope_pi_scan`.
    voltage_gen_init, voltage_beam_init, generator_current_init
        State seeding the first cell.
    r_over_q
        Cavity ``R/Q`` [Ohm].
    generator_frame_rotation, kick_frame_rotation, error_frame_rotation
        Per-cell frame rotations.
    beam_step_rotation
        Per-cell counter-rotation of the carried beam component.
    controller_update_interval
        Cells between controller samples.
    controller_update_phase
        Phase of the first cell on that clock.
    voltage_setpoint
        Setpoint in the IQ frame.
    setpoint_feedforward
        Per-cell addition to the setpoint [V].
    generator_current_feedforward
        Per-cell addition to the bias [A], read on controller samples
        and clamped with it. Zero is a no-op.
    omega_input
        Segment angular frequency, to recover the cell length for the
        klystron pole; a proportional law has no time constant of its own.
    gain_proportional, generator_current_bias
        P tuning.
    delay_buffer, delay_head
        The loop's delay line, advanced in place.
    max_output
        Klystron current-magnitude limit, or ``inf``.
    klystron_time_constant
        Time constant of the klystron pole [s]; 0, no pole, if omitted.
    klystron_command
        Command the klystron is chasing as the span starts.

    Returns
    -------
    delay_buffer, delay_head, klystron_command
        The P state after the span, for
        ``GeneratorCurrentPController.absorb_envelope_scan_state``.
    """
    voltage_gen_previous = voltage_gen_init
    voltage_beam_previous = voltage_beam_init
    for cell in range(omega_times_dt.shape[0]):
        if cell == 0:
            generator_current_drive = generator_current_init
        else:
            generator_current_drive = generator_current_out[cell - 1]
        voltage_beam, voltage_gen, voltage = propagate_envelope_cell(
            voltage_beam_previous,
            voltage_gen_previous,
            generator_current_drive,
            beam_current[cell],
            omega_times_dt[cell],
            voltage_multiplier[cell],
            drive_weight[cell],
            r_over_q,
            beam_step_rotation[cell],
            generator_frame_rotation[cell],
        )
        voltage_beam_out[cell] = voltage_beam
        voltage_gen_out[cell] = voltage_gen
        voltage_out[cell] = voltage
        voltage_beam_previous = voltage_beam
        voltage_gen_previous = voltage_gen
        if (cell + controller_update_phase) % controller_update_interval != 0:
            if klystron_time_constant > 0.0:
                generator_current_out[cell] = klystron_cell(
                    generator_current_drive,
                    klystron_command,
                    omega_times_dt[cell] / omega_input,
                    klystron_time_constant,
                )
            else:
                generator_current_out[cell] = generator_current_drive
            continue
        error = regulation_error(
            voltage,
            voltage_setpoint,
            setpoint_feedforward[cell],
            kick_frame_rotation[cell],
            error_frame_rotation[cell],
        )
        delayed_error, delay_head = delay_line_push(
            delay_buffer, delay_head, error
        )
        klystron_command = p_law_step(
            delayed_error,
            gain_proportional,
            generator_current_bias + generator_current_feedforward[cell],
            max_output,
        )
        if klystron_time_constant > 0.0:
            generator_current_out[cell] = klystron_cell(
                generator_current_drive,
                klystron_command,
                omega_times_dt[cell] / omega_input,
                klystron_time_constant,
            )
        else:
            generator_current_out[cell] = klystron_command
    return delay_buffer, delay_head, klystron_command
