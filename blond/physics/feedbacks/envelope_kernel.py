# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Numba kernels of the cavity model: the coarse-grid envelope recursion.

This module is the CAVITY and nothing else. It knows the envelope ODE, the
source split and the frame rotations, and it knows no controller: no gain,
no integral, no delay line and no setpoint appear anywhere in it. A
generator-current control law closes the loop *around* this model, in
:mod:`~blond.physics.feedbacks.control_law_kernels`, where each law's
compiled scan calls :func:`propagate_envelope_cell` once per cell and
interleaves its own update.

The per-cell antenna-voltage recursion of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackCoarseGrid`
is inherently sequential (each cell reads the previous cell's voltage and
generator current), so it cannot be vectorised, and in pure Python the
~10^5 per-turn cells are dominated by interpreter overhead; these kernels
compile it.

The envelope ODE is linear, so the recursion runs as TWO independent state
components through the same propagator -- superposition is exact:

- the *beam-sourced* component, driven by ``-I_beam / 2`` alone, anchored
  to the demodulation frame;
- the *generator-sourced* component, driven by ``I_gen`` alone, natively
  anchored to the piecewise design clock the coarse grid samples.

Each cell composes the demodulation-frame sum
``V = V_beam + V_gen * generator_frame_rotation[c]`` (the rotation takes
the design-anchored component into the demodulation frame; see
``IQCavityFeedbackCoarseGrid._update_frame_rotations``).

The kernels do not evaluate the propagator: the per-cell voltage multiplier
``B = e^L`` and drive weight ``W = (e^L - 1) / L`` of the exact exponential
step depend only on the step size and detuning, so they are precomputed on
the host (``_circuit_track_cells_kernel``) and passed in, as are the
per-cell frame rotations. The kernels carry only the state-dependent update
``V = V_prev * B + drive * W``, which keeps them identical -- byte-for-byte
on complex128 -- to the Python reference path. The derivation of ``B`` and
``W`` is in the Notes of ``IQCavityFeedbackCoarseGrid._advance_coarse_voltage``.
"""

from __future__ import annotations

import numba as nb  # type: ignore


@nb.njit(cache=True)  # pragma: no cover
def propagate_envelope_cell(
    voltage_beam_previous,
    voltage_gen_previous,
    generator_current_drive,
    beam_current,
    omega_times_dt,
    voltage_multiplier,
    drive_weight,
    r_over_q,
    beam_step_rotation,
    generator_frame_rotation,
):
    r"""
    Advance the cavity envelope by one coarse cell.

    Advances the two source-split components of the (linear) envelope ODE
    through the same propagator,

    .. math::
        V_{\mathrm{beam}} = V_{\mathrm{beam}}^{-}\,B
            + (R/Q)\,\omega\Delta t\,(0 - \tfrac12 I_{\mathrm{beam}})\,W,
        \quad
        V_{\mathrm{gen}} = V_{\mathrm{gen}}^{-}\,B
            + (R/Q)\,\omega\Delta t\,I_{\mathrm{gen}}\,W,

    and composes the demodulation-frame sum
    ``V = V_beam + V_gen * generator_frame_rotation``.

    Parameters
    ----------
    voltage_beam_previous
        Beam-sourced voltage of the previous cell.
    voltage_gen_previous
        Generator-sourced voltage of the previous cell.
    generator_current_drive
        Generator current driving this cell (the previous cell's command).
    beam_current
        Beam current of this cell; zero for a no-beam segment.
    omega_times_dt
        ``omega * dt`` of this cell.
    voltage_multiplier
        Voltage multiplier ``B`` of this cell.
    drive_weight
        Drive weight ``W`` of this cell.
    r_over_q
        Cavity ``R/Q`` [Ohm].
    beam_step_rotation
        Counter-rotation ``exp(-i * step)`` of the carried beam-sourced
        component when the station's phase-loop offset stepped into this
        cell -- the field stays put while the RF reference moves; exactly
        unity otherwise, where it is skipped.
    generator_frame_rotation
        Rotation taking this cell's design-anchored generator component
        into the demodulation frame.

    Returns
    -------
    voltage_beam, voltage_gen, voltage
        The two components and their demodulation-frame sum.
    """
    # Beam-sourced component: no generator current. ``0.0 -`` rather than
    # a bare negation keeps an empty cell's drive at +0.0, as in the
    # reference; ``-0.5 * beam_current`` would make it -0.0.
    drive_beam = r_over_q * omega_times_dt * (0.0 - 0.5 * beam_current)
    if beam_step_rotation != 1.0:
        voltage_beam_previous = voltage_beam_previous * beam_step_rotation
    voltage_beam = voltage_beam_previous * voltage_multiplier + (
        drive_beam * drive_weight
    )
    # Generator-sourced component: same propagator, no beam current. With
    # nothing driving the generator it stays exactly zero and the
    # composition adds an exact zero.
    drive_gen = r_over_q * omega_times_dt * generator_current_drive
    voltage_gen = voltage_gen_previous * voltage_multiplier + (
        drive_gen * drive_weight
    )
    voltage = voltage_beam + voltage_gen * generator_frame_rotation
    return voltage_beam, voltage_gen, voltage


@nb.njit(cache=True)  # pragma: no cover
def envelope_open_loop_scan(
    voltage_multiplier,
    drive_weight,
    omega_times_dt,
    beam_current,
    voltage_gen_out,
    voltage_beam_out,
    voltage_out,
    generator_current,
    voltage_gen_init,
    voltage_beam_init,
    generator_current_init,
    r_over_q,
    generator_frame_rotation,
    beam_step_rotation,
):
    """
    Run the cavity model over a span with a given drive and no regulation.

    The feedback's path when no controller is attached: every cell is
    driven by the generator current already on the grid, which this scan
    reads and never writes.

    Parameters
    ----------
    voltage_multiplier
        Per-cell voltage multiplier ``B`` (complex128, length ``N``).
    drive_weight
        Per-cell drive weight ``W`` (complex128, length ``N``).
    omega_times_dt
        Per-cell ``omega * dt`` (float64, length ``N``).
    beam_current
        Per-cell beam current (complex128, length ``N``).
    voltage_gen_out
        Output generator-sourced voltage, written in place.
    voltage_beam_out
        Output beam-sourced voltage, written in place.
    voltage_out
        Output demodulation-frame sum, written in place.
    generator_current
        Generator current on the grid (complex128, length ``N``), read
        only: cell ``c`` is driven by entry ``c - 1``.
    voltage_gen_init
        Generator-sourced voltage seeding the first cell.
    voltage_beam_init
        Beam-sourced voltage seeding the first cell.
    generator_current_init
        Generator current driving the first cell.
    r_over_q
        Cavity ``R/Q`` [Ohm].
    generator_frame_rotation
        Per-cell rotation of the generator component into the
        demodulation frame (complex128, length ``N``).
    beam_step_rotation
        Per-cell counter-rotation of the carried beam component
        (complex128, length ``N``); unity where the reference did not step.
    """
    voltage_gen_previous = voltage_gen_init
    voltage_beam_previous = voltage_beam_init
    for cell in range(omega_times_dt.shape[0]):
        if cell == 0:
            generator_current_drive = generator_current_init
        else:
            generator_current_drive = generator_current[cell - 1]
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
