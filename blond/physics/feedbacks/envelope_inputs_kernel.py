# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Compiled per-cell inputs of the coarse-envelope scan.

The scan of :mod:`~blond.physics.feedbacks.envelope_kernel` takes its
state-independent inputs precomputed: per cell the propagator's voltage
multiplier ``B = e^L`` and drive weight ``W = (e^L - 1) / L``, and the frame
rotations ``exp(+-i phase)``. Every station computes them over its whole
turn of coarse cells on every passage (~10^5 cells), so as a chain of NumPy
temporaries on one core they cost more than the scan itself. These kernels
compute them in one fused, threaded pass.

They reproduce the NumPy expressions they replace **byte-for-byte**, which
the scan's bit identity with the per-cell Python reference requires:

- ``B`` spells out NumPy's complex ``exp`` for a finite exponent,
  ``(e^x cos y, e^x sin y)``, whose three calls ``W`` reuses;
- ``W`` spells out NumPy's complex ``expm1`` (numba's own loses ~1e-13
  relative accuracy for ``|L| << 1``, which is exactly the coarse-step
  regime) and NumPy's complex division (Smith's method with a reciprocal
  scale), rather than numba's, which divides by the denominator;
- a zero phase gives exactly ``1 + 0j``, as the ``np.where`` short-circuit
  did, so an unrotated passage stays free of ``exp`` sign dust.

Pinned in ``tests/unittests/physics/feedbacks/test_envelope_inputs_kernel.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba as nb  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

#: Below this many cells a serial loop beats the thread launch.
PARALLEL_THRESHOLD = 4096


@nb.njit(cache=True, inline="always")  # pragma: no cover
def _numpy_complex_divide(
    numerator_real, numerator_imag, denominator_real, denominator_imag
):
    """
    Divide as NumPy divides complex numbers (Smith, reciprocal scale).

    The denominator is never zero here (``L != 0`` for a positive step), so
    NumPy's division-by-zero branch is not reproduced.

    Parameters
    ----------
    numerator_real, numerator_imag
        Numerator.
    denominator_real, denominator_imag
        Denominator, not zero.

    Returns
    -------
    complex
        The quotient.
    """
    if abs(denominator_real) >= abs(denominator_imag):
        ratio = denominator_imag / denominator_real
        scale = 1.0 / (denominator_real + denominator_imag * ratio)
        return complex(
            (numerator_real + numerator_imag * ratio) * scale,
            (numerator_imag - numerator_real * ratio) * scale,
        )
    ratio = denominator_real / denominator_imag
    scale = 1.0 / (denominator_imag + denominator_real * ratio)
    return complex(
        (numerator_real * ratio + numerator_imag) * scale,
        (numerator_imag * ratio - numerator_real) * scale,
    )


@nb.njit(cache=True, inline="always")  # pragma: no cover
def _step_multipliers_cell(
    omega_times_dt, q_loaded, relative_detuning, index, multiplier, weight
):
    """
    Store the propagator weights of one coarse cell.

    Computes the voltage multiplier ``B = e^L`` and the drive weight
    ``W = (e^L - 1) / L`` of cell ``index`` and writes them to
    ``multiplier[index]`` and ``weight[index]``.

    Parameters
    ----------
    omega_times_dt
        Per-cell ``omega * dt`` [rad].
    q_loaded
        Loaded quality factor.
    relative_detuning
        ``delta_omega / omega``.
    index
        The cell.
    multiplier, weight
        Output arrays.
    """
    step = omega_times_dt[index]
    # coarse_step_exponent: -0.5 * wdt / Q_L + 1j * rel * wdt, whose real
    # and imaginary parts NumPy forms as exactly these two products.
    exponent_real = -0.5 * step / q_loaded
    exponent_imag = relative_detuning * step
    # B = e^L as NumPy's complex exp forms it for a finite exponent,
    # (e^x cos y, e^x sin y); NumPy's expm1 needs the same e^x, cos y and
    # sin y, so the cell costs five transcendental calls instead of eight.
    growth = math.exp(exponent_real)
    cosine = math.cos(exponent_imag)
    sine = math.sin(exponent_imag)
    multiplier[index] = complex(growth * cosine, growth * sine)
    # NumPy's complex expm1, operation for operation.
    half_sine = math.sin(exponent_imag / 2)
    weight[index] = _numpy_complex_divide(
        math.expm1(exponent_real) * cosine - 2 * half_sine * half_sine,
        growth * sine,
        exponent_real,
        exponent_imag,
    )


@nb.njit(cache=True)  # pragma: no cover
def _step_multipliers_serial(
    omega_times_dt, q_loaded, relative_detuning, multiplier, weight
):
    """
    Fill ``B`` and ``W`` serially; see :func:`step_multipliers`.

    Parameters
    ----------
    omega_times_dt
        Per-cell ``omega * dt`` [rad].
    q_loaded
        Loaded quality factor.
    relative_detuning
        ``delta_omega / omega``.
    multiplier, weight
        Output arrays, one entry per cell.
    """
    for index in range(omega_times_dt.size):
        _step_multipliers_cell(
            omega_times_dt,
            q_loaded,
            relative_detuning,
            index,
            multiplier,
            weight,
        )


@nb.njit(cache=True, parallel=True)  # pragma: no cover
def _step_multipliers_parallel(
    omega_times_dt, q_loaded, relative_detuning, multiplier, weight
):
    """
    Fill ``B`` and ``W`` on threads; see :func:`step_multipliers`.

    Parameters
    ----------
    omega_times_dt
        Per-cell ``omega * dt`` [rad].
    q_loaded
        Loaded quality factor.
    relative_detuning
        ``delta_omega / omega``.
    multiplier, weight
        Output arrays, one entry per cell.
    """
    for index in nb.prange(omega_times_dt.size):
        _step_multipliers_cell(
            omega_times_dt,
            q_loaded,
            relative_detuning,
            index,
            multiplier,
            weight,
        )


def step_multipliers(
    omega_times_dt: NumpyArray,
    q_loaded: float,
    relative_detuning: float,
) -> tuple[NumpyArray, NumpyArray]:
    """
    Per-cell propagator weights ``B = e^L`` and ``W = (e^L - 1) / L``.

    Byte-for-byte equal to ``exponential_voltage_multiplier`` and
    ``exponential_drive_weight`` of
    ``coarse_step_exponent(omega_times_dt, q_loaded, relative_detuning)``
    (all in :mod:`~blond.physics.feedbacks.cavity_solvers`), in one pass.

    Parameters
    ----------
    omega_times_dt
        Per-cell ``omega * dt`` [rad], strictly positive (a zero step makes
        ``L = 0``, where ``W`` is undefined; the caller routes those spans
        to the reference path).
    q_loaded
        Loaded quality factor of the cavity.
    relative_detuning
        Detuning normalised to the segment frequency, ``delta_omega /
        omega``.

    Returns
    -------
    voltage_multiplier
        Per-cell ``B`` (complex128).
    drive_weight
        Per-cell ``W`` (complex128).
    """
    omega_times_dt = np.ascontiguousarray(omega_times_dt, dtype=np.float64)
    multiplier = np.empty(omega_times_dt.size, dtype=np.complex128)
    weight = np.empty(omega_times_dt.size, dtype=np.complex128)
    fill = (
        _step_multipliers_parallel
        if omega_times_dt.size >= PARALLEL_THRESHOLD
        else _step_multipliers_serial
    )
    fill(
        omega_times_dt,
        float(q_loaded),
        float(relative_detuning),
        multiplier,
        weight,
    )
    return multiplier, weight


@nb.njit(cache=True)  # pragma: no cover
def _unit_phasors_serial(phases, sign, phasors):
    """
    Fill the phasors serially; see :func:`unit_phasors`.

    Parameters
    ----------
    phases
        Per-cell phases [rad].
    sign
        Direction of the rotation, ``+1.0`` or ``-1.0``.
    phasors
        Output array, one entry per cell.
    """
    for index in range(phases.size):
        phase = phases[index]
        if phase == 0.0:
            phasors[index] = complex(1.0, 0.0)
        else:
            phasors[index] = np.exp(complex(0.0, sign * phase))


@nb.njit(cache=True, parallel=True)  # pragma: no cover
def _unit_phasors_parallel(phases, sign, phasors):
    """
    Fill the phasors on threads; see :func:`unit_phasors`.

    Parameters
    ----------
    phases
        Per-cell phases [rad].
    sign
        Direction of the rotation, ``+1.0`` or ``-1.0``.
    phasors
        Output array, one entry per cell.
    """
    for index in nb.prange(phases.size):
        phase = phases[index]
        if phase == 0.0:
            phasors[index] = complex(1.0, 0.0)
        else:
            phasors[index] = np.exp(complex(0.0, sign * phase))


def unit_phasors(phases: NumpyArray, sign: float) -> NumpyArray:
    """
    Per-cell ``exp(sign * 1j * phase)``, exactly ``1 + 0j`` at zero.

    Byte-for-byte equal to ``np.where(phases == 0.0, 1.0 + 0.0j,
    np.exp(sign * 1j * phases))``, without evaluating ``exp`` where the
    phase is zero -- so a passage with no rotation costs one pass of
    stores.

    Parameters
    ----------
    phases
        Per-cell phases [rad].
    sign
        ``+1.0`` or ``-1.0``, the direction of the rotation.

    Returns
    -------
    phasors
        Per-cell unit phasors (complex128).
    """
    phases = np.ascontiguousarray(phases, dtype=np.float64)
    phasors = np.empty(phases.size, dtype=np.complex128)
    fill = (
        _unit_phasors_parallel
        if phases.size >= PARALLEL_THRESHOLD
        else _unit_phasors_serial
    )
    fill(phases, float(sign), phasors)
    return phasors
