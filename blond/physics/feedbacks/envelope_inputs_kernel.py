# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Compiled per-cell frame rotations of the coarse-envelope scan.

The scan of :mod:`~blond.physics.feedbacks.envelope_kernel` takes its
state-independent inputs precomputed, among them the per-cell frame
rotations ``exp(+-i phase)``. Every station computes them over its whole
turn of coarse cells on every passage (~10^5 cells), so as a chain of NumPy
temporaries on one core they cost more than the scan itself. This kernel
computes them in one fused, threaded pass. (The propagator's ``B`` and ``W``
need no kernel: they are per segment, see
``IQCavityFeedbackCoarseGrid._segment_step_multipliers``.)

It reproduces the NumPy expression it replaces **byte-for-byte**, which the
scan's bit identity with the per-cell Python reference requires; a zero
phase gives exactly ``1 + 0j``, as the ``np.where`` short-circuit did, so an
unrotated passage stays free of ``exp`` sign dust.

Pinned in ``tests/unittests/physics/feedbacks/test_envelope_inputs_kernel.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray

#: Below this many cells a serial loop beats the thread launch.
PARALLEL_THRESHOLD = 4096


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
