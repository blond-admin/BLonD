# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
First-order (forward-Euler) cavity-response solver.

The muon-collider timing-class feedback and the (experimental) LHC feedback
both use :func:`cavity_response_sparse_matrix`. The beam-current demodulation
and IQ helpers that used to live here moved to ``beam_current.py`` and
``iq.py``; they are re-exported below so existing imports keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from blond.physics.feedbacks.beam_current import (  # noqa: F401  (re-export)
    low_pass_filter,
    rf_beam_current,
    rf_beam_current_partial,
)
from blond.physics.feedbacks.iq import (  # noqa: F401  (re-export)
    cartesian_to_polar,
    polar_to_cartesian,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def cavity_response_sparse_matrix(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    samples_per_rf: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
):
    """
    Solver for the ACS cavity response model as a sparse matrix problem.

    Solving the ACS cavity response model as a sparse matrix problem
    for a given set of initial conditions, resonator parameters and
    generator and RF beam currents. The input arrays are extended by
    one entry (I_gen_init and V_ant_init respectively) to take
    respect the fact that the first matrix entry is not part of the solution
    domain.

    Parameters
    ----------
    I_beam : complex array
        RF beam current.
    I_gen : complex array
        Generator current.
    V_ant_init : complex float
        Initial condition for the antenna voltage.
    I_gen_init : complex float
        Initial condition for the generator current.
    samples_per_rf : float
        RF phase advanced per sample [rad], i.e. ``omega_rf * sampling_time``
        (= 2*pi / samples-per-period). The solver coefficients use it directly
        as ``omega * dt``; callers pass ``omega_input * profile.hist_step``.
    R_over_Q : float
        The R over Q of the cavity.
    Q_L : float
        The loaded quality factor of the cavity.
    relative_detuning : float
        The detuning of the cavity in frequency divided by the rf frequency.

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen of length len(I_gen).
    """
    assert len(I_beam) == len(I_gen), (
        "length of beam and generator currents need to match"
    )

    # Extend arrays to take initial values into account
    internal_I_gen = np.concatenate(([I_gen_init], I_gen))
    internal_I_beam = np.concatenate(([0j], I_beam))

    n_samples = len(internal_I_gen)

    # Compute matrix elements
    A = 0.5 * R_over_Q * samples_per_rf
    B = (
        1
        - 0.5 * samples_per_rf / Q_L
        + 1j * relative_detuning * samples_per_rf
    )

    # Initialize the two sparse matrices needed to find antenna voltage
    B_matrix = diags(
        [-B, 1],
        [-1, 0],
        (n_samples, n_samples),
        dtype=complex,
        format="csc",
    )
    I_matrix = diags([A], [-1], (n_samples, n_samples), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * internal_I_gen - internal_I_beam)
    b[0] = V_ant_init

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)[1:]
    # first value is intial condition
