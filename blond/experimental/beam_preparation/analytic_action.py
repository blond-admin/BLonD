# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

r"""Action :math:`J(H)` of the analytic longitudinal Hamiltonian.

For a given Hamiltonian level :math:`H`, the particle oscillates between
the turning points where :math:`V(t) = H`, with

.. math::

    \Delta E(t) = \sqrt{\frac{H - V(t)}{\mathrm{eom\_factor\_dE}}}

and the action is the phase-space area enclosed by that orbit, divided by
:math:`2\pi`:

.. math::

    J(H) = \frac{1}{2\pi} \oint \Delta E\, \mathrm{d}t
         = \frac{1}{\pi} \int_{V(t) \leq H} \Delta E(t)\, \mathrm{d}t

(the closed orbit is symmetric in :math:`\pm\Delta E`, hence the factor 2
that turns :math:`1/2\pi` into :math:`1/\pi`). The longitudinal emittance
is :math:`\varepsilon = 2\pi J`.

This reproduces the BLonD 2 action integral used by
``matched_from_distribution_function`` / ``compute_x_grid``, and allows
the distribution to be specified in either the Hamiltonian or the Action
variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.experimental.beam_preparation.analytic_potential_well import (
    check_single_bucket_well,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def action_from_potential_well(
    time_array: NumpyArray,
    potential_well: NumpyArray,
    *,
    eom_factor_dE: float,
    single_bucket_tolerance: float = 1e-2,
    allow_inner_buckets: bool = False,
    verbose: bool = False,
    plot: bool = False,
) -> tuple[NumpyArray, NumpyArray]:
    r"""
    Action :math:`J` as a function of the Hamiltonian :math:`H`.

    The action is evaluated at every Hamiltonian level present in
    ``potential_well`` (i.e. at :math:`H = V(t_i)` for each sample), then
    sorted by :math:`H` so the result can be used directly with
    :func:`numpy.interp`.

    Parameters
    ----------
    time_array
        Time coordinates of the potential well, in [s]. May be
        non-uniform (the integration uses the actual coordinates).
    potential_well
        Potential well at ``time_array``, in [eV]. Must be cut to a
        single bucket with its minimum at 0 — validated by
        :func:`check_single_bucket_well` (``ValueError`` otherwise):
        integrating an uncut (margined, multi-bucket or tilted) well
        would silently inflate :math:`J`.
    eom_factor_dE
        Kinetic coefficient :math:`|\eta_0|/(2\beta^2 E)`, in [1/eV].
    single_bucket_tolerance
        Relative tolerance forwarded to
        :func:`check_single_bucket_well` (defined in
        ``analytic_potential_well``); loosen for coarsely sampled
        separatrix cuts, tighten for pristine stationary wells.
    allow_inner_buckets
        If True, a well with prominent inner maxima (e.g. split by an
        induced potential during intensity iterations) is accepted with
        a warning instead of raising; the zero-padded integral then
        sums the islands below the inner separatrices (BLonD 2
        behavior).
    verbose
        If True, print diagnostic quantities.
    plot
        If True, draw a diagnostic :math:`J(H)` curve.

    Returns
    -------
    sorted_hamiltonian
        Hamiltonian levels sorted in increasing order, in [eV].
    sorted_action
        Action at those levels, in [eV.s].

    Notes
    -----
    The evaluation is ``O(n^2)`` in the number of well samples, as in
    BLonD 2, and the number of :math:`H` levels is currently tied to the
    well resolution (decoupling into an ``n_levels`` parameter is
    planned). A few thousand points is ample for matching purposes.

    Accuracy caveat: at coarse resolutions (~1e3 samples) the
    small-amplitude tail of :math:`J(H)` (levels below ~1e-4 of the
    separatrix) carries O(10 %) discretization error, which propagates
    to ``dH/dJ``-based synchrotron-frequency estimates at the 1e-3 to
    1e-2 level; use >= 1e4 samples when the small-amplitude region or
    an f_s extraction matters. Emittance/bunch-length targeting at
    typical bunch sizes is unaffected.
    """
    time_array = np.asarray(time_array, dtype=float)
    potential_well = np.asarray(potential_well, dtype=float)
    assert time_array.shape == potential_well.shape, (
        f"{time_array.shape=} must match {potential_well.shape=}"
    )
    check_single_bucket_well(
        potential_well,
        relative_tolerance=single_bucket_tolerance,
        allow_inner_buckets=allow_inner_buckets,
    )

    # Zero-padded formulation (as in BLonD 2): dE is zero outside the
    # sublevel set and the integral runs over the full grid. On a single
    # bucket this matches the masked integral; on a well with several
    # minima (e.g. intensity-split, accepted via a loosened
    # `single_bucket_tolerance`) it avoids the spurious gap-wide chord a
    # masked integral would draw between disconnected islands, and the
    # islands' areas are summed — the legacy semantics.
    action = np.zeros(len(potential_well), dtype=float)
    for index in range(len(potential_well)):
        hamiltonian_level = potential_well[index]
        deltaE_trajectory = np.sqrt(
            np.maximum(hamiltonian_level - potential_well, 0.0) / eom_factor_dE
        )
        action[index] = np.trapezoid(deltaE_trajectory, x=time_array) / np.pi

    order = potential_well.argsort()
    sorted_hamiltonian = potential_well[order]
    sorted_action = action[order]

    if verbose:
        print(
            "[action_from_potential_well] "
            f"n_levels={len(sorted_action)}, "
            f"H range=[{sorted_hamiltonian[0]:.3e}, "
            f"{sorted_hamiltonian[-1]:.3e}] eV, "
            f"J max={sorted_action.max():.3e} eV.s, "
            f"emittance max={2 * np.pi * sorted_action.max():.3e} eV.s"
        )

    if plot:
        _plot_action(sorted_hamiltonian, sorted_action)

    return sorted_hamiltonian, sorted_action


def action_grid(
    hamilton_2D: NumpyArray,
    sorted_hamiltonian: NumpyArray,
    sorted_action: NumpyArray,
) -> NumpyArray:
    r"""
    Map a 2D Hamiltonian grid onto the action variable.

    Parameters
    ----------
    hamilton_2D
        2D Hamiltonian grid, in [eV].
    sorted_hamiltonian
        Hamiltonian levels sorted increasingly, in [eV].
    sorted_action
        Action at those levels, in [eV.s].

    Returns
    -------
    action_2D
        Action on the same grid, in [eV.s]. Values above the largest
        tabulated Hamiltonian (i.e. outside the bucket) are set to
        ``np.inf``, following BLonD 2, so that distributions evaluate to
        zero there.

    Notes
    -----
    Contract for consumers (the analytic density families): the
    ``np.inf`` markers make exponential-form densities vanish cleanly
    (``np.exp(-inf) == 0``), but power-law families of the form
    ``(1 - X/X0)**exponent`` produce ``nan`` (fractional exponents, with
    a RuntimeWarning) or ``inf`` (integer exponents) — mask
    ``X > X0`` *before* applying the power, do not clean up afterwards.
    """
    return np.interp(
        hamilton_2D,
        sorted_hamiltonian,
        sorted_action,
        left=0.0,
        right=np.inf,
    )


def hamiltonian_from_emittance(
    emittance: float,
    sorted_hamiltonian: NumpyArray,
    sorted_action: NumpyArray,
) -> float:
    r"""
    Hamiltonian level enclosing a given longitudinal emittance.

    Inverts :math:`\varepsilon = 2\pi J` through the tabulated
    :math:`J(H)`, reproducing the BLonD 2 ``compute_H0``.

    Parameters
    ----------
    emittance
        Target longitudinal emittance, in [eV.s]. Must not exceed the
        bucket area :math:`2\pi J_\mathrm{sep}`.
    sorted_hamiltonian
        Hamiltonian levels sorted increasingly, in [eV].
    sorted_action
        Action at those levels, in [eV.s].

    Returns
    -------
    hamiltonian_0
        Hamiltonian level whose orbit encloses ``emittance``, in [eV].

    Raises
    ------
    ValueError
        If ``emittance`` exceeds the bucket area (BLonD 2 silently
        clamped to the separatrix in this case).
    """
    bucket_area = 2.0 * np.pi * float(sorted_action[-1])
    if emittance > bucket_area:
        raise ValueError(
            f"Requested emittance {emittance:.4e} eV.s exceeds the "
            f"bucket area {bucket_area:.4e} eV.s"
        )
    return float(
        np.interp(emittance / (2.0 * np.pi), sorted_action, sorted_hamiltonian)
    )


def _plot_action(
    sorted_hamiltonian: NumpyArray, sorted_action: NumpyArray
) -> None:
    """Quick diagnostic plot of the action versus the Hamiltonian."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(num="action_from_potential_well")
    ax.plot(sorted_hamiltonian, sorted_action)
    ax.set_xlabel("Hamiltonian [eV]")
    ax.set_ylabel("Action J [eV.s]")
    ax.set_title("Action versus Hamiltonian")
    ax.grid(alpha=0.3)
    fig.tight_layout()
