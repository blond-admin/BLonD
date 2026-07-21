# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Analytic RF potential well from the RF voltage waveform.

Building blocks that reconstruct the single-turn RF potential well
analytically, following the BLonD 2 ``potential_well_generation`` philosophy
(the RF voltages are averaged over one turn). These feed the analytic
distribution / line-density matchers ported from BLonD 2.

The wells produced here are *uncut*: restricting a well to a single RF
bucket (separatrix cut) is a separate, upcoming step. Downstream consumers
(``hamiltonian_grid``, ``action_from_potential_well``) require a cut,
single-bucket well and enforce it via :func:`check_single_bucket_well`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def bucket_time_array(
    omega_rf: float,
    n_points: int = 10_000,
    dt_margin_fraction: float = 0.0,
) -> NumpyArray:
    """
    Uniform time grid spanning one RF period of the main harmonic.

    Parameters
    ----------
    omega_rf
        Angular frequency of the main RF harmonic, in [rad/s].
    n_points
        Number of points in the grid.
    dt_margin_fraction
        Fraction of one RF period added as margin, split evenly before
        and after, so that extrema sitting exactly on the frame edges
        remain visible. ``0.4`` adds 40 % of one RF period (same meaning
        as the BLonD 2 ``dt_margin_percent`` parameter).

    Returns
    -------
    time_array
        Time coordinates spanning one (optionally margined) RF period,
        in [s].

    Notes
    -----
    The frame spans ``[0, t_rf]`` plus margin. Following the BLonD 2
    convention, choose ``phi_rf`` such that the stable phase sits near
    the frame centre: for positive ``eta_0 * charge`` (e.g. protons
    above transition) use ``phi_rf = 0``; for negative ``eta_0 * charge``
    (e.g. protons below transition) use ``phi_rf = pi``. Otherwise the
    bucket is split across the frame edges — detected loudly downstream
    by :func:`check_single_bucket_well`.
    """
    rf_period = 2.0 * np.pi / omega_rf
    margin = dt_margin_fraction * rf_period
    return np.linspace(
        -margin / 2.0, rf_period + margin / 2.0, int(n_points)
    )


def rf_potential_well(
    time_array: NumpyArray,
    total_voltage: NumpyArray,
    *,
    charge: float,
    t_rev: float,
    eta_0: float,
    energy_gain_per_turn: float = 0.0,
    subtract_min: bool = True,
    verbose: bool = False,
    plot: bool = False,
) -> NumpyArray:
    r"""
    Analytic RF potential well from a total RF voltage waveform.

    Integrates the RF voltage over one turn to obtain the effective
    potential landscape the beam experiences, reproducing the BLonD 2
    ``potential_well_generation`` result:

    .. math::

        V_\mathrm{eff}(t) = V_\mathrm{RF}(t)
            - \frac{\Delta E_\mathrm{turn}}{|q|}

        \Phi(t) = - \int_{t_0}^{t}
            \mathrm{sign}(\eta_0)\, \frac{q}{T_\mathrm{rev}}\,
            V_\mathrm{eff}(t')\, \mathrm{d}t'

    The subtracted term references the well to the synchronous particle
    (the average accelerating voltage per turn).

    Parameters
    ----------
    time_array
        Time coordinates of the voltage waveform, in [s]. May be
        non-uniform.
    total_voltage
        Total RF voltage summed over all harmonics at ``time_array``,
        in [V], as a host NumPy array — e.g. the output of
        ``rf_station.calc_gap_voltage_without_feedbacks`` converted
        with ``copy_to_cpu``.
    charge
        Particle charge, as number of elementary charges ``e``.
    t_rev
        Revolution period, in [s].
    eta_0
        Zeroth-order slippage factor. Only its sign enters the potential.
    energy_gain_per_turn
        Design energy gain per turn, in [eV]. Zero for a coasting or
        constant cycle; non-zero on a ramp, where it tilts the well.
    subtract_min
        If True (default), shift the well so its minimum is at zero.
    verbose
        If True, print diagnostic quantities.
    plot
        If True, draw a diagnostic figure of the voltage and the well.

    Returns
    -------
    potential_well
        Effective potential at ``time_array``, in [eV].

    Notes
    -----
    The returned well is *uncut*. Downstream Hamiltonian/action
    functions require a single-bucket well (cut at the separatrix, with
    the stable phase inside the frame — see the ``phi_rf`` convention in
    :func:`bucket_time_array` and :func:`check_single_bucket_well`).
    """
    time_array = np.asarray(time_array, dtype=float)
    total_voltage = np.asarray(total_voltage, dtype=float)
    assert time_array.shape == total_voltage.shape, (
        f"{time_array.shape=} must match {total_voltage.shape=}"
    )

    eom_factor_potential = np.sign(eta_0) * charge / t_rev

    # RF voltage seen relative to the synchronous accelerating voltage
    effective_voltage = total_voltage - energy_gain_per_turn / abs(charge)

    potential_well = -cumulative_trapezoid(
        eom_factor_potential * effective_voltage,
        x=time_array,
        initial=0.0,
    )

    if subtract_min:
        potential_well = potential_well - np.min(potential_well)

    if verbose:
        print(
            "[rf_potential_well] "
            f"eom_factor={eom_factor_potential:.3e} e/s, "
            f"span={time_array[-1] - time_array[0]:.3e} s, "
            f"well min={potential_well.min():.3e} eV, "
            f"well max={potential_well.max():.3e} eV"
        )

    if plot:
        _plot_potential_well(time_array, total_voltage, potential_well)

    return potential_well


def check_single_bucket_well(
    potential_well: NumpyArray,
    *,
    relative_tolerance: float = 1e-2,
    raise_error: bool = True,
) -> bool:
    """
    Check that a potential well is cut to a single RF bucket.

    A single-bucket well decreases from a maximum at the first sample to
    a unique minimum region and rises back to a maximum at the last
    sample: both frame edges must reach the well maximum (within
    tolerance) and no prominent local maximum may exist in between. The
    2D Hamiltonian frame and the action integral are only meaningful on
    such a well; margined frames, multi-bucket spans, tilted
    (accelerating) uncut wells and wells with the minimum on a frame
    edge all violate the condition and would silently corrupt the
    results.

    Parameters
    ----------
    potential_well
        Potential well samples, in [eV]. At least 3 samples, no NaN.
    relative_tolerance
        Tolerance of the edge and interior-prominence criteria, relative
        to the well depth. The default (``1e-2``) accepts sample-aligned
        separatrix cuts of tilted wells (edge mismatch of order
        ``slope * dt``, ~1e-4 to 1e-2 at realistic resolutions) and
        ignores sub-percent numerical wiggles, while still rejecting
        margined frames, multi-bucket spans and edge-split buckets
        (violations there are of order 0.1 to 1).
    raise_error
        If True (default), raise ``ValueError`` on failure; otherwise
        return False.

    Returns
    -------
    is_single_bucket
        True if the well satisfies the single-bucket condition.

    Raises
    ------
    ValueError
        If the well is not a single cut bucket and ``raise_error`` is
        True.
    """
    potential_well = np.asarray(potential_well, dtype=float)

    problems = []
    if potential_well.ndim != 1 or len(potential_well) < 3:
        problems.append(
            "a well needs at least 3 samples in a 1D array "
            f"(got shape {potential_well.shape})"
        )
    elif np.any(np.isnan(potential_well)):
        # NaN compares False everywhere and would silently pass the
        # numeric checks below (and bridge the action integral).
        problems.append("the potential well contains NaN")
    else:
        well_max = float(potential_well.max())
        well_min = float(potential_well.min())
        barrier = well_max - well_min
        if barrier <= 0.0:
            problems.append("the potential well is flat")
        else:
            tolerance = relative_tolerance * barrier
            if (
                potential_well[0] < well_max - tolerance
                or potential_well[-1] < well_max - tolerance
            ):
                problems.append(
                    "the frame edges do not both reach the well "
                    "maximum (margined frame, tilted/accelerating "
                    "well, or bucket split across the frame edges)"
                )
            interior = potential_well[1:-1]
            has_interior_maximum = bool(
                np.any(
                    (interior > potential_well[:-2])
                    & (interior >= potential_well[2:])
                    & (interior > well_min + tolerance)
                )
            )
            if has_interior_maximum:
                problems.append(
                    "a prominent local maximum exists inside the frame "
                    "(multi-bucket span or inner separatrix)"
                )

    if not problems:
        return True
    if raise_error:
        raise ValueError(
            "The potential well is not cut to a single bucket: "
            + "; ".join(problems)
            + ". Cut the well around the separatrix first (well-cut "
            "step, see plan.md). Below transition, remember the "
            "BLonD 2 convention phi_rf=pi (for positive charge) so "
            "the stable phase sits mid-frame."
        )
    return False


def _plot_potential_well(
    time_array: NumpyArray,
    total_voltage: NumpyArray,
    potential_well: NumpyArray,
) -> None:
    """Draw a quick diagnostic figure of the voltage and potential well."""
    import matplotlib.pyplot as plt

    fig, (ax_v, ax_p) = plt.subplots(
        2, 1, sharex=True, num="rf_potential_well"
    )
    time_ns = time_array * 1e9
    ax_v.plot(time_ns, total_voltage / 1e6, color="C0")
    ax_v.set_ylabel("RF voltage [MV]")
    ax_v.grid(alpha=0.3)
    ax_p.plot(time_ns, potential_well, color="C1")
    ax_p.set_xlabel("Time [ns]")
    ax_p.set_ylabel("Potential well [eV]")
    ax_p.grid(alpha=0.3)
    fig.suptitle("Analytic RF potential well")
    fig.tight_layout()
