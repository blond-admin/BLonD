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
    dt_margin_percent: float = 0.0,
) -> NumpyArray:
    """
    Uniform time grid spanning one RF period of the main harmonic.

    Parameters
    ----------
    omega_rf
        Angular frequency of the main RF harmonic, in [rad/s].
    n_points
        Number of points in the grid.
    dt_margin_percent
        Fractional margin added on both sides of the RF period, so that
        extrema sitting exactly on the edges remain visible. ``0.4`` adds
        40 % of one RF period, split evenly before and after.

    Returns
    -------
    time_array
        Time coordinates spanning one (optionally margined) RF period, in [s].
    """
    rf_period = 2.0 * np.pi / omega_rf
    margin = dt_margin_percent * rf_period
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

    Integrates the RF voltage over one turn to obtain the effective potential
    landscape the beam experiences, reproducing the BLonD 2
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
        Time coordinates of the voltage waveform, in [s]. May be non-uniform.
    total_voltage
        Total RF voltage summed over all harmonics at ``time_array``, in [V]
        (e.g. ``rf_station.calc_gap_voltage_without_feedbacks(ts=time_array)``).
    charge
        Particle charge, as number of elementary charges ``e``.
    t_rev
        Revolution period, in [s].
    eta_0
        Zeroth-order slippage factor. Only its sign enters the potential.
    energy_gain_per_turn
        Design energy gain per turn, in [eV]. Zero for a coasting/constant
        cycle; non-zero on a ramp, where it tilts the well.
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
            f"eom_factor={eom_factor_potential:.3e} 1/(V.s), "
            f"span={time_array[-1] - time_array[0]:.3e} s, "
            f"well min={potential_well.min():.3e} eV, "
            f"well max={potential_well.max():.3e} eV"
        )

    if plot:
        _plot_potential_well(time_array, total_voltage, potential_well)

    return potential_well


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
