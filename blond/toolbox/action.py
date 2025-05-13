# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Phase-space variable conversions related to action. Single-RF case without
intensity effects is considered.**

:Authors: **Helga Timko**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ellipe, ellipk

from ..input_parameters.rf_parameters import RFStation
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray, ArrayLike

    from ..input_parameters.ring import Ring
    from ..input_parameters.rf_parameters import RFStation


def x(phimax: ArrayLike) -> ArrayLike:
    return np.sin(0.5 * phimax)


def x2(phimax: ArrayLike) -> ArrayLike:
    return np.sin(0.5 * phimax) ** 2


def action_from_phase_amplitude(x2: NumpyArray) -> NumpyArray:
    """
    Returns the relative action for given oscillation amplitude in time.
    Action is normalised to the value at the separatrix, given in units of 1.
    """

    action = np.zeros(len(x2))

    indices = np.where(x2 != 1.0)[0]
    indices0 = np.where(x2 == 1.0)[0]
    action[indices] = ellipe(x2[indices]) - (1.0 - x2[indices]) * ellipk(
        x2[indices]
    )

    if indices0:
        action[indices0] = float(ellipe(x2[indices0]))

    return action


def tune_from_phase_amplitude(phimax: float) -> float:
    """
    Find the tune w.r.t. the central synchrotron frequency corresponding to a
    given amplitude of synchrotron oscillations in phase
    """

    return 0.5 * np.pi / ellipk(x(phimax))


def phase_amplitude_from_tune(tune: NumpyArray) -> NumpyArray:
    """
    Find the amplitude of synchrotron oscillations in phase corresponding to a
    given tune w.r.t. the central synchrotron frequency
    """

    n = len(tune)
    phimax = np.zeros(n)

    for i in range(n):
        if tune[i] == 1.0:
            phimax[i] = 0.0

        elif tune[i] == 0.0:
            phimax[i] = np.pi

        else:
            guess = 0.5 * np.pi
            difference = 0.25 * np.pi
            k = 0

            while (
                np.fabs(tune[i] - tune_from_phase_amplitude(guess)) / tune[i]
                > 0.001
                and np.fabs(difference / guess) > 1.0e-10
            ):
                guess += (
                    np.sign(tune_from_phase_amplitude(guess) - tune[i])
                    * difference
                )
                difference *= 0.5
                k += 1
                if k > 100:
                    # PhaseSpaceError
                    raise RuntimeError(
                        "Exceeded maximum number of "
                        + "iterations in "
                        + "phase_amplitude_from_tune()!"
                    )

            phimax[i] = guess

    return phimax


@handle_legacy_kwargs
def oscillation_amplitude_from_coordinates(
    ring: Ring,
    rf_station: RFStation,
    dt: NumpyArray,
    dE: NumpyArray,
    timestep: int = 0,
    Np_histogram: Optional[NumpyArray] = None,
):
    """
    Returns the oscillation amplitude in time for given particle coordinates,
    assuming single-harmonic RF system and no intensity effects.
    Optional: RF parameters at a given timestep (default = 0) are used.
    Optional: Number of points for histogram output
    """

    omega_rf = rf_station.omega_rf[0, timestep]
    phi_rf = rf_station.phi_rf[0, timestep]
    phi_s = rf_station.phi_s[timestep]
    eta = rf_station.eta_0[0]
    T0 = ring.t_rev[0]
    V = rf_station.voltage[0, 0]
    beta_sq = rf_station.beta[0] ** 2
    E = rf_station.energy[0]
    const = eta * T0 * omega_rf / (2.0 * V * beta_sq * E)

    dtmax = (
        np.fabs(
            np.arccos(np.cos(omega_rf * dt + phi_rf) + const * dE**2)
            - phi_rf
            - phi_s
        )
        / omega_rf
    )

    if Np_histogram is not None:
        histogram, bins = np.histogram(
            dtmax, Np_histogram, (0, np.pi / omega_rf)
        )
        histogram = np.double(histogram) / np.sum(histogram[:])
        bin_centres = 0.5 * (bins[0:-1] + bins[1:])

        return dtmax, bin_centres, histogram

    else:
        return dtmax


@handle_legacy_kwargs
def action_from_oscillation_amplitude(
    rf_station: RFStation,
    dtmax: float,
    timestep: int = 0,
    Np_histogram: Optional[NumpyArray] = None,
):
    """
    Returns the relative action for given oscillation amplitude in time,
    assuming single-harmonic RF system and no intensity effects.
    Action is normalised to the value at the separatrix, given in units of 1.
    Optional: RF parameters at a given timestep (default = 0) are used.
    Optional: Number of points for histogram output
    """

    omega_rf = rf_station.omega_rf[0, timestep]
    xx = x2(omega_rf * dtmax)
    action = np.zeros(len(xx))

    indices = np.where(xx != 1.0)[0]
    indices0 = np.where(xx == 1.0)[0]
    action[indices] = ellipe(xx[indices]) - (1.0 - xx[indices]) * ellipk(
        xx[indices]
    )
    if indices0:
        action[indices0] = float(ellipe(xx[indices0]))

    if Np_histogram is not None:
        histogram, bins = np.histogram(action, Np_histogram, (0, 1))
        histogram = np.double(histogram) / np.sum(histogram[:])
        bin_centres = 0.5 * (bins[0:-1] + bins[1:])

        return action, bin_centres, histogram

    else:
        return action
