# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of relativistic equations."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeVar

    from numpy.typing import NDArray as NumpyArray

    T = TypeVar("T")


@overload
def calc_beta(mass: float, momentum: float) -> float: ...


@overload
def calc_beta(mass: float, momentum: NumpyArray) -> NumpyArray: ...


def calc_beta(mass: float, momentum: float | NumpyArray) -> float | NumpyArray:
    """
    Relativistic beta factor (v = beta * c0).

    Parameters
    ----------
    mass
        Particle mass, in [eV/c²].
    momentum
        Particle momentum, in [eV/c].

    Returns
    -------
    beta
        Relativistic beta factor (unitless), such that v = beta * c.
    """
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))


@overload
def calc_gamma(mass: float, momentum: float) -> float: ...


@overload
def calc_gamma(mass: float, momentum: NumpyArray) -> NumpyArray: ...


def calc_gamma(
    mass: float, momentum: float | NumpyArray
) -> float | NumpyArray:
    """
    Relativistic gamma factor (Lorentz factor).

    Parameters
    ----------
    mass
        Particle mass, in [eV/c²].
    momentum
        Particle momentum, in [eV/c].

    Returns
    -------
    gamma
        Lorentz factor (unitless).
    """
    my_fraction = momentum / mass
    return np.sqrt(1 + (my_fraction * my_fraction))


@overload
def calc_total_energy(mass: float, momentum: float) -> float: ...


@overload
def calc_total_energy(mass: float, momentum: NumpyArray) -> NumpyArray: ...


def calc_total_energy(
    mass: float, momentum: float | NumpyArray
) -> float | NumpyArray:
    """
    Total relativistic energy of the particle.

    Parameters
    ----------
    mass
        Particle mass, in [eV/c²].
    momentum
        Particle momentum, in [eV/c].

    Returns
    -------
    energy
        Total relativistic energy, in [eV].
    """
    return np.sqrt(momentum * momentum + mass * mass)


@overload
def calc_energy_kin(mass: float, momentum: float) -> float: ...


@overload
def calc_energy_kin(mass: float, momentum: NumpyArray) -> NumpyArray: ...


def calc_energy_kin(
    mass: float, momentum: float | NumpyArray
) -> float | NumpyArray:
    """
    Relativistic kinetic energy of the particle.

    Parameters
    ----------
    mass
        Particle mass, in [eV/c²].
    momentum
        Particle momentum, in [eV/c].

    Returns
    -------
    kinetic_energy
        Kinetic energy, in [eV], defined as total energy - rest energy.
    """
    return calc_total_energy(mass, momentum) - mass


@overload
def beta_by_momentum(momentum: float, mass: float) -> float: ...


@overload
def beta_by_momentum(momentum: NumpyArray, mass: float) -> NumpyArray: ...


def beta_by_momentum(
    momentum: float | NumpyArray, mass: float
) -> float | NumpyArray:
    """
    Calculate fraction of velocity over speed of light.

    Parameters
    ----------
    momentum
        Particle momentum, in [eV/c].
    mass
        Particle mass, in [eV/c²].

    Returns
    -------
    beta
        Fraction of velocity over speed of light.

    Notes
    -----
    Internal assumption is :math:`c_0=1`.
    """
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))
