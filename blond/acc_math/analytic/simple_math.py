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

import numba as nb
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, TypeVar

    from numpy.typing import NDArray

    NumpyArray = NDArray[Any]

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


@nb.njit(
    [
        nb.float64(nb.float64, nb.float64),
        nb.float64[:](nb.float64, nb.float64[:]),
        nb.float64[:, :](nb.float64, nb.float64[:, :]),
    ]
)  # pragma: no cover
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


@overload
def momentum_compaction_factor(  # NOQA D103
    transition_gamma: complex,
) -> float: ...


@overload
def momentum_compaction_factor(  # NOQA D103
    transition_gamma: NumpyArray,
) -> NumpyArray: ...


def momentum_compaction_factor(
    transition_gamma,
):
    """
    Calculate the momentum compaction factor.

    Parameters
    ----------
    transition_gamma
        Relativistic gamma of beam transition crossing.

    Returns
    -------
    momentum_compaction_factor
        Momentum compaction factor.
    """
    _assert_purely_real_or_imaginary(transition_gamma)
    momentum_compaction_factor_ = 1 / (transition_gamma * transition_gamma)
    return momentum_compaction_factor_.real


def _assert_purely_real_or_imaginary(val: complex | NumpyArray):
    """
    Assert that a complex number is purely real or purely imaginary.

    A complex number is considered *purely real* if its imaginary part is zero,
    and *purely imaginary* if its real part is zero. This function raises an
    `AssertionError` if the number has both nonzero real and imaginary parts.

    Parameters
    ----------
    val : complex
        Complex number to be validated.

    Raises
    ------
    AssertionError
        If `val` has both real and imaginary parts nonzero.

    Examples
    --------
    >>> _assert_purely_real_or_imaginary(5 + 0j)   # purely real
    >>> _assert_purely_real_or_imaginary(0 + 3j)   # purely imaginary
    >>> _assert_purely_real_or_imaginary(0j)       # zero (both parts 0) is fine
    >>> _assert_purely_real_or_imaginary(2 + 4j)
    Traceback (most recent call last):
        ...
    AssertionError: Expected number with only real or only imaginary part, not (2+4j)
    """
    if np.any((val.real != 0) & (val.imag != 0)):
        raise ValueError(
            f"Expected purely real or purely imaginary number, not {val}."
        )


def gaussian_distribution(
    time_array: NumpyArray, sigma_t: float, center: float
):
    """
    Return a gaussian distribution on a given time array.

    Parameters
    ----------
    time_array
        Time array for which the gaussian distribution is to be calculated.
    sigma_t
        Standard deviation of the gaussian distribution.
    center
        Center of the gaussian distribution.

    Returns
    -------
    gauss
        Gaussian distribution.
    """
    return (
        1
        / (sigma_t * np.sqrt(2 * np.pi))
        * np.exp(-((time_array - center) ** 2) / (2 * sigma_t**2))
    )
