from __future__ import annotations
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        Union,
        TypeVar,
    )

    T = TypeVar("T")

    from numpy.typing import NDArray as NumpyArray

    FloatOrArray = Union[float, NumpyArray]


def calc_beta(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic beta factor (v = beta * c0)

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    beta : float or NDArray
        Relativistic beta factor (unitless), such that v = beta * c
    """
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))


def calc_gamma(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic gamma factor (Lorentz factor)

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    gamma : float or NDArray
        Lorentz factor (unitless)
    """
    my_fraction = momentum / mass
    return np.sqrt(1 + (my_fraction * my_fraction))


def calc_total_energy(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Total relativistic energy of the particle

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    energy : float or NDArray
        Total relativistic energy in [eV]
    """
    return np.sqrt(momentum * momentum + mass * mass)


def calc_energy_kin(mass: float, momentum: FloatOrArray) -> FloatOrArray:
    """
    Relativistic kinetic energy of the particle

    Parameters
    ----------
    mass : float
        Particle mass in [eV/c²]
    momentum : float or NDArray
        Particle momentum in [eV/c]

    Returns
    -------
    kinetic_energy : float or NDArray
        Kinetic energy in [eV], defined as total energy - rest energy
    """
    return calc_total_energy(mass, momentum) - mass


def beta_by_momentum(momentum: T, mass: float) -> T:
    return np.sqrt(1 / (1 + (mass / momentum) ** 2))
