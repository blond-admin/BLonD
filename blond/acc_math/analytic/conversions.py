# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Module to hold simple relativistic conversions.

Notes
-----
Authors:
Simon Lauber
Simon Albright
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.constants import speed_of_light as c0

from blond.core.backends.backend import backend

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray

############################################
############################################
### FUNCTION OVERLOADS FOR TYPE CHECKING ###
###### CONCRETE IMPLEMENTATIONS FOLLOW #####
############################################
############################################


@overload
def magnetic_rigidity_to_momentum(
    magnetic_rigidity: float,
    charge: float,
) -> float: ...


@overload
def magnetic_rigidity_to_momentum(
    magnetic_rigidity: NumpyArray,
    charge: float,
) -> NumpyArray: ...


@overload
def magnetic_rigidity_to_momentum(
    magnetic_rigidity: CupyArray,
    charge: float,
) -> CupyArray: ...


@overload
def beta_to_gamma(beta: float) -> float: ...


@overload
def beta_to_gamma(beta: NumpyArray) -> NumpyArray: ...


@overload
def beta_to_gamma(beta: CupyArray) -> CupyArray: ...


@overload
def gamma_to_beta(gamma: float) -> float: ...


@overload
def gamma_to_beta(gamma: NumpyArray) -> NumpyArray: ...


@overload
def gamma_to_beta(gamma: CupyArray) -> CupyArray: ...


@overload
def frev_to_beta(frev: float, circumference: float) -> float: ...


@overload
def frev_to_beta(frev: NumpyArray, circumference: float) -> NumpyArray: ...


@overload
def frev_to_beta(frev: CupyArray, circumference: float) -> CupyArray: ...


@overload
def beta_to_frev(beta: float, circumference: float) -> float: ...


@overload
def beta_to_frev(beta: NumpyArray, circumference: float) -> NumpyArray: ...


@overload
def beta_to_frev(beta: CupyArray, circumference: float) -> CupyArray: ...


@overload
def beta_to_trev(beta: float, circumference: float) -> float: ...


@overload
def beta_to_trev(beta: NumpyArray, circumference: float) -> NumpyArray: ...


@overload
def beta_to_trev(beta: CupyArray, circumference: float) -> CupyArray: ...


@overload
def momentum_to_beta(momentum: float, rest_mass: float) -> float: ...


@overload
def momentum_to_beta(momentum: NumpyArray, rest_mass: float) -> NumpyArray: ...


@overload
def momentum_to_beta(momentum: CupyArray, rest_mass: float) -> CupyArray: ...


@overload
def momentum_to_gamma(momentum: float, rest_mass: float) -> float: ...


@overload
def momentum_to_gamma(
    momentum: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def momentum_to_gamma(momentum: CupyArray, rest_mass: float) -> CupyArray: ...


@overload
def momentum_to_frev(
    momentum: float, circumference: float, rest_mass: float
) -> float: ...


@overload
def momentum_to_frev(
    momentum: NumpyArray, circumference: float, rest_mass: float
) -> NumpyArray: ...


@overload
def momentum_to_frev(
    momentum: CupyArray, circumference: float, rest_mass: float
) -> CupyArray: ...


@overload
def momentum_to_trev(
    momentum: float, circumference: float, rest_mass: float
) -> float: ...


@overload
def momentum_to_trev(
    momentum: NumpyArray, circumference: float, rest_mass: float
) -> NumpyArray: ...


@overload
def momentum_to_trev(
    momentum: CupyArray, circumference: float, rest_mass: float
) -> CupyArray: ...


@overload
def momentum_to_total_energy(momentum: float, rest_mass: float) -> float: ...


@overload
def momentum_to_total_energy(
    momentum: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def momentum_to_total_energy(
    momentum: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def momentum_to_kinetic_energy(momentum: float, rest_mass: float) -> float: ...


@overload
def momentum_to_kinetic_energy(
    momentum: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def momentum_to_kinetic_energy(
    momentum: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def momentum_to_magnetic_field(
    momentum: float,
    bending_radius: float,
    charge: int,
) -> float: ...


@overload
def momentum_to_magnetic_field(
    momentum: NumpyArray,
    bending_radius: float,
    charge: int,
) -> NumpyArray: ...


@overload
def momentum_to_magnetic_field(
    momentum: CupyArray,
    bending_radius: float,
    charge: int,
) -> CupyArray: ...


@overload
def total_energy_to_momentum(
    total_energy: float, rest_mass: float
) -> float: ...


@overload
def total_energy_to_momentum(
    total_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def total_energy_to_momentum(
    total_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def total_energy_to_kinetic_energy(
    total_energy: float, rest_mass: float
) -> float: ...


@overload
def total_energy_to_kinetic_energy(
    total_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def total_energy_to_kinetic_energy(
    total_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def total_energy_to_magnetic_field(
    total_energy: float,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float: ...


@overload
def total_energy_to_magnetic_field(
    total_energy: NumpyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def total_energy_to_magnetic_field(
    total_energy: CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> CupyArray: ...


@overload
def total_energy_to_beta(total_energy: float, rest_mass: float) -> float: ...


@overload
def total_energy_to_beta(
    total_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def total_energy_to_beta(
    total_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def total_energy_to_gamma(total_energy: float, rest_mass: float) -> float: ...


@overload
def total_energy_to_gamma(
    total_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def total_energy_to_gamma(
    total_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def kinetic_energy_to_momentum(
    kinetic_energy: float, rest_mass: float
) -> float: ...


@overload
def kinetic_energy_to_momentum(
    kinetic_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def kinetic_energy_to_momentum(
    kinetic_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def kinetic_energy_to_total_energy(
    kinetic_energy: float, rest_mass: float
) -> float: ...


@overload
def kinetic_energy_to_total_energy(
    kinetic_energy: NumpyArray, rest_mass: float
) -> NumpyArray: ...


@overload
def kinetic_energy_to_total_energy(
    kinetic_energy: CupyArray, rest_mass: float
) -> CupyArray: ...


@overload
def kinetic_energy_to_magnetic_field(
    kinetic_energy: float,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float: ...


@overload
def kinetic_energy_to_magnetic_field(
    kinetic_energy: NumpyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def kinetic_energy_to_magnetic_field(
    kinetic_energy: CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> CupyArray: ...


@overload
def magnetic_field_to_momentum(
    magnetic_field: float,
    bending_radius: float,
    charge: int,
) -> float: ...


@overload
def magnetic_field_to_momentum(
    magnetic_field: NumpyArray,
    bending_radius: float,
    charge: int,
) -> NumpyArray: ...


@overload
def magnetic_field_to_momentum(
    magnetic_field: CupyArray,
    bending_radius: float,
    charge: int,
) -> CupyArray: ...


@overload
def magnetic_field_to_total_energy(
    magnetic_field: float,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float: ...


@overload
def magnetic_field_to_total_energy(
    magnetic_field: NumpyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def magnetic_field_to_total_energy(
    magnetic_field: CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> CupyArray: ...


@overload
def magnetic_field_to_kinetic_energy(
    magnetic_field: float,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float: ...


@overload
def magnetic_field_to_kinetic_energy(
    magnetic_field: NumpyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def magnetic_field_to_kinetic_energy(
    magnetic_field: CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> CupyArray: ...


@overload
def delta_P_to_delta_E(
    delta_P: float,
    momentum: float,
    rest_mass: float,
) -> float: ...


@overload
def delta_P_to_delta_E(
    delta_P: NumpyArray,
    momentum: NumpyArray,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def delta_P_to_delta_E(
    delta_P: CupyArray,
    momentum: CupyArray,
    rest_mass: float,
) -> CupyArray: ...


@overload
def delta_E_to_delta_P(
    delta_P: float,
    momentum: float,
    rest_mass: float,
) -> float: ...


@overload
def delta_E_to_delta_P(
    delta_P: NumpyArray,
    momentum: NumpyArray,
    rest_mass: float,
) -> NumpyArray: ...


@overload
def delta_E_to_delta_P(
    delta_P: CupyArray,
    momentum: CupyArray,
    rest_mass: float,
) -> CupyArray: ...


############################################
############################################
######### CONCRETE IMPLEMENTATIONS #########
############################################
############################################


def magnetic_rigidity_to_momentum(
    magnetic_rigidity: float | NumpyArray | CupyArray,
    charge: float,
) -> float | NumpyArray | CupyArray:
    r"""
    Convert magnetic rigidity to momentum.

    Parameters
    ----------
    magnetic_rigidity
        Magnetic rigidity :math:`B \rho`, in [Tm].
    charge
        Particle charge, i.e. number of elementary charges `e`.
        Example: For an electron `charge=-1`.

    Returns
    -------
    momentum
        Relativistic momentum, in [eV/c].

    Notes
    -----
    The momentum is calculated using the relation:

    .. math::

        p = B \rho \cdot |q| \cdot c

    where:
        - :math:`p`  is the momentum,
        - :math:`B \rho` is the magnetic rigidity,
        - :math:`q`  is the particle charge in units of `e`,
        - :math:`c` is the speed of light in vacuum.
    """
    return magnetic_rigidity * np.abs(charge) * c0


def beta_to_gamma(
    beta: float | NumpyArray | CupyArray,
) -> float | NumpyArray | CupyArray:
    """
    Convert relativistic beta to gamma.

    Parameters
    ----------
    beta
        Relativistic beta.

    Returns
    -------
    gamma
        Relativistic gamma.
    """
    return 1 / np.sqrt(1 - beta * beta)


def gamma_to_beta(
    gamma: float | NumpyArray | CupyArray,
) -> float | NumpyArray | CupyArray:
    """
    Convert relativistic gamma to beta.

    Parameters
    ----------
    gamma
        Relativistic gamma.

    Returns
    -------
    beta
        Relativistic beta.
    """
    return np.sqrt(1 - 1 / gamma**2)


def frev_to_beta(
    frev: float | NumpyArray | CupyArray, circumference: float
) -> float | NumpyArray | CupyArray:
    """
    Convert revolution frequency to relativistic beta.

    Parameters
    ----------
    frev
        Revolution frequency in [Hz].
    circumference
        Accelerator circumference in [m].

    Returns
    -------
    gamma
        Relativistic gamma.
    """
    return circumference * frev / c0


def beta_to_frev(
    beta: float | NumpyArray | CupyArray, circumference: float
) -> float | NumpyArray | CupyArray:
    """
    Convert relativistic beta to revolution frequency.

    Parameters
    ----------
    beta
        Relativistic beta.
    circumference
        Accelerator circumference in [m].

    Returns
    -------
    frev
        Revolution frequency in [Hz].
    """
    return (beta * c0) / circumference


def beta_to_trev(
    beta: float | NumpyArray | CupyArray, circumference: float
) -> float | NumpyArray | CupyArray:
    """
    Convert relativistic beta to revolution period.

    Parameters
    ----------
    beta
        Relativistic beta.
    circumference
        Accelerator circumference in [m].

    Returns
    -------
    trev
        Revolution period in [s].
    """
    return circumference / (beta * c0)


def momentum_to_beta(
    momentum: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to relativistic beta.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    rest_mass
        Particle rest mass in [eV/c^2].

    Returns
    -------
    trev
        Revolution period in [s].
    """
    return 1 / np.sqrt(1 + rest_mass**2 / momentum**2)


def momentum_to_gamma(
    momentum: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to relativistic beta.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    rest_mass
        Particle rest mass in [eV/c^2].

    Returns
    -------
    gamma
        Relativistic gamma.
    """
    return np.sqrt((momentum / rest_mass) ** 2 + 1)


def momentum_to_frev(
    momentum: float | NumpyArray | CupyArray,
    circumference: float,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to revolution frequency.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    circumference
        Accelerator circumference in [m].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    frev
        Revolution frequency in [Hz].
    """
    beta = momentum_to_beta(momentum, rest_mass)
    return beta_to_frev(beta, circumference)


def momentum_to_trev(
    momentum: float | NumpyArray | CupyArray,
    circumference: float,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to revolution period.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    circumference
        Accelerator circumference in [m].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    trev
        Revolution period in [s].
    """
    beta = momentum_to_beta(momentum, rest_mass)
    return beta_to_trev(beta, circumference)


def momentum_to_total_energy(
    momentum: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to total energy.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    total_energy
        Total energy in [eV].
    """
    return np.sqrt(rest_mass**2 + momentum**2)


def momentum_to_kinetic_energy(
    momentum: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to kinetic energy.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    kinetic_energy
        Kinetic energy in [eV].
    """
    return np.sqrt(rest_mass**2 + momentum**2) - rest_mass


def momentum_to_magnetic_field(
    momentum: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to magnetic field.

    Parameters
    ----------
    momentum
        Momentum in [eV/c].
    bending_radius
        Bending radius in [m].
    charge
        Particle charge in [e].

    Returns
    -------
    magnetic_field
        Magnetic field in [T].
    """
    return momentum / (bending_radius * charge * c0)


def total_energy_to_momentum(
    total_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert total energy to momentum.

    Parameters
    ----------
    total_energy
        Total energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    momentum
        Momentum in [eV/c].
    """
    return np.sqrt(total_energy**2 - rest_mass**2)


def total_energy_to_kinetic_energy(
    total_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert total energy to kinetic energy.

    Parameters
    ----------
    total_energy
        Total energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    kinetic_energy
        Kinetic energy in [eV].
    """
    return total_energy - rest_mass


def total_energy_to_magnetic_field(
    total_energy: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert momentum to magnetic field.

    Parameters
    ----------
    total_energy
        Total energy in [eV].
    bending_radius
        Bending radius in [m].
    charge
        Particle charge in [e].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    magnetic_field
        Magnetic field in [T].
    """
    return np.sqrt(total_energy**2 - rest_mass**2) / (
        bending_radius * charge * c0
    )


def total_energy_to_beta(
    total_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert total energy to relativistic beta.

    Parameters
    ----------
    total_energy
        Total energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    beta
        Relativistic beta.
    """
    return 1 / np.sqrt(1 + rest_mass**2 / (total_energy**2 - rest_mass**2))


def total_energy_to_gamma(
    total_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert total energy to relativistic gamma.

    Parameters
    ----------
    total_energy
        Total energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    gamma
        Relativistic gamma.
    """
    return np.sqrt((total_energy**2 - rest_mass**2) / rest_mass**2 + 1)


def kinetic_energy_to_momentum(
    kinetic_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert kinetic energy to momentum.

    Parameters
    ----------
    kinetic_energy
        Kinetic energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    momentum
        Momentum in [eV/c].
    """
    return np.sqrt((rest_mass + kinetic_energy) ** 2 - rest_mass**2)


def kinetic_energy_to_total_energy(
    kinetic_energy: float | NumpyArray | CupyArray, rest_mass: float
) -> float | NumpyArray | CupyArray:
    """
    Convert kinetic energy to momentum.

    Parameters
    ----------
    kinetic_energy
        Kinetic energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    total_energy
        Total energy in [eV].
    """
    return kinetic_energy + rest_mass


def kinetic_energy_to_magnetic_field(
    kinetic_energy: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert kinetic energy to magnetic field.

    Parameters
    ----------
    kinetic_energy
        Kinetic energy in [eV].
    bending_radius
        Bending radius in [m].
    charge
        Particle change in [e].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    magnetic_field
        Magnetic field in [T].
    """
    return np.sqrt((rest_mass + kinetic_energy) ** 2 - rest_mass**2) / (
        bending_radius * charge * c0
    )


def magnetic_field_to_momentum(
    magnetic_field: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
) -> float | NumpyArray | CupyArray:
    """
    Convert magnetic field to momentum.

    Parameters
    ----------
    magnetic_field
        Magnetic field in [T].
    bending_radius
        Bending radius in [m].
    charge
        Particle change in [e].

    Returns
    -------
    momentum
        Momentum in [eV/c].
    """
    return magnetic_field * bending_radius * charge * c0


def magnetic_field_to_total_energy(
    magnetic_field: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert magnetic field to total energy.

    Parameters
    ----------
    magnetic_field
        Magnetic field in [T].
    bending_radius
        Bending radius in [m].
    charge
        Particle change in [e].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    total_energy
        Total energy in [eV].
    """
    return np.sqrt(
        (magnetic_field * bending_radius * charge * c0) ** 2 + rest_mass**2
    )


def magnetic_field_to_kinetic_energy(
    magnetic_field: float | NumpyArray | CupyArray,
    bending_radius: float,
    charge: int,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert magnetic field to kinetic energy.

    Parameters
    ----------
    magnetic_field
        Magnetic field in [T].
    bending_radius
        Bending radius in [m].
    charge
        Particle change in [e].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    kinetic_energy
        Kinetic energy in [eV].
    """
    return (
        np.sqrt(
            (magnetic_field * bending_radius * charge * c0) ** 2 + rest_mass**2
        )
        - rest_mass
    )


def delta_P_to_delta_E(
    delta_P: float | NumpyArray | CupyArray,
    momentum: float | NumpyArray | CupyArray,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert off-momentum value to off-energy value.

    Parameters
    ----------
    delta_P
        Off-momentum value in [eV/c].
    momentum
        Momentum in [eV/c].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    delta_E
        Off-energy in [eV].
    """
    energy = momentum_to_total_energy(momentum, rest_mass)
    return np.sqrt(rest_mass**2 + (delta_P + momentum) ** 2) - energy


def delta_E_to_delta_P(
    delta_E: float | NumpyArray | CupyArray,
    total_energy: float | NumpyArray | CupyArray,
    rest_mass: float,
) -> float | NumpyArray | CupyArray:
    """
    Convert off-energy value to off-momentum value.

    Parameters
    ----------
    delta_E
        Off-energy value in [eV/c].
    total_energy
        Total energy in [eV].
    rest_mass
        Particle mass in [eV/c^2].

    Returns
    -------
    delta_P
        Off-momentum in [eV/c].
    """
    momentum = total_energy_to_momentum(total_energy, rest_mass)
    return np.sqrt((total_energy + delta_E) ** 2 - rest_mass**2) - momentum
