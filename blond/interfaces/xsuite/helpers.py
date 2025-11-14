"""
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**, **Elleanor Lamb**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.constants import c as clight

if TYPE_CHECKING:
    from numpy.typing import NDArray


def blond_to_xsuite_transform(
    dt: float | NDArray,
    de: float | NDArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """Coordinate transformation from Xsuite to BLonD.

    The coordinates are transformed in the following way

    .. math::

        p_{\tau} = \frac{\Delta E}{\beta_s E_s}

    .. math::

        \zeta = - \left ( \Delta t - \frac{\phi_s}{\omega_\text{rf}} \right) \beta_s c

    Parameters
    ----------
    dt : float or NDArray
        The deviation in time [s] from the reference clock in BLonD.
    de : float or NDArray
        The deviation in energy [eV] from the synchronous particle.
    beta0 : float
        Synchronous beta [-].
    energy0 : float
        Synchronous energy [eV].
    omega_rf : float
        The rf angular frequency [rad/s].
    phi_s : float
        Synchronous phase [rad] in radians equivalent to Xsuite's :math:`\phi_\text{rf}`
        (below transition energy input should be :math:`\phi_s - \phi_\text{rf}`). The default value is 0.

    Returns
    -------
    zeta : numpy-arrays (or single variable)
        The xsuite longitudinal coordinate [m].
    ptau : numpy-arrays (or single variable)
        The xsuite longitudinal momentum [-].
    """

    ptau = de / (beta0 * energy0)
    zeta = -(dt - phi_s / omega_rf) * beta0 * clight
    return zeta, ptau


def xsuite_to_blond_transform(
    zeta: float | NDArray,
    ptau: float | NDArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """Coordinate transformation from Xsuite to BLonD.

    The coordinates are transformed as

    .. math::

        \Delta E = p_{\tau} \beta_s c

    .. math::

        \Delta t = \frac{\zeta}{\beta_s c} + \frac{\phi_s}{\omega_\text{rf}}

    Parameters
    ----------
    zeta : float or numpy-array
        The zeta coordinate [m] as defined in Xsuite.
    ptau : float or numpy-array
        The ptau coordinate [-] as defined in Xsuite.
    beta0 : float
        The synchronous beta [-].
    energy0 : float
        The synchronous energy [eV].
    omega_rf : float
        The rf angular frequency [rad/s].
    phi_s : float
        The synchronous phase [rad] in radians equivalent to Xsuite's :math:`\phi_\text{rf}`
        (below transition energy input should be :math:`\phi_s - \phi_\text{rf}`)

    Returns
    -------
    dt : numpy-arrays (or single variable)
        The BLonD longitudinal coordinate [s].
    dE : numpy-arrays (or single variable)
        The BLonD longitudinal energy coordinate [eV].
    """

    dE = ptau * beta0 * energy0
    dt = -zeta / (beta0 * clight) + phi_s / omega_rf
    return dt, dE
