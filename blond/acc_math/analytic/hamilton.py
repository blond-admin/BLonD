# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of equations to deal with a single RF Hamiltonian."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.constants import speed_of_light as c  # type: ignore[import-untyped]

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond import Ring
    from blond.cycles.magnetic_cycle import MagneticCycleBase
    from blond.physics.cavities import RfManipulationBaseClass


def is_in_separatrix(
    charge: float,
    harmonic: float,
    voltage: float,
    omega_rf: float,
    phi_rf_d: float,
    phi_s: float,
    etas: list[float],
    beta: float,
    total_energy: float,
    ring_circumference: float,
    dt: NumpyArray,
    dE: NumpyArray,
) -> NumpyArray:
    r"""
    Function checking whether coordinate `dt` & `dE` are inside the separatrix.

    Uses the single-RF sinusoidal Hamiltonian.

    Parameters
    ----------
    charge
        Particle charge, as number of elementary charges `e` [].
    harmonic
        RF Harmonic, i.e. number of RF cycles per synchrotron turn.
    voltage
        RF voltage of the RF station, in [V].
    omega_rf
        Angular frequency of the RF system, in [rad/s].
    phi_rf_d
        Design phase, in [rad].
    phi_s
        Stable phase, in [rad].
    etas
        Drift in arc parameter eta for one turn in synchrotron.
    beta
        Beam reference fraction of speed of light (v/c0).
    total_energy
        Total energy of the reference beam (global total energy), in [eV].
    ring_circumference
        One turn length of the beam, in [m].
    dt
        Macro-particle time coordinates, in [s].
    dE
        Macro-particle energy coordinates, in [eV].

    Returns
    -------
    is_in_separatrix
        An array mask, where 1 means that a particle is inside the separatrix.
    """
    dt_sep = (np.pi - phi_s - phi_rf_d) / omega_rf

    Hsep = single_rf_sin_hamiltonian(
        charge=charge,
        harmonic=harmonic,
        voltage=voltage,
        omega_rf=omega_rf,
        phi_rf_d=phi_rf_d,
        phi_s=phi_s,
        etas=etas,
        beta=beta,
        total_energy=total_energy,
        ring_circumference=ring_circumference,
        dt=dt_sep,
        dE=0,
    )
    is_in_separatrix_ = np.fabs(
        single_rf_sin_hamiltonian(
            charge=charge,
            harmonic=harmonic,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf_d=phi_rf_d,
            phi_s=phi_s,
            etas=etas,
            beta=beta,
            total_energy=total_energy,
            ring_circumference=ring_circumference,
            dt=dt,
            dE=dE,
        )
    ) < np.fabs(Hsep)

    return is_in_separatrix_


@overload
def phase_modulo_above_transition(phi: float) -> float: ...


@overload
def phase_modulo_above_transition(phi: NumpyArray) -> NumpyArray: ...


def phase_modulo_above_transition(
    phi: float | NumpyArray,
) -> float | NumpyArray:
    r"""
    Wrap phase values into the range :math:`[0, 2\pi)`.

    This function projects scalar or array phase values (in radians)
    into the range from :math:`0` (inclusive) to :math:`2\pi` (exclusive),
    ensuring continuity across multiples of :math:`2\pi`.

    Parameters
    ----------
    phi : float or ndarray
        Input phase value(s) in radians. Can be a scalar or a NumPy array.

    Returns
    -------
    phi_corrected
        Phase value(s) wrapped into the range :math:`[0, 2\pi)`.

    Notes
    -----
    This operation performs a modulo of :math:`2\pi` such that negative phase
    values are shifted into the positive domain.

    The transformation is defined as:

    .. math::

        \phi_{corrected} = \phi - 2\pi \left\lfloor \frac{\phi}{2\pi} \right\rfloor

    Examples
    --------
    >>> import numpy as np
    >>> from blond.acc_math.analytic.hamilton import phase_modulo_above_transition
    >>> phase_modulo_above_transition(-np.pi / 2)
    np.float64(4.71238898038469)
    >>> phase_modulo_above_transition(3 * np.pi)
    np.float64(3.141592653589793)
    >>> phi = np.linspace(-10, 10, 5)
    >>> phi_limited = phase_modulo_above_transition(phi)
    """
    return phi - 2.0 * np.pi * np.floor(phi / (2.0 * np.pi))


@overload
def phase_modulo_below_transition(phi: float) -> float: ...


@overload
def phase_modulo_below_transition(phi: NumpyArray) -> NumpyArray: ...


def phase_modulo_below_transition(
    phi: float | NumpyArray,
) -> float | NumpyArray:
    r"""
    Wrap phase values into the range :math:`[0, 2\pi)`.

    This function projects scalar or array phase values (in radians)
    into the range from :math:`0` (inclusive) to :math:`2\pi` (exclusive),
    ensuring continuity across multiples of :math:`2\pi`.

    Parameters
    ----------
    phi : float or ndarray
        Input phase value(s) in radians. Can be a scalar or a NumPy array.

    Returns
    -------
    phi_corrected
        Phase value(s) wrapped into the range :math:`[0, 2\pi)`.

    Notes
    -----
    This operation performs a modulo of :math:`2\pi` such that negative phase
    values are shifted into the positive domain.

    The transformation is defined as:

    .. math::

        \phi_{corrected} = \phi - 2\pi \left\lfloor \frac{\phi}{2\pi} \right\rfloor

    Examples
    --------
    >>> import numpy as np
    >>> phase_modulo_above_transition(-np.pi / 2)
    4.71238898038469
    >>> phase_modulo_above_transition(3 * np.pi)
    3.141592653589793
    >>> phi = np.linspace(-10, 10, 5)
    >>> phase_modulo_above_transition(phi)
    """
    return phi - 2.0 * np.pi * (np.floor(phi / (2.0 * np.pi) + 0.5))


def single_rf_sin_hamiltonian(
    charge: float,
    harmonic: float,
    voltage: float,
    omega_rf: float,
    phi_rf_d: float,
    phi_s: float,
    etas: list[float],
    beta: float,
    total_energy: float,
    ring_circumference: float,
    dt: float | NumpyArray,
    dE: float | NumpyArray,
) -> float | NumpyArray:
    """
    Single RF sinusoidal Hamiltonian.

    Parameters
    ----------
    charge
        Particle charge, as number of elementary charges `e` [].
    harmonic
        RF Harmonic, i.e. number of RF cycles per synchrotron turn.
    voltage
        RF voltage of the RF station, in [V].
    omega_rf
        Angular frequency of the RF system, in [rad/s].
    phi_rf_d
        Design phase, in [rad].
    phi_s
        Stable phase, in [rad].
    etas
        Drift in arc parameter eta for one turn in synchrotron.
    beta
        Beam reference fraction of speed of light (v/c0).
    total_energy
        Total energy of the reference beam (global total energy), in [eV].
    ring_circumference
        One turn length of the beam, in [m].
    dt
        Macro-particle time coordinates, in [s].
    dE
        Macro-particle energy coordinates, in [eV].

    Returns
    -------
    hamiltonians
        Hamiltonian values at dt and dE.
    """
    h0 = harmonic
    V0 = float(voltage * charge)

    delta = dE / (beta**2 * total_energy)
    eta_tracking = sum([eta_i * (delta**i) for i, eta_i in enumerate(etas)])

    c1 = eta_tracking * c * np.pi / (ring_circumference * beta * total_energy)
    c2 = c * beta * V0 / (h0 * ring_circumference)

    phi_s_ = phi_s
    phi_b = omega_rf * dt + phi_rf_d

    eta0 = etas[0]

    # Modulo 2 Pi of bunch phase
    if eta0 < 0:
        phi_b = phase_modulo_below_transition(phi_b)
    elif eta0 > 0:
        phi_b = phase_modulo_above_transition(phi_b)

    return c1 * dE**2 + c2 * (
        np.cos(phi_b) - np.cos(phi_s_) + (phi_b - phi_s_) * np.sin(phi_s_)
    )


def calc_phi_s_single_harmonic(
    charge: float,
    voltage: float,
    phase: float,
    energy_gain: float,
    above_transition: bool,
) -> float:
    """
    Derive the analytical synchronous phase for a single harmonic RF.

    Parameters
    ----------
    charge
        Particle charge, i.e. number of elementary charges `e`
        Example: For an electron `charge=-1`.
    voltage
        RF voltage of the RF station, in [V].
    phase
        Phi_rf of the main harmonic, in [rad].
    energy_gain
        Energy gain per turn, in [eV].
    above_transition
        Whether the beam energy is below or above transition.

    Returns
    -------
    phi_s
        The synchronous phase, in [rad].
    """
    phi = np.arcsin(energy_gain / (voltage * charge))

    negative_charge = charge < 0  # for readability

    if above_transition != negative_charge:
        # Only if one of both conditions is met.
        # Otherwise, they cancel each other out like ``-1 * -1 = 1``
        phi = np.pi - phi

    return phi - phase


def separatrix_single_rf(
    rf_station: RfManipulationBaseClass,
    magnetic_cylce: MagneticCycleBase,
    ring: Ring,
    dt_array: NumpyArray,
    turn_number: int,
) -> float | NumpyArray:
    """
    Derive the analytical separatrix for a single harmonic RF.

    Parameters
    ----------
    rf_station
        The RF station object.
    magnetic_cylce
        The magnetic cycle object defining energy evolution.
    ring
        The ring object.
    dt_array
        Array of time coordinates at which to sample the separatrix, in [s].
    turn_number
        Turn at which to calculate the separatrix for.

    Returns
    -------
    phi_array
        The array containing the phases equivalent to the dt_array, in [rad].

    separatrix_array
        The separatrix values at each point in dt_array.
    """
    voltage = rf_station.voltage
    harmonic = rf_station.harmonic
    phi_rf = rf_station.phi_rf

    charge = magnetic_cylce.reference_particle.charge

    if hasattr(magnetic_cylce, "_values_after_turn"):
        energy_array = magnetic_cylce._values_after_turn
        energy_gain = energy_array[1] - energy_array[0]
        energy = energy_array[turn_number]
    else:
        energy_gain = 0
        energy = magnetic_cylce._value

    circumference = ring.circumference

    reference_total_energy = magnetic_cylce.get_target_total_energy(
        particle_type=magnetic_cylce.reference_particle,
        turn_i=turn_number,
        section_i=0,
        reference_time=0,
    )

    reference_gamma = (
        reference_total_energy * magnetic_cylce.reference_particle.mass_inv
    )

    beta = np.sqrt(1.0 - 1.0 / (reference_gamma * reference_gamma))
    reference_velocity = beta * c

    t_rev = circumference / reference_velocity

    omega_0 = (2 * np.pi) / t_rev
    omega_rf = omega_0 * harmonic

    eta = ring.calc_average_eta_0(reference_gamma)

    above_Transition = False
    if eta > 0:
        above_Transition = True

    phi_s = calc_phi_s_single_harmonic(
        charge=charge,
        voltage=voltage,
        phase=phi_rf,
        energy_gain=energy_gain,
        above_transition=above_Transition,
    )

    phi = dt_array * omega_rf

    potential_energy = (
        np.cos(np.pi - phi_s)
        - np.cos(phi)
        + (np.pi - phi - phi_s) * np.sin(phi_s)
    )

    mask_potential = potential_energy > 0
    potential_energy = potential_energy[mask_potential]
    dt_array = dt_array[mask_potential]

    separatrix = np.sqrt(
        2
        * charge
        * voltage
        * beta**2
        * energy
        / (np.pi * harmonic * np.abs(eta))
        * potential_energy
    )

    separatrix_array = np.concatenate((separatrix, -separatrix[::-1]))
    phi_array = np.concatenate((dt_array, dt_array[::-1]))

    return phi_array, separatrix_array
