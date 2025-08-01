from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import speed_of_light as c

if TYPE_CHECKING:
    from typing import List
    from numpy.typing import NDArray as NumpyArray


def is_in_separatrix(
    charge: float,
    harmonic: float,
    voltage: float,
    omega_rf: float,
    phi_rf_d: float,
    phi_s: float,
    etas: List[float],
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
        Particle charge, as number of elementary charges `e` []
    harmonic
        RF Harmonic, i.e. number of RF cycles per synchrotron turn
    voltage
        RF voltage of the cavity, in [V]
    omega_rf
        Angular frequency of the RF system, in [rad/s]
    phi_rf_d
        Design phase, in [rad]
    phi_s
        Stable phase, in [rad]
    etas
        Drift in arc parameter eta for one turn in synchrotron # TODO unit
    beta
        Beam reference fraction of speed of light (v/c0)
    total_energy
        Total energy of the reference beam (global total energy), in [eV]
    ring_circumference
        One turn length of the beam, in [m]
    dt
        Macro-particle time coordinates, in [s]
    dE
        Macro-particle energy coordinates, in [eV]

    Returns
    -------

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


def phase_modulo_above_transition(phi: NumpyArray) -> NumpyArray:
    """
    Projects a phase array into the range -Pi/2 to +3*Pi/2.

    Parameters
    ----------
    phi
        Phase, in [rad]

    Returns
    -------
    phi_corrected
         Phase array into the range -Pi/2 to +3*Pi/2.

    """

    return phi - 2.0 * np.pi * np.floor(phi / (2.0 * np.pi))


def phase_modulo_below_transition(phi: NumpyArray) -> NumpyArray:
    """
    Projects a phase array into the range -Pi/2 to +3*Pi/2.

    Parameters
    ----------
    phi
        Phase, in [rad]

    Returns
    -------
    phi_corrected
         Phase array into the range -Pi/2 to +3*Pi/2.

    """

    return phi - 2.0 * np.pi * (np.floor(phi / (2.0 * np.pi) + 0.5))


def single_rf_sin_hamiltonian(
    charge: float,
    harmonic: float,
    voltage: float,
    omega_rf: float,
    phi_rf_d: float,
    phi_s: float,
    etas: List[float],
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
        Particle charge, as number of elementary charges `e` []
    harmonic
        RF Harmonic, i.e. number of RF cycles per synchrotron turn
    voltage
        RF voltage of the cavity, in [V]
    omega_rf
        Angular frequency of the RF system, in [rad/s]
    phi_rf_d
        Design phase, in [rad]
    phi_s
        Stable phase, in [rad]
    etas
        Drift in arc parameter eta for one turn in synchrotron
    beta
        Beam reference fraction of speed of light (v/c0)
    total_energy
        Total energy of the reference beam (global total energy), in [eV]
    ring_circumference
        One turn length of the beam, in [m]
    dt
        Macro-particle time coordinates, in [s]
    dE
        Macro-particle energy coordinates, in [eV]

    Returns
    -------
    hamiltonians
        Hamiltonian values at dt and dE

    """
    h0 = harmonic
    V0 = float(voltage * charge)

    delta = dE / (beta**2 * total_energy)
    eta_tracking = sum([eta_i * (delta**i) for i, eta_i in enumerate(etas)])

    c1 = eta_tracking * c * np.pi / (ring_circumference * beta * total_energy)
    c2 = c * beta * V0 / (h0 * ring_circumference)

    phi_s = phi_s
    phi_b = omega_rf * dt + phi_rf_d

    eta0 = etas[0]

    # Modulo 2 Pi of bunch phase
    if eta0 < 0:
        phi_b = phase_modulo_below_transition(phi_b)
    elif eta0 > 0:
        phi_b = phase_modulo_above_transition(phi_b)

    return c1 * dE**2 + c2 * (
        np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s)
    )


def calc_phi_s_single_harmonic(
    charge: float,
    voltage: float,
    phase: float,
    energy_gain: float,
    above_transition: bool,
) -> float:
    phi = np.arcsin(energy_gain / (voltage * charge))
    if above_transition:
        phi = np.pi - phi
    return phi - phase
