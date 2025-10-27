"""Collection of equations to deal with a single RF Hamiltonian."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import speed_of_light as c

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def is_in_separatrix(
    charge: np.float32 | np.float64,
    harmonic: np.float32 | np.float64,
    voltage: np.float32 | np.float64,
    omega_rf: np.float32 | np.float64,
    phi_rf_d: np.float32 | np.float64,
    phi_s: np.float32 | np.float64,
    etas: list[np.float32 | np.float64],
    beta: np.float32 | np.float64,
    total_energy: np.float32 | np.float64,
    ring_circumference: np.float32 | np.float64,
    dt: NumpyArray,
    dE: NumpyArray,
) -> NumpyArray:
    r"""Function checking whether coordinate `dt` & `dE` are inside the separatrix.

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


def phase_modulo_above_transition(phi: NumpyArray) -> NumpyArray:
    """Projects a phase array into the range -Pi/2 to +3*Pi/2.

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
    """Projects a phase array into the range -Pi/2 to +3*Pi/2.

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
    charge: np.float32 | np.float64,
    harmonic: np.float32 | np.float64,
    voltage: np.float32 | np.float64,
    omega_rf: np.float32 | np.float64,
    phi_rf_d: np.float32 | np.float64,
    phi_s: np.float32 | np.float64,
    etas: list[np.float32 | np.float64],
    beta: np.float32 | np.float64,
    total_energy: np.float32 | np.float64,
    ring_circumference: np.float32 | np.float64,
    dt: np.float32 | np.float64 | NumpyArray,
    dE: np.float32 | np.float64 | NumpyArray,
) -> np.float32 | np.float64 | NumpyArray:
    """Single RF sinusoidal Hamiltonian.

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
    charge: np.float32 | np.float64,
    voltage: np.float32 | np.float64,
    phase: np.float32 | np.float64,
    energy_gain: np.float32 | np.float64,
    above_transition: bool,
) -> np.float32 | np.float64:
    """Derives the analytical synchronous phase for a single harmonic RF.

    Parameters
    ----------
    charge
        Particle charge, i.e. number of elementary charges `e`
        Example: For an electron `charge=-1`.
    voltage
        RF voltage of the cavity, in [V].
    phase
        phi_rf of the main harmonic, in [rad].
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
    if above_transition:
        phi = np.pi - phi
    return phi - phase
