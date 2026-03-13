# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e

from blond.beam_preparation.base import MatchingRoutine
from blond.core.helpers import int_from_float_with_warning
from blond.generals.distributed.helpers import mpi_local_size

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class SynchrotronRadiationMatcher(MatchingRoutine):
    def __init__(
        self,
        n_macroparticles: int | float,
        seed: int | None = 0,
    ) -> None:
        super().__init__()

        self._n_macroparticles_local = mpi_local_size(
            int_from_float_with_warning(
                n_macroparticles, warning_stacklevel=2
            ),
            warning_hint="n_macroparticles",
        )

        self._seed = seed

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with macro-particles.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beam.Beam` object.
        """

        # Get the parameters from the simulation


def match_with_synchrotron_radiation(
    energy,
    ring_circumference,
    momentum_compaction_factor,
    bending_radius,
    total_voltage,
    harmonic,
    n_macroparticles,
    charge=-1,
    mass=m_e * c**2 / e,
    n_sections=1,
    energy_gain_per_turn=0.0,
    seed=None,
):
    """
    This is the functional bit!

    Equilibrium params at the start of the tracking map
    The function now works in the context of having one RF station
    with one RF harmonic and multiple ring sections and the
    synchrotron radiation computed at the end of each section.
    The one-turn map is expected to be (RF + [Drift + SR] * n_sections)

    """

    U0, _, sigma_dE = _calculate_SR_params(
        energy,
        ring_circumference,
        momentum_compaction_factor,
        bending_radius,
        charge,
        mass,
    )

    # Get some base parameters that should be provided by BLonD2/3 objects
    _, beta, _, eta_0, t_rev, t_rf, omega_rf, phi_s = _calculate_base_params(
        energy,
        charge,
        mass,
        ring_circumference,
        momentum_compaction_factor,
        total_voltage,
        harmonic,
        energy_gain_per_turn=energy_gain_per_turn,
    )

    # Compute the expected stable phase offset
    phi_s_offset = np.arcsin(U0 / (charge * total_voltage))
    dt_offset = phi_s_offset / omega_rf

    # Define the Kick Drift parameters
    K_param = -charge * total_voltage * omega_rf * np.cos(phi_s)
    D_param = -t_rev * eta_0 / (beta**2.0 * energy)

    # Compute the Courant-Snyder parameters for the kick drift
    Qs = np.arcsin(np.sqrt(-(D_param * K_param) / 4)) / np.pi
    mu = np.sign(D_param) * 2 * np.pi * Qs
    beta_cs = D_param / np.sin(mu)
    gamma_cs = -K_param / np.sin(mu)
    alpha_cs = np.sign(D_param) * np.tan(np.pi * Qs)

    # Get the longitudinal emittance
    epsilon_rms_tilted = (sigma_dE * energy) ** 2.0 / gamma_cs

    # Get the covariance matrix
    covariance_matrix = epsilon_rms_tilted * np.array(
        [[beta_cs, -alpha_cs], [-alpha_cs, gamma_cs]]
    )

    # Get the "scaled" covariance matrix (NB: multivariate_normal doesn't like big order of magnitude values)
    scaling_factor = 10 ** np.floor(np.log10(np.abs(beta_cs)))
    covariance_matrix_scaled = np.array(covariance_matrix)
    covariance_matrix_scaled[0, 0] /= scaling_factor
    covariance_matrix_scaled[1, 1] *= scaling_factor

    # Generate the random distribution
    rng = np.random.default_rng(seed=seed)
    dt_distrib, dE_distrib = rng.multivariate_normal(
        [0, 0], covariance_matrix_scaled, size=n_macroparticles
    ).T

    # Scale the distribution
    dt_distrib *= np.sqrt(scaling_factor)
    dE_distrib *= np.sqrt(1 / scaling_factor)

    # Position the beam in the stable point in (time, energy)
    dt_distrib += t_rf / 2 + dt_offset
    dE_distrib += -U0 * sawtooth_factor(n_sections)

    # Return the distribution and useful parameters
    equilibrium_params = {
        "covariance_matrix": covariance_matrix,
        "epsilon_rms_tilted": epsilon_rms_tilted,
        "time_offset": t_rf / 2 + dt_offset,
        "energy_offset": -U0 * sawtooth_factor(n_sections),
        "sigma_dt": np.sqrt(covariance_matrix[0, 0]),
        "sigma_dE": np.sqrt(covariance_matrix[1, 1]),
    }

    return np.array(dt_distrib), np.array(dE_distrib), equilibrium_params


def sawtooth_factor(n_sections):
    """The sawtooth factor is the fraction of the total energy loss due to
    synchrotron radiation at which the synchronous energy is sitting right
    before the RF cavity with a single RF station (for the one-turn map
    being (RF + [Drift + SR] * n_sections))

    This will depend on the layout and needs to be generalized.
    """
    return (n_sections + 1) / (2 * n_sections)


def _calculate_base_params(
    energy,
    charge,
    mass,
    ring_circumference,
    momentum_compaction_factor,
    total_voltage,
    harmonic,
    energy_gain_per_turn=0.0,
):
    """
    This already exists in BLonD2, to be ported in BLonD3.
    """
    gamma = energy / mass
    beta = np.sqrt(1.0 - 1.0 / gamma**2.0)
    momentum = beta * energy

    eta_0 = momentum_compaction_factor - 1 / gamma**2

    t_rev = ring_circumference / (beta * c)

    omega_rf = 2 * np.pi / (t_rev) * harmonic
    t_rf = t_rev / harmonic

    phi_s = np.arcsin((charge * energy_gain_per_turn) / total_voltage)

    return gamma, beta, momentum, eta_0, t_rev, t_rf, omega_rf, phi_s


def _calculate_SR_params(
    energy,
    ring_circumference,
    momentum_compaction_factor,
    bending_radius,
    charge=-1,
    mass=m_e * c**2 / e,
):
    """
    This already exists in BLonD2, to be ported in BLonD3.

    This computes the synchrotron radiation parameters for one full turn
    assuming an isomagnetic machine for the synchrotron radiation integrals.

    NB: can be adapted for non-isomagnetic machines with arbitrary radiation integrals
    like it already exists in BLonD2

    """

    # Lorentz factor
    gamma = energy / mass

    # Classical particle radius [m]
    radius_cl = 0.25 / (np.pi * epsilon_0) * e**2 * charge**2 / (mass * e)

    # Sand's radiation constant [ m / eV^3]
    c_gamma = 4 * np.pi / 3 * radius_cl / mass**3

    # Quantum radiation constant [m]
    c_q = 55.0 / (32.0 * np.sqrt(3.0)) * hbar * c / (mass * e)

    # Assuming isomagnetic machine
    I2 = 2.0 * np.pi / bending_radius
    I3 = 2.0 * np.pi / bending_radius**2.0
    I4 = ring_circumference * momentum_compaction_factor / bending_radius**2.0
    jz = 2.0 + I4 / I2

    # Energy loss per turn/RF section [eV]
    U0 = c_gamma * energy**4.0 * I2 / (2.0 * np.pi)

    # Damping time [turns]
    tau_z = 2.0 / jz * energy / U0

    # Equilibrium energy spread
    sigma_dE = np.sqrt(c_q * gamma**2.0 * I3 / (jz * I2))

    return U0, tau_z, sigma_dE
