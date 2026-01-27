# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Functions and classes to interface BLonD with xsuite.

:Authors: **Birk Emil Karlsen-Baeck**, **Thom Arnoldus van Rijswijk**, **Helga Timko**, **Elleanor Lamb**
"""

from collections.abc import Sequence

import numpy as np
import xpart as xp
from numpy.typing import NDArray
from scipy.constants import c
from scipy.constants import c as c_light
from xtrack import Line, Particles, ReferenceEnergyIncrease, ZetaShift

from blond import Beam, SingleHarmonicRFStation
from blond.core.beam.base import BeamBaseClass, BeamFlags
from blond.core.beam.particle_types import ParticleType


def xsuite_to_blond_transform(
    zeta: float | NDArray,
    ptau: float | NDArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """
    Convert Xsuite longitudinal coordinates to BLonD coordinates.

    Parameters
    ----------
    zeta : float or numpy.ndarray
        Longitudinal position in Xsuite coordinates [m].
    ptau : float or numpy.ndarray
        Relative momentum deviation in Xsuite.
    beta0 : float
        Reference relativistic beta.
    energy0 : float
        Reference total energy [eV].
    omega_rf : float
        RF angular frequency [rad/s].
    phi_s : float, optional
        Synchronous phase [rad]. Default is 0.

    Returns
    -------
    dt : float or numpy.ndarray
        Time deviation with respect to the synchronous particle [s].
    dE : float or numpy.ndarray
        Energy deviation with respect to the reference energy [eV].
    """
    dE = ptau * beta0 * energy0
    dt = -zeta / (beta0 * c) + phi_s / omega_rf  # todo
    return dt, dE


def blond_to_xsuite_transform(
    dt: float | NDArray,
    de: float | NDArray,
    beta0: float,
    energy0: float,
    omega_rf: float,
    phi_s: float = 0,
):
    """
    Convert BLonD coordinates to Xsuite coordinates.

    Parameters
    ----------
    dt : float or numpy.ndarray
        Time deviation with respect to the synchronous particle [s].
    de : float or numpy.ndarray
        Energy deviation with respect to the reference energy [eV].
    beta0 : float
        Reference relativistic beta.
    energy0 : float
        Reference total energy [eV].
    omega_rf : float
        RF angular frequency [rad/s].
    phi_s : float, optional
        Synchronous phase [rad]. Default is 0.

    Returns
    -------
    zeta : float or numpy.ndarray
        Longitudinal position in Xsuite coordinates [m].
    ptau : float or numpy.ndarray
        Relative momentum deviation in Xsuite.
    """
    ptau = de / (beta0 * energy0)
    zeta = -(dt - phi_s / omega_rf) * beta0 * c
    return zeta, ptau


def particle_xsuite_to_blond(particle: xp.Particles):
    """
    Construct a BLonD ParticleType from an xpart Particles object.

    This function gets the particle rest mass and charge from an
    `xpart.Particles` instance and uses them to initialise the
    corresponding BLonD `ParticleType`.

    Parameters
    ----------
    particle : xpart.Particles
        Xsuite particles object containing particle properties such as
        rest mass and charge.

    Returns
    -------
    ParticleType
        BLonD particle type with matching mass and charge.
    """
    particle_type_blond = ParticleType(
        mass=float(particle.mass), charge=float(particle.q0)
    )  # todo test check units
    return particle_type_blond


class BLonD3Cavity:
    """
    Wrapper enabling BLonD longitudinal elements to be tracked inside Xsuite.

    This class converts Xsuite particle coordinates to BLonD beam coordinates,
    tracks the beam through a BLonD RF element, and converts the coordinates
    back to Xsuite format.

    Parameters
    ----------
    cavity : SingleHarmonicRFStation
        BLonD RF cavity element providing a `track(beam)` method.
    particles : xtrack.Particles
        Xsuite particles used to initialise the BLonD beam coordinates.
    line : xtrack.Line
        Xsuite line containing the reference particle and machine length.
    initial_intensity : float or int or None, optional
        Initial beam intensity. If None, intensity handling is disabled.
    update_zeta : bool, optional
        Whether to update the Xsuite longitudinal coordinate `zeta`
        after tracking. Default is False.
    """

    def __init__(
        self,
        cavity: SingleHarmonicRFStation,
        particles: Particles,
        line: Line,
        initial_intensity: float | int | None = None,
        update_zeta: bool = False,
    ):
        omega_rf = (
            2
            * np.pi
            * c_light
            * cavity.harmonic
            * line.particle_ref.beta0
            / line.get_length()
        )

        dt, dE = xsuite_to_blond_transform(
            zeta=particles.zeta,
            ptau=particles.ptau,
            beta0=line.particle_ref.beta0,
            energy0=line.particle_ref.energy0,
            omega_rf=omega_rf,
        )
        self.line = line

        particle_type = particle_xsuite_to_blond(self.line.particle_ref)

        beam = Beam(
            intensity=initial_intensity,
            particle_type=particle_type,
        )

        beam.setup_beam(
            dt=[dt],
            dE=[dE],
            reference_time=0,
            reference_total_energy=float(self.line.particle_ref.energy0),
        )

        self.beam = beam
        self.trackable = cavity
        self.update_zeta = update_zeta
        self.orbit_shift = ZetaShift(dzeta=0)

    def track(self, particles: Particles):
        """
        Track particles through the wrapped BLonD element.

        This method:
        1. Converts Xsuite particle coordinates to BLonD beam coordinates.
        2. Calls the BLonD element `track` method.
        3. Converts the updated coordinates back to Xsuite format.

        Parameters
        ----------
        particles : xtrack.Particles
            Xsuite particles to be tracked.
        """
        # Convert xsuite -> blond
        self.xsuite_to_blond_transform(particles, self.beam)

        # todo get new energy from xsuite

        self.trackable._magnetic_cycle.get_target_total_energy.return_value = (
            float(self.line._particle_ref.energy0)
        )  # todo, check for units

        self.trackable.track(self.beam)  # calls the BLonD track method

        # Convert blond -> xsuite
        self.blond_to_xsuite_transform(particles, self.beam)

    def xsuite_to_blond_transform(
        self, particles: Particles, beam: BeamBaseClass
    ):
        """
        Convert Xsuite particle coordinates to BLonD beam coordinates.

        Only active (alive) particles are converted. Lost particles are
        flagged and removed from the BLonD beam representation.

        Parameters
        ----------
        particles : xtrack.Particles
            Xsuite particles providing `zeta` and `ptau`.
        beam : BeamBaseClass
            BLonD beam object whose `dt` and `dE` arrays are updated.
        """
        active_mask = particles.state > 0
        n_active = active_mask.sum()

        # Energy deviation
        dt = beam.write_partial_dt()
        dE = beam.write_partial_dE()
        flags = beam.write_partial_flags()

        dt[:n_active] = -particles.zeta[active_mask] / (
            particles.beta0[active_mask] * c
        )

        dE[:n_active] = (
            particles.beta0[active_mask]
            * particles.energy0[active_mask]
            * particles.ptau[active_mask]
        )

        flags[n_active:] = BeamFlags.LOST.value

        beam.purge_flagged_entries()

        # Particle activity flags
        self._previous_active_mask = active_mask

        # todo check for really killed state and remove intensity effects
        beam.purge_flagged_entries()

    def blond_to_xsuite_transform(
        self, particles: Particles, beam: BeamBaseClass
    ):
        """
        Convert BLonD beam coordinates back to Xsuite particle coordinates.

        Only particles that were active during the last Xsuite-to-BLonD
        transformation are updated.

        Parameters
        ----------
        particles : xtrack.Particles
            Xsuite particles whose `zeta` and `ptau` are updated.
        beam : BeamBaseClass
            BLonD beam object providing updated `dt` and `dE`.
        """
        # Relative energy deviation
        dE = beam.read_partial_dE()

        particles.ptau[self._previous_active_mask] = dE.ravel() / (
            particles.beta0 * particles.energy0
        )

        # Longitudinal position
        dt = beam.read_partial_dt()
        particles.zeta[self._previous_active_mask] = (
            -dt.ravel() * particles.beta0 * c_light
        )


class EnergyUpdate:
    """
    Turn-by-turn reference energy update for Xsuite particles.

    This class applies a reference momentum increment using
    `xtrack.ReferenceEnergyIncrease` based on a predefined momentum
    program. It is intended for use without the BLonD–Xsuite interface.

    Parameters
    ----------
    momentum : Sequence
        Sequence of reference momenta (p0c) for each turn.
    """

    def __init__(self, momentum: Sequence):
        self.momentum = momentum

        init_p0c = self.momentum[1] - self.momentum[0]

        self.xsuite_energy_update = ReferenceEnergyIncrease(Delta_p0c=init_p0c)

    def track(self, particles: Particles):
        """
        Update the reference energy of particles for the current turn.

        The energy increment is computed from the predefined momentum
        sequence and applied to all active particles.

        Parameters
        ----------
        particles : xtrack.Particles
            Xsuite particles whose reference energy is updated.
        """
        mask_alive = particles.state > 0

        # Use the still alive particles to find the current turn momentum
        p0c_before = particles.p0c[mask_alive]

        # Find the momentum for the next turn
        p0c_after = self.momentum[particles.at_turn[mask_alive][0]]

        # Update the energy increment
        self.xsuite_energy_update.Delta_p0c = p0c_after - p0c_before[0]

        # Apply the energy increment to the particles
        self.xsuite_energy_update.track(particles)
