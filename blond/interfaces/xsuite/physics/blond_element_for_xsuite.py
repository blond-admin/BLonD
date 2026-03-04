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

import numpy as np
import xpart as xp
from numpy.typing import NDArray
from scipy.constants import c
from xtrack import Line, Particles, ZetaShift

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
    dt = -zeta / (beta0 * c) + phi_s / omega_rf
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
        mass=float(particle.mass.item()), charge=float(particle.q0.item())
    )
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
    momentum_compaction_factor : float, optional
        Momentum compaction factor. Default is None. Must be provided if there is an energy ramp.
    """

    def __init__(
        self,
        cavity: SingleHarmonicRFStation,
        particles: Particles,
        line: Line,
        initial_intensity: float | int | None = None,
        momentum_compaction_factor: float | None = None,
    ):
        self.line = line

        self._dt_shift = None
        self._trackable = cavity
        self._orbit_shift = ZetaShift(dzeta=0)

        particle_type = particle_xsuite_to_blond(self.line.particle_ref)

        energy_value = float(self.line.particle_ref.energy0[0])

        mag_cycle = self._trackable._magnetic_cycle

        mag_cycle.get_target_total_energy.return_value = energy_value

        # get the momentum program from BLonD
        if self.line.energy_program is not None:
            # time = self.line.energy_program.t_s

            # must have momentum compaction factor defined
            if momentum_compaction_factor is None:
                raise ValueError(
                    "momentum_compaction_factor must be provided when line has an energy_program."
                )

            self._momentum_compaction_factor = momentum_compaction_factor

        else:
            twiss = self.line.twiss4d()
            self._momentum_compaction_factor = twiss[
                "momentum_compaction_factor"
            ]

        omega_rf = (
            2
            * np.pi
            * c
            * cavity.harmonic
            * float(line.particle_ref.beta0[0])
            / float(line.get_length())
        )

        dt, dE = xsuite_to_blond_transform(
            zeta=particles.zeta,
            ptau=particles.ptau,
            beta0=float(line.particle_ref.beta0[0]),
            energy0=float(line.particle_ref.energy0[0]),
            omega_rf=float(omega_rf),
        )

        beam = Beam(
            intensity=initial_intensity,
            particle_type=particle_type,
        )

        beam.setup_beam(
            dt=dt,
            dE=dE,
            reference_time=0,
            reference_total_energy=float(self.line.particle_ref.energy0[0]),
        )

        self._beam = beam

        self._trackable._magnetic_cycle.get_target_total_energy.return_value = float(
            self.line.particle_ref.energy0[0]
        )

        eta = self._momentum_compaction_factor - (
            1 / (self._beam.reference.gamma**2)
        )
        self._trackable._ring.is_below_transition.return_value = bool(eta < 0)

        # self.set_time_shift() # initial setting of time shift # this was changd

        self.orbit_shift = ZetaShift(dzeta=0.0)

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
        # update time shift
        self.set_time_shift()

        eta = self._momentum_compaction_factor - (
            1 / (self._beam.reference.gamma**2)
        )
        self._trackable._ring.is_below_transition.return_value = bool(eta < 0)

        self.xsuite_to_blond_transform_particles(particles, self._beam)

        p0c_after = self.line.particle_ref.p0c

        mass0 = particles.mass0

        E0_after = np.sqrt(p0c_after**2 + mass0**2)

        # Update BLonD reference energy
        self._trackable._magnetic_cycle.get_target_total_energy.return_value = float(
            E0_after
        )

        self._beam.reference.total_energy = float(E0_after)

        self._trackable.track(self._beam)  # calls the BLonD track method

        # Convert blond -> xsuite
        self.blond_to_xsuite_transform_particles(particles, self._beam)

        self._apply_orbit_shift(particles)

    def set_time_shift(self):
        """
        Calculate the time shift of the BLonD beam coordinates and Xsuite.

        Sets the self.dt_shift attribute.
        """
        omega_rf = self._trackable.calc_main_harmonic_omega_rf_design(
            beam_beta=self._beam.reference.beta,
            ring_circumference=self.line.get_length(),
        )
        phi_s = self._trackable.calc_phi_s_main_harmonic(beam=self._beam)

        self._dt_shift = phi_s / omega_rf  # differs to BLonD 2

    def calc_phi_s(self):
        """
        Calculate the phi_s.

        Returns
        -------
        phi_s
            Phi_s value.
        """
        phi_s = self._trackable.calc_phi_s_main_harmonic(beam=self._beam)
        return phi_s

    def _apply_orbit_shift(self, particles):
        # Ring circumference
        circumference = self.line.get_length()

        # Harmonic number (use your main harmonic getter)
        h = self._trackable.get_main_harmonic()

        # Current beta from updated reference particle
        beta = particles.beta0[particles.state > 0][0]

        # Design RF frequency (harmonic condition)
        omega_rf_design = 2 * np.pi * h * beta * c / circumference

        # Actual RF frequency used in BLonD
        omega_rf = self._trackable.calc_main_harmonic_omega_rf_design(
            beam_beta=beta,
            ring_circumference=circumference,
        )

        # Frequency mismatch
        domega = omega_rf - omega_rf_design

        # Compute dzeta
        dzeta = circumference * domega / omega_rf_design

        # Apply shift
        self.orbit_shift = ZetaShift(dzeta=dzeta)
        self.orbit_shift.track(particles)

    def xsuite_to_blond_transform_particles(
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

        dt[:n_active] = (
            -particles.zeta[active_mask] / (particles.beta0[active_mask] * c)
            + self._dt_shift
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

        beam.purge_flagged_entries()

    def blond_to_xsuite_transform_particles(
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
            particles.beta0[self._previous_active_mask]
            * particles.energy0[self._previous_active_mask]
        )

        # Longitudinal position
        dt = beam.read_partial_dt()

        particles.zeta[self._previous_active_mask] = (
            -(dt.ravel() - self._dt_shift)
            * particles.beta0[self._previous_active_mask]
            * c
        )
