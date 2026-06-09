# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Coordinate and particle-state conversion between BLonD and xsuite.

BLonD describes the longitudinal phase space with ``(dt, dE)`` (time deviation
[s] and energy deviation [eV] with respect to the synchronous particle), while
xsuite uses ``(zeta, ptau)`` (longitudinal position [m] and normalised momentum
deviation).

All conversions are parameterised by a :class:`ReferenceFrame` carrying the
per-turn reference quantities (``beta0``, ``energy0``). The host tracking loop
owns this frame: xsuite when a BLonD element runs inside an ``xtrack.Line``,
BLonD otherwise.

Conventions (``c`` = speed of light)::

    dt   = -zeta / (beta0 * c)
    zeta = -dt   * (beta0 * c)
    dE   =  ptau * beta0 * energy0
    ptau =  dE   / (beta0 * energy0)

The mapping ``zeta ↔ dt`` is purely kinematic and does not involve any RF
frequency: xsuite's ``zeta`` is the position offset from the reference particle
in its co-moving frame. The reference particle in xsuite plays the role of the
synchronous particle in BLonD; an RF cavity's frequency, if any, only enters
inside the cavity itself when it computes a phase at a given time of flight,
not in the coordinate definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.constants import c

from blond.core.beam.flags import BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from blond.core.beam.base import BeamBaseClass


@dataclass(frozen=True)
class ReferenceFrame:
    """
    Per-turn reference quantities linking BLonD and xsuite coordinates.

    Attributes
    ----------
    beta0
        Reference relativistic beta [1].
    energy0
        Reference total energy [eV].
    """

    beta0: float
    energy0: float


def zeta_to_dt(zeta, frame: ReferenceFrame):
    """
    Convert xsuite ``zeta`` [m] to BLonD ``dt`` [s].

    Parameters
    ----------
    zeta
        Longitudinal position offset [m].
    frame
        Reference frame linking the two coordinate systems.

    Returns
    -------
    dt
        Time deviation [s].
    """
    return -zeta / (frame.beta0 * c)


def dt_to_zeta(dt, frame: ReferenceFrame):
    """
    Convert BLonD ``dt`` [s] to xsuite ``zeta`` [m].

    Parameters
    ----------
    dt
        Time deviation [s].
    frame
        Reference frame linking the two coordinate systems.

    Returns
    -------
    zeta
        Longitudinal position offset [m].
    """
    return -dt * frame.beta0 * c


def ptau_to_dE(ptau, frame: ReferenceFrame):
    """
    Convert xsuite ``ptau`` [1] to BLonD ``dE`` [eV].

    Parameters
    ----------
    ptau
        Normalised momentum deviation [1].
    frame
        Reference frame linking the two coordinate systems.

    Returns
    -------
    dE
        Energy deviation [eV].
    """
    return ptau * frame.beta0 * frame.energy0


def dE_to_ptau(dE, frame: ReferenceFrame):
    """
    Convert BLonD ``dE`` [eV] to xsuite ``ptau`` [1].

    Parameters
    ----------
    dE
        Energy deviation [eV].
    frame
        Reference frame linking the two coordinate systems.

    Returns
    -------
    ptau
        Normalised momentum deviation [1].
    """
    return dE / (frame.beta0 * frame.energy0)


def particles_to_beam(particles, beam: BeamBaseClass, frame: ReferenceFrame):
    """
    Write xsuite particle coordinates into a BLonD beam (index-aligned).

    Slot ``i`` of the beam corresponds to particle ``i``: coordinates are
    converted in place and each slot's flag is set to ``ACTIVE`` if the xsuite
    particle is alive (``state > 0``) or ``LOST`` otherwise. The beam must have
    the same number of slots as ``particles``.

    Parameters
    ----------
    particles
        An xsuite ``Particles`` providing ``zeta``, ``ptau`` and ``state``.
    beam
        BLonD beam whose ``dt``/``dE``/flag arrays are updated in place.
    frame
        Reference frame linking the two coordinate systems.

    Returns
    -------
    active
        Boolean mask of alive particles (``state > 0``), for symmetric
        write-back via :func:`beam_to_particles`.
    """
    active = particles.state > 0
    dt = beam.write_partial_dt()
    dE = beam.write_partial_dE()
    flags = beam.write_partial_flags()

    dt[:] = zeta_to_dt(particles.zeta, frame)
    dE[:] = ptau_to_dE(particles.ptau, frame)
    flags[active] = BeamFlags.ACTIVE.value
    flags[~active] = BeamFlags.LOST.value
    return active


def beam_to_particles(
    beam: BeamBaseClass,
    particles,
    frame: ReferenceFrame,
    active: NDArray | None = None,
) -> None:
    """
    Write BLonD beam coordinates back into xsuite particles (index-aligned).

    Only alive particles are updated; lost particles keep their coordinates.

    Parameters
    ----------
    beam
        BLonD beam providing updated ``dt``/``dE`` arrays.
    particles
        An xsuite ``Particles`` whose ``zeta``/``ptau`` are updated in place.
    frame
        Reference frame linking the two coordinate systems.
    active
        Boolean mask of alive particles. If ``None`` it is recomputed from
        ``particles.state > 0``.
    """
    if active is None:
        active = particles.state > 0
    dt = beam.read_partial_dt()
    dE = beam.read_partial_dE()
    particles.zeta[active] = dt_to_zeta(dt[active], frame)
    particles.ptau[active] = dE_to_ptau(dE[active], frame)
