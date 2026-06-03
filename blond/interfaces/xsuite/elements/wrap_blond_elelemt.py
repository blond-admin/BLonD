# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Wrap a BLonD trackable so it can be tracked inside an xsuite ``Line``."""

from __future__ import annotations

import numpy as np

from blond.core.beam.beams import Beam
from blond.core.beam.particle_types import ParticleType
from blond.cycles.magnetic_cycle import ExternalReferenceCycle
from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    beam_to_particles,
    particles_to_beam,
)


def _scalar(x) -> float:
    """Extract a Python float from an xsuite scalar/0-d/1-d quantity."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flat[0])


class WrapBlond4Xsuite:
    """
    Track a single BLonD trackable inside an xsuite ``Line``.

    The wrapper implements xsuite's element interface (``track(particles)``).
    Each call:

    1. Reads the reference frame from xsuite (``particles.beta0[0]``,
       ``particles.energy0[0]``) — xsuite is the source of truth.
    2. If the wrapped element holds an :class:`ExternalReferenceCycle`, pushes
       the new reference total energy into it so the element follows the ramp.
    3. Converts active particle coordinates into a reusable BLonD ``Beam``.
    4. Calls ``element.track(beam)``.
    5. Writes the updated coordinates back into the active particles, leaving
       lost particles untouched.

    Parameters
    ----------
    element
        A BLonD trackable, typically constructed via the element's
        ``headless(...)`` factory. If it owns an
        :class:`~blond.cycles.magnetic_cycle.ExternalReferenceCycle`, the
        reference energy is driven from ``particles.energy0`` each turn.

    Notes
    -----
    The reference frame is read from ``particles`` (live state), not from
    ``line.particle_ref`` (design state). xsuite updates the particles'
    ``beta0``/``energy0``/``p0c`` in place when a ramping element such as
    ``ReferenceEnergyIncrease`` fires earlier in the line, so by the time this
    wrapper is reached the particles already carry the current reference.
    """

    def __init__(self, element):
        self._element = element
        self._beam: Beam | None = None
        cycle = getattr(element, "_magnetic_cycle", None)
        self._cycle = (
            cycle if isinstance(cycle, ExternalReferenceCycle) else None
        )

    def track(self, particles) -> None:
        """
        Track xsuite ``particles`` through the wrapped BLonD element.

        Parameters
        ----------
        particles
            xsuite ``Particles`` whose ``zeta`` / ``ptau`` are updated in place.
        """
        beta0 = _scalar(particles.beta0)
        energy0 = _scalar(particles.energy0)
        if self._cycle is not None:
            self._cycle.set_total_energy(energy0)

        frame = ReferenceFrame(beta0=beta0, energy0=energy0)

        n = int(np.asarray(particles.zeta).shape[0])
        if self._beam is None or len(self._beam.read_partial_dt()) != n:
            self._build_beam(particles, energy0, n)

        active = particles_to_beam(particles, self._beam, frame)
        self._element.track(self._beam)
        beam_to_particles(self._beam, particles, frame, active)

    def _build_beam(self, particles, energy0: float, n: int) -> None:
        particle_type = ParticleType(
            mass=_scalar(particles.mass0),
            charge=_scalar(particles.q0),
        )
        self._beam = Beam(intensity=1.0, particle_type=particle_type)
        self._beam.setup_beam(
            dt=np.zeros(n),
            dE=np.zeros(n),
            reference_total_energy=energy0,
        )
