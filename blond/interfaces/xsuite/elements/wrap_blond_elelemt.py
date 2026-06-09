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
from blond.core.beam.flags import BeamFlags
from blond.core.beam.particle_types import ParticleType
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

    Universal wrapper: it knows nothing about the wrapped element. The xsuite
    line is the source of truth for the reference, and each call to
    :meth:`track`:

    1. Reads the live reference from ``particles`` (``beta0[0]``, ``energy0[0]``).
    2. Writes that reference total energy into the BLonD ``beam.reference`` so
       any element that reads ``beam.reference.beta``/``gamma`` (e.g. an RF
       station computing ``omega_rev``) follows the xsuite ramp.
    3. Converts active particle coordinates into a reusable BLonD ``Beam``,
       marking xsuite-lost slots as LOST in the beam flags.
    4. Calls ``element.track(beam)``.
    5. Propagates any *new* LOST flags the BLonD element raised during track
       back into ``particles.state`` so downstream xsuite elements also skip
       those slots.
    6. Writes the updated coordinates back into the active particles, leaving
       lost particles' coordinates untouched.

    Parameters
    ----------
    element
        A BLonD trackable, typically constructed via the element's
        ``headless(...)`` factory. Elements that would otherwise advance the
        reference themselves (RF stations) must be constructed with
        ``magnetic_cycle=None`` so xsuite remains the sole driver of the
        reference energy.

    Notes
    -----
    The reference frame is read from ``particles`` (live state), not from
    ``line.particle_ref`` (design state). xsuite advances the particles'
    ``beta0``/``energy0``/``p0c`` between turns when ``line.energy_program`` is
    attached, or in-line when an explicit ``ReferenceEnergyIncrease`` element
    fires — by the time this wrapper runs, ``particles`` already carries the
    current reference.
    """

    def __init__(self, element):
        self._element = element
        self._beam: Beam | None = None

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
        frame = ReferenceFrame(beta0=beta0, energy0=energy0)

        n = int(np.asarray(particles.zeta).shape[0])
        if self._beam is None or len(self._beam.read_partial_dt()) != n:
            self._build_beam(particles, energy0, n)

        self._beam.reference.total_energy = energy0
        active_at_input = particles_to_beam(particles, self._beam, frame)
        self._element.track(self._beam)

        # The BLonD element may have flagged additional particles LOST
        # during its track (loss boxes, energy cuts, ...). Propagate those
        # losses back into xsuite's state so subsequent xsuite elements
        # also skip them — and only write coordinates back for slots that
        # survived BLonD's track.
        blond_active_now = (
            np.asarray(self._beam.read_partial_flags())
            == BeamFlags.ACTIVE.value
        )
        newly_lost = active_at_input & ~blond_active_now
        if newly_lost.any():
            particles.state[newly_lost] = -1
        beam_to_particles(
            self._beam, particles, frame, active_at_input & blond_active_now
        )

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
