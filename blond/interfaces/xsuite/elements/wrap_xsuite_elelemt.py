# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Wrap an xsuite element/``Line`` so it can be tracked inside a BLonD ``Ring``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import UserDefinedElement
from blond.core.beam.flags import BeamFlags
from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass


class WrapXsuite4Blond(UserDefinedElement):
    """
    Track an xsuite element or ``Line`` inside a BLonD ``Ring``.

    A normal BLonD element: BLonD owns the main tracking loop and the
    reference frame (``beam.reference``). Each call to ``_track``:

    1. Builds a reference frame from ``beam.reference``.
    2. Converts the beam ``(dt, dE)`` into a reusable xsuite ``Particles``
       buffer (``zeta``, ``ptau``).
    3. Calls ``element.track(particles)`` on the wrapped guest.
    4. Writes the updated coordinates back into the beam, flags slots whose
       xsuite particle was lost, and — if the guest advanced
       ``particles.energy0`` (e.g. via an attached energy program) — pushes the
       new reference total energy into ``beam.reference``.

    Parameters
    ----------
    xsuite_element
        Any object exposing a ``track(particles)`` method (e.g.
        ``xtrack.Drift``, an ``xtrack.Line``, ...).
    """

    def __init__(self, xsuite_element):
        super().__init__()
        self._xs = xsuite_element
        self._particles = None

    def _track(self, beam: BeamBaseClass) -> None:
        beta0 = float(beam.reference.beta)
        energy0 = float(beam.reference.total_energy)
        frame = ReferenceFrame(beta0=beta0, energy0=energy0)

        n = len(beam.dt.array_local)

        if self._particles is None or self._particles.zeta.shape[0] != n:
            self._build_particles(beam, energy0, n)

        self._particles.zeta[:] = dt_to_zeta(
            np.asarray(beam.dt.array_local), frame
        )
        self._particles.ptau[:] = dE_to_ptau(
            np.asarray(beam.dE.array_local), frame
        )

        self._xs.track(self._particles)

        # Pick up any reference advance the xsuite guest performed (energy
        # program, ReferenceEnergyIncrease, ...) so subsequent BLonD elements
        # see the new reference.
        new_energy0 = float(np.asarray(self._particles.energy0).flat[0])
        if new_energy0 != energy0:
            beam.reference.total_energy = new_energy0
            frame = ReferenceFrame(
                beta0=float(np.asarray(self._particles.beta0).flat[0]),
                energy0=new_energy0,
            )

        active = np.asarray(self._particles.state) > 0
        beam.dt.array_local[active] = zeta_to_dt(
            np.asarray(self._particles.zeta)[active], frame
        )
        beam.dE.array_local[active] = ptau_to_dE(
            np.asarray(self._particles.ptau)[active], frame
        )
        if not active.all():
            flags = beam.flags.array_local
            flags[~active] = BeamFlags.LOST.value

    def _build_particles(
        self, beam: BeamBaseClass, energy0: float, n: int
    ) -> None:
        from xtrack import Particles

        mass = float(beam.particle_type.mass)
        p0c = float(np.sqrt(energy0**2 - mass**2))
        self._particles = Particles(
            p0c=p0c,
            mass0=mass,
            q0=float(beam.particle_type.charge),
            zeta=np.zeros(n),
            ptau=np.zeros(n),
        )
