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
import xtrack as xt

from blond import backend
from blond.core.base import UserDefinedElement
from blond.core.beam.flags import BeamFlags
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.interfaces.xsuite.elements.helpers import (
    ReferenceFrame,
    dE_to_ptau,
    dt_to_zeta,
    ptau_to_dE,
    zeta_to_dt,
)
from blond.physics.drifts import (
    DriftBaseClass,  # todo is this the correct way?
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass

# todo architectural decisions..


class WrapXsuite4Blond(
    UserDefinedElement, DriftBaseClass
):  # todo should this be RfStationBaseClass too maybe?
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

    # prevent on_init resolution breaking with magic __getattribute__ from xsuite
    skip_find_instances_attributes = ["_xsuite_element"]

    def __init__(
        self, xsuite_element: xt.Line | xt.LineSegmentMap | xt.BeamElement
    ):  # todo should this work for all xsuite elements?
        super().__init__(
            orbit_length=float(xsuite_element.length)
        )  # todo  this only works for `LineSegmentMap`
        self._xsuite_element = xsuite_element
        self._particles = None

    def eta_0(self, gamma: float) -> backend.float:
        # todo think about the
        #  architecture of blond and if its a good idea to have DrifBaseClass
        #  and RfStationBaseClass used explicitly, if an Xsuite
        tw = self._xsuite_element.twiss()

        gamma_t = tw.gamma_transition
        n0 = 1 / np.square(gamma_t) - 1 / np.square(gamma)
        return n0

    def track_reference(self, reference: ReferenceCoordinates, **kwargs):  #
        # todo think about the architecture of blond and if its a good idea
        #  to have DrifBaseClass and RfStationBaseClass used explicitly,
        #  if an Xsuite
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        beta0 = float(beam.reference.beta)
        energy0 = float(beam.reference.total_energy)
        frame = ReferenceFrame(beta0=beta0, energy0=energy0)

        n = len(beam.dt.array_local)

        if self._particles is None or self._particles.zeta.shape[0] != n:
            self._build_particles(beam, energy0, n)
        else:
            # Push BLonD's current reference into the cached Particles so the
            # guest sees the same (beta0, p0c, energy0) as BLonD. Without this,
            # multi-turn tracking with a BLonD-owned ramp would feed the guest
            # the stale reference from the build call.
            mass = float(beam.particle_type.mass)
            new_p0c = float(np.sqrt(energy0**2 - mass**2))
            self._particles.update_p0c(
                np.full(n, new_p0c)
            )  # todo n particles? thats a lot for a reference update

        self._particles.zeta[:] = dt_to_zeta(
            np.asarray(beam.dt.array_local), frame
        )
        self._particles.ptau[:] = dE_to_ptau(
            np.asarray(beam.dE.array_local), frame
        )

        self._xsuite_element.track(self._particles)

        # Pick up any reference advance the xsuite guest performed (energy
        # program, ReferenceEnergyIncrease, ...) so subsequent BLonD elements
        # see the new reference. Read from an active slot — update_p0c masks
        # on state>0 so lost slots may hold a stale value.
        active = np.asarray(self._particles.state) > 0
        if active.any():
            i = int(np.argmax(active))
            new_energy0 = float(np.asarray(self._particles.energy0)[i])
            if new_energy0 != energy0:
                beam.reference.total_energy = new_energy0
                frame = ReferenceFrame(
                    beta0=float(np.asarray(self._particles.beta0)[i]),
                    energy0=new_energy0,
                )

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
