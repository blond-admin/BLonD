# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Periodic boundary enforcement for longitudinal tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.physics.cavities import RFStationBaseClass
    from blond.physics.drifts import DriftBaseClass


class PeriodicBoundary(BeamPhysicsRelevant):
    r"""
    Enforce periodic boundary conditions on :math:`[0, t_\text{rev})`.

    This element must be added **twice** to the simulation element list,
    bracketing the rf station and drift:

    .. code-block:: python

        ring.add_elements(bound, rf_station, drift, bound)

    **First call** (before rf+drift)
        Particles with ``dt > t_rev`` are folded back by subtracting
        :math:`t_\text{rev}`.  Their post-fold state (after the fold, before
        the kick) is saved so that the incorrect kick+drift applied by the
        enclosed elements can be undone later.

    **rf+drift** (standard elements, unmodified)
        Both the inside particles and the (now-folded) right-escaped
        particles go through the standard kick and drift.

    **Second call** (after rf+drift)
        The right-escaped particles are restored to their saved post-fold
        state, effectively cancelling the kick+drift they should not have
        received.  Then particles that drifted past ``dt = 0`` are folded
        forward and given an additional kick+drift via the rf station and
        drift elements using a :class:`_FrozenReference` proxy (so that the
        reference coordinates are not advanced a second time).

    Parameters
    ----------
    section_index
        Section index to group elements into sections.
    name
        Human-readable name for this element.

    Notes
    -----
    The rf station and drift are resolved automatically from the ring via
    :meth:`on_init_simulation`; they remain independent elements in the
    element list and are not owned by this class.
    """

    def __init__(
        self,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(section_index=section_index, name=name)
        self._rf_station: RFStationBaseClass | None = None
        self._drift: DriftBaseClass | None = None

        # State persisted between the first and second call within one turn.
        # None  -> next _track call is the first call of this turn.
        # tuple -> next _track call is the second call; tuple holds the saved state.
        self._saved_state: (
            tuple[NumpyArray, NumpyArray, NumpyArray, float] | None
        ) = None

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Resolve the rf station and drift from the ring at simulation init time.
        """
        self._rf_station = simulation.ring.elements.get_element(
            RFStationBaseClass,
        )
        self._drift = simulation.ring.elements.get_element(
            DriftBaseClass,
        )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Apply the periodic boundary logic for one half-turn.

        The first call of a turn folds and saves; the second call restores and
        re-applies.  The saved state distinguishes which half is executing.

        Parameters
        ----------
        beam
            The beam whose longitudinal coordinates are updated.
        """
        if self._saved_state is None:
            self._first_call(beam)
        else:
            self._second_call(beam)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _first_call(self, beam: BeamBaseClass) -> None:
        """
        Fold right-escaped particles and save their pre-kick state.

        Called before the rf station and drift elements execute.
        """
        # Drift conserves energy, so velocity and t_rev are constant over the step.
        t_rev = self._drift.orbit_length / beam.reference.velocity

        dt = beam.write_partial_dt()
        dE = beam.write_partial_dE()

        right_mask = dt > t_rev
        dt[right_mask] -= t_rev

        # Save state: right-escaped particles must have their post-fold
        # coordinates restored in the second call to undo the kick+drift
        # that the enclosed elements will (incorrectly) apply to them.
        self._saved_state = (
            right_mask,
            dt[right_mask].copy(),
            dE[right_mask].copy(),
            t_rev,
        )

    def _second_call(self, beam: BeamBaseClass) -> None:
        """
        Restore right-escaped particles and re-apply kick+drift to left-escaped ones.

        Called after the rf station and drift elements have executed.
        """
        right_mask, saved_dt, saved_dE, t_rev = self._saved_state
        self._saved_state = None  # reset for the next turn

        dt = beam.write_partial_dt()
        dE = beam.write_partial_dE()

        # Restore right-escaped particles to their post-fold, pre-kick state,
        # undoing the kick+drift the enclosed elements applied to them.
        dt[right_mask] = saved_dt
        dE[right_mask] = saved_dE

        # Re-apply kick+drift to any particles that drifted past the left edge.
        # A frozen reference prevents the reference coordinates from advancing
        # a second time within this turn.
        left_mask = dt < 0
        if backend.any(left_mask):
            left_view = _BeamSubset(
                beam, left_mask, reference=_FrozenReference(beam.reference)
            )
            left_view._dt += t_rev
            self._rf_station._track(left_view)
            self._drift._track(left_view)
            left_view.write_back(dt, dE)


if __name__ == "__main__":
    # pseudocode
    bound = PeriodicBoundary()
    ring.add_elements(
        bound,  # first call
        rf,
        drift,
        wakefield,  # this is ignored for example?
        bound,  # second call
    )
