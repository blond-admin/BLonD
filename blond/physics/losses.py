"""Collection of implementations to handle beam losses in synchrotrons.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond._core.base import BeamPhysicsRelevant

if TYPE_CHECKING:  # pragma: no cover
    from blond._core.beam.base import BeamBaseClass


class LossesBaseClass(BeamPhysicsRelevant):
    """Abstract class to group/implement losses."""

    def __init__(self) -> None:
        super().__init__()

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        beam.purge_flagged_entries()
