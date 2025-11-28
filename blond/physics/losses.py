# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to handle beam losses in synchrotrons.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.core.base import BeamPhysicsRelevant

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass


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
