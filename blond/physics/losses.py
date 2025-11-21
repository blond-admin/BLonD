"""Collection of implementations to handle beam losses in synchrotrons.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass

from blond.core.base import BeamPhysicsRelevant


class LossesBaseClass(BeamPhysicsRelevant):
    """Abstract class to group/implement losses."""

    def __init__(self) -> None:
        super().__init__()
