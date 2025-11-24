# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Base class to manage preprogrammed cycles."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from blond.core.base import Preparable

if TYPE_CHECKING:  # pragma: no cover
    pass


class ProgrammedCycle(Preparable, ABC):
    """Programmed cycle of parameters."""

    def __init__(self):
        super().__init__()
