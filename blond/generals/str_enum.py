# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Compatibility shim for ``enum.StrEnum``.

``enum.StrEnum`` was added in Python 3.11.  This module provides an
equivalent base class on Python 3.10.  Delete it and import ``StrEnum``
from ``enum`` directly once Python 3.10 support is dropped.
"""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Stand-in for Python 3.11's :class:`enum.StrEnum`."""


__all__ = ["StrEnum"]
