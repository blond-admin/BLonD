# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Re-export of :class:`blond.core.backends.flags.BeamFlags`.

`BeamFlags` lives in `blond.core.backends` so that the backends can read
`BeamFlags.LOST` without depending on `blond.core.beam`.
"""

from blond.core.backends.flags import BeamFlags

__all__ = ["BeamFlags"]
