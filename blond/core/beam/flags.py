# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module to define particle flags."""

from enum import IntEnum


class BeamFlags(IntEnum):
    """Flags that define the beam state."""

    # Please mind that the LOST flag is hardcoded in all backends
    # for loss_box
    LOST = -500  # by convention with XSuite team.
    ACTIVE = 1
