# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Custom exception definitions for BLonD3."""


class BLonDException(Exception):
    """BLonD exception base class."""

    pass


class UnevenArraySizes(BLonDException, ValueError):
    """Exception to be raised when some array dimensions don't match."""

    pass


class ArrayCastingError(BLonDException, ValueError, TypeError):
    """Exception raised when automatic array casting fails."""

    pass


class ArrayShapeError(BLonDException, ValueError):
    """Exception raised when an array is the wrong shape."""

    pass
