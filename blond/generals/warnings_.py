# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Custom warning definitions for BLonD3.

Authors
-------
Simon Lauber
"""


class PerformanceWarning(UserWarning):
    """Warning for performance-related issues."""

    pass


class NotTestedWarning(UserWarning):
    """Warning for not tested code/classes."""

    pass


class PrecisionWarning(UserWarning):
    """Warning for changing the precision of floats."""

    pass
