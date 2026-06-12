# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Utility functions for the CERN Large Hadron Collider."""

from blond.specifics.cern.lhc.filling_schemes import (
    BUCKETS_PER_SLOT,
    LHC_HARMONIC_NUMBER,
    LHC_N_SLOTS,
    filling_pattern_from_scheme_file,
)

__all__ = [
    "BUCKETS_PER_SLOT",
    "LHC_HARMONIC_NUMBER",
    "LHC_N_SLOTS",
    "filling_pattern_from_scheme_file",
]
