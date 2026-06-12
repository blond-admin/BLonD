# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Conversion helpers for filling patterns.

Convention: physical distances are start-to-start times in seconds;
integer results count RF buckets.
"""

from __future__ import annotations

import warnings


def as_n_buckets(
    time_distance: float, f_rf: float, tolerance: float = 0.05
) -> int:
    """
    Return the number of RF buckets matching a physical time distance.

    Rounds to the nearest integer number of buckets and warns when
    ``time_distance`` deviates from that integer by more than ``tolerance``
    buckets (default 0.05 — loose enough that e.g. a 25 ns spacing on the
    LHC 400 MHz RF, 10.02 buckets, passes silently).

    Parameters
    ----------
    time_distance
        Physical start-to-start distance in seconds.
    f_rf
        RF frequency in Hz.
    tolerance
        Maximum accepted deviation from an integer, in buckets.

    Returns
    -------
    n_buckets
        Number of RF buckets, rounded to the nearest integer.
    """
    n_buckets_exact = time_distance * f_rf
    n_buckets = round(n_buckets_exact)
    if abs(n_buckets_exact - n_buckets) > tolerance:
        warnings.warn(
            f"time_distance = {time_distance} s corresponds to "
            f"{n_buckets_exact:.4f} RF buckets, which is not an integer "
            f"number of buckets (rounded to {n_buckets}).",
            stacklevel=2,
        )
    return n_buckets
