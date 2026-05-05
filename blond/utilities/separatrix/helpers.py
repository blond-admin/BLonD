# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with `SymbolicSeparatrixHelper`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.ring.ring import Ring


logger = logging.getLogger(__name__)


def _get_omega_min(ring: Ring) -> float:
    """
    Get the minimum RF frequency presently in the ring.

    Parameters
    ----------
    ring
        The `Ring` object to fetch the parameters from.

    Returns
    -------
    omega_min
        The minimum angular frequency in the `Ring`, in [Hz].
    """
    from blond import MultiHarmonicRFStation, SingleHarmonicRFStation

    omega_min = None
    shc = ring.elements.get_elements(SingleHarmonicRFStation)
    mhc = ring.elements.get_elements(MultiHarmonicRFStation)
    for element in shc + mhc:
        omega_design = element.omega_rf_design
        if omega_design is not None:
            candidates = np.abs(np.atleast_1d(omega_design))
            nonzero = candidates[candidates > 0]
            if nonzero.size:
                candidate = float(np.min(nonzero))
                if omega_min is None or candidate < omega_min:
                    omega_min = candidate
    assert omega_min is not None, (
        "None of the RF stations provided for `omega_min`."
    )
    return omega_min
