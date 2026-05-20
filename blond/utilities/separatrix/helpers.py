# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to work with `SymbolicSeparatrixHelper`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.physics.cavities import RFStationBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.ring.ring import Ring


# Might be refactored as method of Ring if required.
def _get_omega_min(ring: Ring) -> float:
    """
    Get the minimum RF angular frequency presently in the ring.

    Parameters
    ----------
    ring
        The `Ring` object to fetch the parameters from.

    Returns
    -------
    omega_min
        The minimum angular frequency in the `Ring`, in [Hz].
    """
    omega_min = None
    for element in ring.elements.get_elements(RFStationBaseClass):
        omega_design = element.omega_rf_design
        if omega_design is not None:
            candidates = np.atleast_1d(omega_design)
            candidate = float(np.min(candidates))
            if omega_min is None or candidate < omega_min:
                omega_min = candidate
    assert omega_min is not None, (
        "None of the RF stations provided for `omega_min`."
    )
    return omega_min
