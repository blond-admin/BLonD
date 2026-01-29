# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Run test to check xsuite interface works properly."""

import numpy as np

from .simple_xsuite_blond_line import run_simulation as run_blond
from .simple_xsuite_line import run_simulation as run_xsuite


def test_blond_and_xsuite_interface():
    """Run xsuite + blond element simulation."""
    zeta_xsuite, delta_xsuite, init_dist = run_xsuite()
    zeta_blond, delta_blond = run_blond(init_distribution=init_dist)

    assert zeta_blond.shape == zeta_xsuite.shape
    assert delta_blond.shape == delta_xsuite.shape

    np.testing.assert_allclose(
        zeta_blond,
        zeta_xsuite,
        rtol=1e-8,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        delta_blond,
        delta_xsuite,
        rtol=1e-8,
        atol=1e-10,
    )
