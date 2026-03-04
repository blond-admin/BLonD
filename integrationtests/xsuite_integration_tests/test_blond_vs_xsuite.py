# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Run test to check xsuite interface works properly."""

import numpy as np
from matplotlib import pyplot as plt

from .simple_xsuite_blond_line import run_simulation as run_blond
from .simple_xsuite_line import run_simulation as run_xsuite


def test_blond_xsuite_interface():
    """Run xsuite + blond element simulation."""
    PLOT = True
    n_turns = 1000
    zeta_xsuite, delta_xsuite, init_dist = run_xsuite(n_turns=n_turns)
    zeta_blond, delta_blond = run_blond(
        n_turns=n_turns, init_distribution=init_dist
    )

    assert zeta_blond.shape == zeta_xsuite.shape
    assert delta_blond.shape == delta_xsuite.shape

    if PLOT:
        plt.scatter(
            zeta_blond[:, 0], delta_blond[:, 0], label="xsuite + BLonD"
        )
        plt.scatter(zeta_xsuite[:, 0], delta_xsuite[:, 0], label="xsuite")
        plt.title("After 0 turns")
        plt.xlabel("$\zeta$ [m]")
        plt.ylabel(r"$\delta$")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.scatter(
            zeta_blond[:, -1], delta_blond[:, -1], label="xsuite + BLonD"
        )
        plt.scatter(zeta_xsuite[:, -1], delta_xsuite[:, -1], label="xsuite")
        plt.title(f"After {n_turns} turns")
        plt.xlabel("$\zeta$ [m]")
        plt.ylabel(r"$\delta$")
        plt.legend()
        plt.tight_layout()
        plt.show()

        dz = []
        dp = []
        for i in range(n_turns):
            dz.append(zeta_blond[0, i] - zeta_xsuite[0, i])
            dp.append(delta_blond[0, i] - delta_xsuite[0, i])
        dz = np.array(dz)
        dp = np.array(dp)

        plt.plot(dz, marker="o", linestyle="-", label=r" $|\Delta\zeta|$")
        plt.plot(dp, marker="o", linestyle="-", label=r" $|\delta |$")
        plt.xlabel("Turn")
        plt.ylabel("Difference")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    np.testing.assert_allclose(
        zeta_blond,
        zeta_xsuite,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        delta_blond,
        delta_xsuite,
        atol=1e-11,
    )
