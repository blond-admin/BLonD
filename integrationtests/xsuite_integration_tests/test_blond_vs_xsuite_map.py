# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Run test to check xsuite interface works properly."""

import numpy as np

from .simple_ import run_simulation as run_blond
from .simple_xsuite_line import run_simulation as run_xsuite
from matplotlib import pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

def test_all_the_same():
    """Run xsuite + blond element simulation."""
    zeta_xsuite, delta_xsuite, init_dist = run_xsuite()
    zeta_blond, delta_blond = run_blond(init_distribution=init_dist)

    print('here')

    assert zeta_blond.shape == zeta_xsuite.shape
    assert delta_blond.shape == delta_xsuite.shape

    plt.scatter(zeta_blond[:,0], delta_blond[:,0], label='xsuite + BLonD')
    plt.scatter(zeta_xsuite[:,0], delta_xsuite[:,0], label='xsuite')
    plt.title('After 0 turns')
    plt.xlabel('$\zeta$ [m]')
    plt.ylabel(r'$p_{\tau}$')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./ebeg_turns_1000.png')

    plt.show()



    plt.scatter(zeta_blond[:,-1], delta_blond[:,-1], label='xsuite + BLonD')
    plt.scatter(zeta_xsuite[:,-1], delta_xsuite[:,-1], label='xsuite')
    plt.title('After 100 turns')
    plt.xlabel('$\zeta$ [m]')
    plt.ylabel(r'$p_{\tau}$')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./end_turns_1000.png')
    plt.show()


    dz = []
    dp = []
    for i in range(1000):
        dz.append(zeta_blond[0,i] - zeta_xsuite[0,i])  # shape (n_particles, n_turns)
        dp.append(delta_blond[0,i] -delta_xsuite[0,i])

    plt.plot(dz, label=r" $|\Delta\zeta|$")
    plt.plot(dp, label=r" $|\Delta p_{\tau}|$")
    plt.xlabel("Turn")
    plt.ylabel("difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./difference_turns_1000_oscillation.png')
    plt.show()

    np.testing.assert_allclose(
        zeta_blond[-1],
        zeta_xsuite[-1],
        rtol=1e-15,
        atol=1e-15,
    )
    #
    # np.testing.assert_allclose(
    #     delta_blond,
    #     delta_xsuite,
    #     rtol=1e-100,
    #     atol=1e-100,
    # )