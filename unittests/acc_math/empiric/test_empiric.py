# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import unittest

import numpy as np
from scipy.stats import norm

from blond.acc_math.empiric.empiric import gauss, gauss_fit, multi_gauss_fit


def test_gauss_fit():
    x = np.arange(-4, 4, 0.001)
    p = [0.4, 0, 1]

    gauss_test = norm.pdf(x, p[1], p[2])

    fit = gauss_fit(x, gauss_test)

    np.testing.assert_almost_equal(fit, p, decimal=2)


def test_multi_gauss_fit():
    x = np.arange(-4, 12, 0.001)
    p = np.array([[0.4, 0, 1], [0.4, 8, 1]])

    gauss_test = norm.pdf(x, p[0, 1], p[0, 2]) + norm.pdf(x, p[1, 1], p[1, 2])

    fit = multi_gauss_fit(x, gauss_test, n_bunches=2)

    np.testing.assert_almost_equal(fit, p, decimal=2)


def test_gauss():
    x = np.arange(-4, 4, 0.001)
    p = [0.4, 0, 1]

    gauss_func = gauss(x, p[0], p[1], p[2])
    gauss_test = norm.pdf(x, p[1], p[2])

    np.testing.assert_almost_equal(gauss_func, gauss_test, decimal=2)


if __name__ == "__main__":
    unittest.main()
