# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for coasting beam generation.

'''

import unittest

import numpy as np
import scipy.optimize as spOpt

import blond.beam.beam as bBeam
import blond.beam.coasting_beam as cBeam
import blond.input_parameters.ring as Ring
import blond.utils.exceptions as blExcept
from blond.beam.beam import Proton


class testCoastingBeamModule(unittest.TestCase):

    def setUp(self):

        self.ring = Ring.Ring(2 * np.pi * 25, 1 / 4.4**2, 1E9, Proton())
        self.beam = bBeam.Beam(self.ring, 1E6, 0)
        np.random.seed(1234)

    def test_defaults(self):

        cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0])

        self.assertAlmostEqual(np.min(self.beam.dt) * 1E9, 0, places=2,
                               msg="Min dt should be almost 0")
        self.assertAlmostEqual(np.max(self.beam.dt) * 1E9,
                               self.ring.t_rev[0] * 1E9, places=2,
                               msg="Max dt should be almost ring.t_rev")

        vals, edges = np.histogram(self.beam.dE, bins=150)

        cents = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]

        popt, pcov = spOpt.curve_fit(self._gauss, cents, vals, p0=[1, 0, 1E6])

        fitResult = self._gauss(cents, *popt)
        diff = vals - fitResult

        self.assertAlmostEqual(np.mean(diff), 2.4, delta=0.1,
                               msg='Default not Gaussian enough')

        self.assertAlmostEqual(np.std(diff), 86, delta=1,
                               msg='Default not Gaussian enough')

        vals, edges = np.histogram(self.beam.dt, bins=150)
        cents = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]

        self.assertAlmostEqual(np.mean(vals), 6666, delta=1,
                               msg="Default not flat enough")
        self.assertAlmostEqual(np.std(vals), 80, delta=1,
                               msg="Default not flat enough")

    def test_spread_types(self):

        testTypes = ['dp/p', 'dE/E', 'dp', 'dE']
        for t in testTypes:
            cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                         spread_type=t)

        with self.assertRaises(blExcept.DistributionError,
                               msg="""Invalid spread type 
                               should raise exception"""):

            cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                         spread_type='bad spread')

    def test_distribution_types(self):

        testDists = ['gaussian', 'parabolic']
        for t in testDists:
            cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                         distribution=t)

        with self.assertRaises(blExcept.DistributionError,
                               msg="""'user' distribution without required
                               input should raise exception"""):

            cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                         spread_type='user')

        with self.assertRaises(blExcept.DistributionError,
                               msg="""Invalid distribution type 
                               should raise exception"""):

            cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                         spread_type='bad distribution')

    def test_spread_offset(self):

        spreads = [1E6, 2E6, 3E6]
        offsets = [0, 1E6, 2E6, 3E6]

        for s in spreads:
            for o in offsets:
                np.random.seed(1234)
                cBeam.generate_coasting_beam(self.beam, 0, self.ring.t_rev[0],
                                             distribution='parabolic',
                                             spread_type='dE',
                                             energy_offset=o,
                                             spread=s)

                vals, edges = np.histogram(self.beam.dE, bins=50)
                cents = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]

                self.assertAlmostEqual(edges[0], o - s, delta=1E5,
                                       msg='Beam edge too far from offset-spread')
                self.assertAlmostEqual(edges[-1], o + s, delta=1E5,
                                       msg='Beam edge too far from offset+spread')

                self.assertAlmostEqual(cents[np.where(vals
                                                      == np.max(vals))[0][0]],
                                       o, delta=2.5 * (cents[1] - cents[0]),
                                       msg='Beam center too far from offset')

    def _gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


if __name__ == '__main__':

    unittest.main()
