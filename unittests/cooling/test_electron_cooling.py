# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for the ElectronCooling class.

Run as python testElectronCooling.py in console or via travis
'''

# General imports
# -----------------
from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as np_test
import scipy.constants as cont

# BLonD imports
# -----------------
import blond.input_parameters.ring as inputRing
import blond.input_parameters.ring_options as ringOpt
import blond.beam.beam as beam
import blond.cooling.electron_cooling as ecool
import blond.utils.bmath as bm

_M_E = cont.physical_constants['electron mass energy equivalent in MeV'][0]*1E6

class testElectronCooling(unittest.TestCase):

    def setUp(self):
        momentum = 0.3391E9
        charge = 54
        mass = 193.702E9

        momentum *= charge

        circ = 2*np.pi*12.5
        gamma_t = 2.8
        alpha = 1/(gamma_t**2)

        lead = beam.Particle(mass, charge)

        n_turns = 1000

        ring_options = ringOpt.RingOptions("linear")
        self.ring = inputRing.Ring(circ, alpha, momentum, n_turns = n_turns,
                                   Particle = lead, RingOptions = ring_options)


    def test_cooling_force(self):

        length = 1
        v_gun = 2E3
        e_spread = 1E-3
        e_dens = 1E20
        cycle_time = [0]
        counter = [0]

        cooler = ecool.ElectronCooling(beam.Beam(self.ring, 1, 0), length,
                                       v_gun, e_spread, e_dens, cycle_time,
                                       counter)

        e_energy = _M_E + v_gun
        e_beta = 1/bm.sqrt(1 + _M_E**2/(e_energy**2 - _M_E**2))
        e_vel = cont.c*e_beta

        self.assertEqual(cooler.cooling_force(0, e_vel), 0,
                         "At 0 velocity offset, cooling force should be 0")

        self.assertAlmostEqual(cooler.cooling_force(0, e_vel*0.5), 0, 5,
                         "At large velocity offset, cooling force should be "
                         + "almost 0")

        vel_range = np.linspace(e_vel*0.995, e_vel*1.005, 1000)
        force_range = cooler.cooling_force(0, vel_range)

        max_at = np.where(force_range == np.max(force_range))[0][0]
        min_at = np.where(force_range == np.min(force_range))[0][0]

        self.assertAlmostEqual(force_range[max_at], +3.01E-19, 1,
                               "Force should have a minimum of ~-3.01E-19")
        self.assertAlmostEqual(force_range[min_at], -3.01E-19, 1,
                               "Force should have a maximum of ~+3.01E-19")

        self.assertAlmostEqual(vel_range[max_at]-e_vel, -8868.45, 1,
                               "Force should have a minimum at ~-8868.45")
        self.assertAlmostEqual(vel_range[min_at]-e_vel, +8868.45, 1,
                               "Force should have a maximum at ~+8868.45")


    def test_tracking(self):

        length = 1
        e_spread = 1E-3
        e_dens = 0
        cycle_time = [0]
        counter = [0]

        test_beam = beam.Beam(self.ring, 3, 0)


        e_beta = self.ring.beta[0][0]
        e_energy = np.sqrt(_M_E**2 * (1 + 1/(1/e_beta**2 - 1)))
        v_gun = e_energy - _M_E

        cooler = ecool.ElectronCooling(test_beam, length,
                                       v_gun, e_spread, e_dens, cycle_time,
                                       counter)

        test_beam.dE[0] = 0
        test_beam.dE[1] = 1E6
        test_beam.dE[2] = 2E6

        for i in range(1000):
            cooler.track()

        np_test.assert_array_equal(test_beam.dE, [0, 1E6, 2E6],
                                   "Energy change should be 0 with 0 electron"
                                   +" density")

        cooler.density_electrons = [1E20]*1000
        for i in range(1000):
            cooler.track()

        self.assertAlmostEqual(test_beam.dE[0]*1E8, -6.6353, 3,
                               msg = "Particle[0] energy incorrect")
        self.assertAlmostEqual(test_beam.dE[1]/1E5, 9.9876, 3,
                               msg = "Particle[1] energy incorrect")
        self.assertAlmostEqual(test_beam.dE[2]/1E6, 1.9995, 3,
                               msg = "Particle[2] energy incorrect")


if __name__ == "__main__":
    unittest.main()