# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for the MultiTurnInjection class.

Run as python testBeamObject.py in console or via travis
'''

# General imports
# -----------------

import unittest
import numpy as np

# BLonD imports
# --------------
from blond.beam.beam import Beam, Proton
from blond.beam.multi_turn_injection import MultiTurnInjection
from blond.input_parameters.ring import Ring



class testMultiTurnInjection(unittest.TestCase):

    def setUp(self):

        N_turn = 1
        C = 6911.5038  # Machine circumference [m]
        p = 450e9  # Synchronous momentum [eV/c]
        gamma_t = 17.95142852  # Transition gamma
        alpha = 1. / gamma_t**2  # First order mom. comp. factor

        self.general_params = Ring(C, alpha, p, Proton(), N_turn)


    def test_injection_declaration(self):

        N_p = 1000

        main_beam = Beam(self.general_params, N_p, 0)

        multi_turn_injection = MultiTurnInjection(main_beam)
        multi_turn_injection.set_counter([0])
    
        for i in range(5):
            multi_turn_injection.add_injection(Beam(self.general_params, N_p,
                                                    0))

        self.assertEqual(len(multi_turn_injection._injections), 5)

        for i in range(5):
            self.assertIn(i+1, multi_turn_injection._injections.keys())

        for i in range(5):
            multi_turn_injection.add_injection(np.zeros([2, 1000]))

        self.assertEqual(len(multi_turn_injection._injections), 10)

        for i in range(10):
            self.assertIn(i+1, multi_turn_injection._injections.keys())

        for i in range(10):
            if i < 5:
                self.assertIsInstance(multi_turn_injection._injections[i+1],
                                      Beam)
            else:
                self.assertIsInstance(multi_turn_injection._injections[i+1],
                                      np.ndarray)


    def test_injection_removal(self):

        N_p = 1000

        main_beam = Beam(self.general_params, N_p, 0)

        multi_turn_injection = MultiTurnInjection(main_beam)
        multi_turn_injection.set_counter([0])
    
        for i in range(5):
            multi_turn_injection.add_injection(Beam(self.general_params, N_p,
                                                    0))
        
        for i in range(5):
            multi_turn_injection._counter[0] += 1
            self.assertEqual(len(multi_turn_injection._injections), 5-i)
            next(multi_turn_injection)
        self.assertEqual(len(multi_turn_injection._injections), 0)


    def test_iteration(self):

        N_p = 1000

        main_beam = Beam(self.general_params, N_p, 0)

        multi_turn_injection = MultiTurnInjection(main_beam)
        multi_turn_injection.set_counter([0])
    
        for i in range(5):
            multi_turn_injection.add_injection(Beam(self.general_params, N_p,
                                                    0))

        for _ in multi_turn_injection:
            multi_turn_injection._counter[0] += 1
            pass
        
        self.assertEqual(main_beam.n_macroparticles, 6000)
        self.assertEqual(len(multi_turn_injection._injections), 0)


    def test_tracking(self):


        N_p = 1000

        main_beam = Beam(self.general_params, N_p, 0)

        multi_turn_injection = MultiTurnInjection(main_beam)
        multi_turn_injection.set_counter([0])
    
        for i in range(5):
            multi_turn_injection.add_injection(Beam(self.general_params, N_p,
                                                    0))

        for _ in multi_turn_injection:
            multi_turn_injection._counter[0] += 1

        with self.assertRaises(StopIteration):
            multi_turn_injection.__next__()

        with self.assertRaises(StopIteration):
            multi_turn_injection.track()

    def test_late_injection(self):

        N_p = 1000

        main_beam = Beam(self.general_params, N_p, 0)

        multi_turn_injection = MultiTurnInjection(main_beam)
        multi_turn_injection.set_counter([0])

        multi_turn_injection.add_injection(Beam(self.general_params, N_p, 0),
                                           1000)
        
        self.assertEqual(len(multi_turn_injection._injections), 1)
        self.assertEqual(tuple(multi_turn_injection._injections.keys())[0],
                         1000)
        
        for _ in multi_turn_injection:
            multi_turn_injection._counter[0] += 1

        self.assertEqual(len(multi_turn_injection._injections), 0)
        self.assertEqual(multi_turn_injection._counter[0], 1001)
