# -*- coding: utf-8 -*-

# General imports
# -----------------
from __future__ import division, print_function
import unittest
import numpy as np

# BLonD imports
# --------------
from blond.beam.beam import Proton
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import Profile
from blond.beam.beam import Beam
from blond.trackers.tracker import RingAndRFTracker
from blond.trackers.tracker import FullRingAndRF
from blond.utils.track_iteration import TrackIteration

class TestTrackIteration(unittest.TestCase):

    def setUp(self):

        initial_time = 0
        final_time = 1E-3

        # Machine and RF parameters
        radius = 25
        gamma_transition = 4.4  # [1]
        C = 2 * np.pi * radius  # [m]       
        momentum_compaction = 1 / gamma_transition**2 # [1]
        particle_type = 'proton'

        self.ring = Ring(C, momentum_compaction, \
                                   ([0, 1E-3], [3.13E8, 3.13E8]), Proton())

        self.rf_params = RFStation(self.ring, [1], [1E3], [np.pi], 1)
        self.beam = Beam(self.ring, 1, 0)
        self.profile = Profile(self.beam)

        self.long_tracker = RingAndRFTracker(self.rf_params, self.beam)
        self.full_ring = FullRingAndRF([self.long_tracker])

        self.n_turns = self.ring.n_turns

        self.map_ = [self.full_ring, self.profile]

        self.trackIt = TrackIteration(self.map_)



    def test_magic(self):

        self.assertTrue(hasattr(self.trackIt, '__iter__'), msg='__iter__ does not exist')
        self.assertTrue(hasattr(self.trackIt, '__call__'), msg='__call__ does not exist')
        self.assertTrue(hasattr(self.trackIt, '__next__'), msg='__next__ does not exist')
        self.assertTrue(hasattr(self.trackIt, '__init__'), msg='__init__ does not exist')


    def test_next(self):

        self.assertEqual(next(self.trackIt), 1, msg='Returned turn number should be 1')
        self.assertEqual(next(self.trackIt), 2, msg='Returned turn number should be 2')
        self.assertEqual(next(self.trackIt), 3, msg='Returned turn number should be 3')

        for i in range(5):
            next(self.trackIt)

        self.assertEqual(self.trackIt.turnNumber, 8, msg='Turn number should have incremented to 8')


    def test_iter(self):

        for i in self.trackIt:
            pass
        self.assertEqual(self.n_turns, self.trackIt.turnNumber, msg='Iterating all turns has not incremented turnNumber correctly')


    def test_call(self):

        self.assertEqual(self.trackIt(10), 10, msg='Call has not returned correct turn number')
        self.assertEqual(self.trackIt(10), 20, msg='Call has not returned correct turn number')



    def test_added_functions(self):

        list1 = [0]
        list2 = [0]
        def increment(map_, turnN, inputList):
            inputList[0] += 1
        self.trackIt.add_function(increment, 1, list1)

        def setToTurn(map_, turnN, inputList):
            inputList[0] = turnN

        self.trackIt.add_function(setToTurn, 4, list2)

        list3 = [0]
        def turnCalc(map_, turnN, a, b, inputList):
            inputList[0] = turnN*a + b

        self.trackIt.add_function(turnCalc, 3, 2, 2, inputList = list3)

        next(self.trackIt)

        self.assertEqual(list1[0], 1, msg='function call should have incremented list')
        self.assertEqual(list2[0], 0, msg='function should not have been called')
        self.assertEqual(list3[0], 0, msg='function should not have been called')

        self.trackIt(3)

        self.assertEqual(list1[0], 4, msg='function call should have incremented list')
        self.assertEqual(list2[0], self.trackIt.turnNumber, msg='function should set list[0] to turn number')
        self.assertEqual(list3[0], 8, msg='function should have been called')



if __name__ == '__main__':

    unittest.main()