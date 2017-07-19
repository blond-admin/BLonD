# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.filters

:Authors: **Helga Timko**
"""

import unittest
import numpy as np
from llrf.filters import moving_average


class TestMovingAverage(unittest.TestCase):
    
    # Run before every test
    def setUp(self, N = 3, center = False):
        self.x = np.array([0,3,6,3,0,3,6,3,0], dtype=float)
        self.y = moving_average(self.x, N, center)
        
    # Run after every test
    def tearDown(self):
         
        del self.x
        del self.y

    def test_1(self):
        
        self.setUp(N = 3, center = False)
        self.assertEqual(len(self.x), len(self.y) + 3 - 1, 
            msg="In TestMovingAverage, test_1: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([3, 4, 3, 2, 3, 4, 3], dtype=float).tolist(), 
            msg="In TestMovingAverage, test_1: arrays differ")

    def test_2(self):
        
        self.setUp(N = 4, center = False)
        self.assertEqual(len(self.x), len(self.y) + 4 - 1, 
            msg="In TestMovingAverage, test_2: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), np.array([3, 3, 3, 3, 3, 3], 
            dtype=float).tolist(), 
            msg="In TestMovingAverage, test_2: arrays differ")

    def test_3(self):
        
        self.setUp(N = 3, center = True)
        self.assertEqual(len(self.x), len(self.y), 
            msg="In TestMovingAverage, test_3: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([1, 3, 4, 3, 2, 3, 4, 3, 1], dtype=float).tolist(), 
            msg="In TestMovingAverage, test_3: arrays differ")
    
    def test_4(self):
        
        self.setUp(N = 4, center = True)
        self.assertEqual(len(self.x), len(self.y), 
            msg="In TestMovingAverage, test_4: wrong array length")
        self.assertSequenceEqual(self.y.tolist(), 
            np.array([1.8, 2.4, 2.4, 3, 3.6, 3, 2.4, 2.4, 1.8], 
            dtype=float).tolist(), 
            msg="In TestMovingAverage, test_4: arrays differ")

# def test_moving_average_2():
#     
#     x = np.array([0,3,6,3,0,3,6,3,0], dtype=float)
#     N = 4
#     center = False
#     y = moving_average(x, N, center)
#     
#     assert(len(x) == len(y) + N - 1)
#     assert(y == np.array([3, 3, 3, 3, 3, 3], dtype=float))
# 
# 
# def test_moving_average_3():
#     
#     x = np.array([0,3,6,3,0,3,6,3,0], dtype=float)
#     N = 3
#     center = True
#     y = moving_average(x, N, center)
#     
#     assert(len(x) == len(y))
#     assert(y == np.array([1, 3, 4, 3, 2, 3, 4, 3, 1], dtype=float))
# 
#     
# def test_moving_average_4():
#     
#     x = np.array([0,3,6,3,0,3,6,3,0], dtype=float)
#     N = 4
#     center = True
#     y = moving_average(x, N, center)
#     
#     assert(len(x) == len(y))
#     assert(y == np.array([1.8, 2.4, 2.4, 3, 3.6, 3, 2.4, 2.4, 1.8], 
#                          dtype=float))


if __name__ == '__main__':
    unittest.main()


