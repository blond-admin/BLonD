# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
"""

import unittest

import numpy as np

import blond.utils.data_check as dCheck


class TestRFModulation(unittest.TestCase):

    def test_check_number(self):

        self.assertTrue(dCheck._check_number(0),
                        msg='Number input should return True')
        self.assertFalse(dCheck._check_number('a'),
                         msg='String input should return False')
        self.assertFalse(dCheck._check_number([1]),
                         msg='List input should return False')
        self.assertFalse(dCheck._check_number((1,)),
                         msg='Tuple input should return False')
        self.assertFalse(dCheck._check_number(np.array([1])),
                         msg='np.array input should return False')

    def test_check_length(self):

        self.assertTrue(dCheck._check_length([1, 2], 2),
                        msg='Length 2 input compared to 2 should return True')
        self.assertFalse(dCheck._check_length([1, 2], 3),
                         msg='Length 2 input compared to 3 should return False')
        self.assertFalse(dCheck._check_length([1, 2, 3, 4], 3),
                         msg='Length 4 input compared to 3 should return False')
        self.assertFalse(dCheck._check_length(dCheck._check_length, 3),
                         msg='Function should return False')

        with self.assertRaises(TypeError, msg='String length should raise TypeError'):
            dCheck._check_length([1, 2], 'a')

    def test_check_dimensions(self):

        self.assertTrue(dCheck._check_dimensions([1, 2], 2),
                        msg='Length 2 list compared to 2 should return True')
        self.assertFalse(dCheck._check_dimensions([[1, 2], [3, 4]], 2),
                         msg='[2, 2] list compared to 2 should return False')
        self.assertTrue(dCheck._check_dimensions([[1, 2], [3, 4]], (2, 2)),
                        msg='[2, 2] list compared to (2, 2) should return True')
        self.assertFalse(dCheck._check_dimensions([1, 2, 3, 4], (2, 2)),
                         msg='Length 4 list compared to (2, 2) should return False')
        self.assertTrue(dCheck._check_dimensions([1, 2, 3, 4], -1),
                        msg='Length 4 list compared to -1 should return True')
        self.assertTrue(dCheck._check_dimensions([[1, 2, 3, 4], [1, 2, 3, 4]], (2, -1)),
                        msg='[2, 4] list compared to (2, -1) should return True')

        with self.assertRaises(TypeError, msg='String dim should raise TypeError'):
            dCheck._check_dimensions([[1, 2], [3, 4]], 'a')

    def test_check_data_dimensions(self):

        inputToCheckTrue = (1, [1, 2], np.zeros([3, 2]), np.zeros([2, 2000]))
        inputToCheckFalse = ([2], np.zeros([3, 2]), np.zeros([2, 2000]), 1)
        toCompareWith = (0, 2, (3, 2), (2, -1))

        for pairs in zip(zip(inputToCheckTrue, toCompareWith),
                         zip(inputToCheckFalse, toCompareWith)):

            self.assertTrue(dCheck.check_data_dimensions(pairs[0][0],
                                                         pairs[0][1])[0],
                            msg=('Should return True with', pairs[0]))
            self.assertFalse(dCheck.check_data_dimensions(pairs[1][0],
                                                          pairs[1][1])[0],
                             msg=('Should return False with', pairs[1]))


if __name__ == '__main__':

    unittest.main()
