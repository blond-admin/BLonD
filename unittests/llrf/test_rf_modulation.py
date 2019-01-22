# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:57:49 2017

@author: schwarz
"""

#General imports
import unittest
import numpy as np

#BLonD imports
import blond.llrf.rf_modulation as rfMod
import blond.utils.exceptions as blExcept


class TestRFModulation(unittest.TestCase):
    
        
    def test_construct(self):
                
        stringMsg = "String input should raise an InputDataError exception"
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation('a', 1, 1)
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(1, 'a', 1)    
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(1, 1, 'a')
        with self.assertRaises(blExcept.InputDataError, msg=stringMsg):
            rfMod.PhaseModulation(1, 1, 1, 'a')    
        with self.assertRaises(blExcept.InputDataError, \
                               msg="non-integer system number should raise \
                                       InputDataError exception"):
            rfMod.PhaseModulation(1, 1, 1, 1, 0.1)
    
    
    def test_extender(self):
        
        testFreqProg = [[0, 1], [1E3, 5E2]]
        testAmpProg = [[0, 0.5, 1], [0, 1, 0]]
        testOffsetProg = [[0, 1], [0, np.pi]]
        testMultProg = 2
        n_rf = 5
        
        self.modulator = rfMod.PhaseModulation(testFreqProg, testAmpProg, \
                                               testOffsetProg, testMultProg,\
                                               system=None)
        self.modulator.n_rf = n_rf
        
        self.assertEqual(self.modulator.multiplier, tuple([testMultProg]*n_rf))

        
        
if __name__ == '__main__':

    unittest.main()