# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test general_parameters.py

'''

import unittest
import numpy as np

from input_parameters.general_parameters import GeneralParameters
from beams.beams import Electron

class test_general_parameters(unittest.TestCase):
    
    def setUp(self):
        self.n_turns = 10
        self.C = [13000, 13659] 
        self.num_sections = 2
        self.alpha = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
        self.momentum = [[450e9 for j in range(self.n_turns+1)] 
                        for i in range(self.num_sections)]
        self.particle = Electron()
            
    def test_kinetic_energy_positive(self):
        # kinetic energy must be greater or equal 0 for all turns
        general_parameters = GeneralParameters(self.n_turns, self.C, 
                                self.alpha, self.momentum, 
                                n_sections = self.num_sections,
                                Particle = self.particle)
        self.assertTrue((general_parameters.kin_energy>=0.0).all(),
                        msg = 'kinetic energy is negative!')

    def test_cycle_time_turn1(self):
        # cycle_time[0] must be equal to t_rev[0]
        general_parameters = GeneralParameters(self.n_turns, self.C, 
                                self.alpha, self.momentum, 
                                n_sections = self.num_sections,
                                Particle = self.particle)
        self.assertEqual(general_parameters.cycle_time[0],
                         general_parameters.t_rev[0],
                         msg = 'cycle_time at first turn not equal to'
                             +'revolution time at first turn!')
    
    def test_ring_length_exception(self):
        # test if 'ring length size' RuntimeError gets thrown for wrong number
        # of rf-sections
        num_sections = 1    # only one rf-section!
        
        with self.assertRaisesRegex(RuntimeError,'ERROR in GeneralParameters: '
                +'Number of sections and ring length size do not match!',
                msg = 'No RuntimeError for wrong n_sections!'):
            GeneralParameters(self.n_turns, self.C, self.alpha, self.momentum, 
                              Particle = self,
                              n_sections = num_sections)

    def test_alpha_shape_exception(self):
        # test if 'momentum compaction' RuntimeError gets thrown for wrong 
        # shape of alpha
        alpha = [[3.21e-4, 2.e-5, 5.e-7]]   # only one array!
        
        with self.assertRaisesRegex(RuntimeError,'ERROR in GeneralParameters: '
            +'Number of sections and size of momentum compaction do not match!'
            ,msg = 'No RuntimeError for wrong shape of alpha!'):
            GeneralParameters(self.n_turns, self.C, alpha, self.momentum, 
                              Particle = self.particle,
                              n_sections = self.num_sections)
    
    def test_synchronous_data_exception(self):
        # what to do when user wants momentum programm for multiple sections?
        # one array of cycle_time? one array per section?
        # Currently, __init__ checks only if the number of cycle_time arrays is
        # the same as the number of momentum_arrays, but not if the array
        # have the correct length.
        cycle_time = np.linspace(0,1,self.n_turns) #wrong length
        with self.assertRaisesRegex(RuntimeError,'ERROR in GeneralParameters: '
            +'sychronous data does not match the time data'
            ,msg = 'No RuntimeError for wrong synchronous_data!'):
            GeneralParameters(self.n_turns, self.C, self.alpha, 
                          ([cycle_time,cycle_time], self.momentum), 
                              Particle = self.particle,
                              n_sections = self.num_sections)
    
    def test_momentum_shape_exception(self):
        # test if RuntimeError gets thrown for wrong shape of momentum
        momentum = 450e9 # only one momentum!
        
        with self.assertRaisesRegex(RuntimeError,'ERROR in GeneralParameters: '
            +'Number of sections and momentum data do not match!',
            msg = 'No RuntimeError for wrong shape of momentum!'):
            GeneralParameters(self.n_turns, self.C, self.alpha, momentum, 
                              Particle = self.particle,
                              n_sections = self.num_sections)
    
    def test_momentum_length_exception(self):
        # test if RuntimeError gets thrown for wrong length of momentum
        # only n_turns elements per section!
        momentum = [[450e9 for j in range(self.n_turns)] 
                    for i in range(self.num_sections)]
        
        with self.assertRaisesRegex(RuntimeError,'ERROR in GeneralParameters: '
            +'The momentum program does not match the proper length '
            +r'\(n_turns\+1\)',
            msg = 'No RuntimeError for wrong length of momentum array!'):
            GeneralParameters(self.n_turns, self.C, self.alpha, momentum, 
                              Particle = self.particle,
                              n_sections = self.num_sections)

    
if __name__ == '__main__':
    unittest.main()