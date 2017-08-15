# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test preprocess.py

'''
import unittest
import math


from input_parameters.preprocess import PreprocessRamp
from beam.beam import Proton


class test_preprocess(unittest.TestCase):
    
    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """
        standardMsg = "%s is not NaN" % str(value)
        try:
            if not math.isnan(value):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            self.fail(self._formatMessage(msg, standardMsg))
    
    def test_interpolation_type_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'ERROR: Interpolation scheme'
                +' in PreprocessRamp not recognised. Aborting...', 
                msg = 'No RuntimeError for wrong interpolation scheme!'):

            PreprocessRamp(interpolation='exponential')
    
    def test_flat_bottom_exception(self):
        with self.assertRaisesRegex(RuntimeError,
                            'ERROR: flat_bottom value in PreprocessRamp'+
                            ' not recognised. Aborting...', 
                            msg = 'No RuntimeError for negative flat_bottom!'):

            PreprocessRamp(flat_bottom = -42)

    def test_flat_top_exception(self):
        with self.assertRaisesRegex(RuntimeError,
                            'ERROR: flat_top value in PreprocessRamp'+
                            ' not recognised. Aborting...', 
                            msg = 'No RuntimeError for negative flat_top!'):

            PreprocessRamp(flat_top = -42)
    
    def test_plot_option_exception(self):
        with self.assertRaisesRegex(RuntimeError,
                'ERROR: plot value in PreprocessRamp'+
                ' not recognised. Aborting...', 
                msg = 'No RuntimeError for wrong plot option!'):

            PreprocessRamp(plot = 42)
    
    def test_sampling_exception(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'ERROR: sampling value in PreprocessRamp'+
                                    ' not recognised. Aborting...', 
                msg = 'No RuntimeError for wrong sampling!'):

            PreprocessRamp(sampling = 0)
    
    def test_convet_data_exception(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'ERROR in PreprocessRamp: Synchronous data'
                                    +' type not recognized!', 
                msg = 'No RuntimeError for wrong synchronous data type!'):

            PreprocessRamp.convert_data('dummy',25e9,
                    synchronous_data_type = 'somethingCompletelyDifferent')
    
    def test_convert_data_value_rest_mass(self):
        self.assertEqual(PreprocessRamp.convert_data('dummy',Proton().mass,
                                Particle = Proton(),
                                synchronous_data_type = 'total energy'),0.0,
                msg = 'Momentum not zero for total engery equal rest mass!')
    
    def test_convert_data_wrong_total_energy(self):
        #use energy 25 instead of 25e9
        self.assertIsNaN(PreprocessRamp.convert_data('dummy',25,
                                Particle = Proton(),
                                synchronous_data_type = 'total energy'),
                msg = 'No NaN for total energy less than rest mass!')
                
    def test_convert_data_wrong_kinetic_energy(self):
        #use negative kinetic energy
        self.assertIsNaN(PreprocessRamp.convert_data('dummy',-25,
                                Particle = Proton(),
                                synchronous_data_type = 'kinetic energy'),
                msg = 'No NaN for total energy less than rest mass!')
    
    
if __name__ == '__main__':

    unittest.main()