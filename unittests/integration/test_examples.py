# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for utils.bmath

:Authors: **Konstantinos Iliakis**
"""

import unittest
import numpy as np
import os
import subprocess
# import glob
# from functools import wraps
# from parameterized import parameterized

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = '__EXAMPLES/main_files'
timeout = 60
# main_files_pattern = 'EX_*.py'
# os.chdir(this_directory + '../../')
# main_files = glob.glob(os.path.join(main_files_dir, main_files_pattern))


class TestExamples(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        file = main_files_dir + '/EX_01_Acceleration.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_02_Main_long_ps_booster(self):
        file = main_files_dir + '/EX_02_Main_long_ps_booster.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_03_RFnoise(self):
        file = main_files_dir + '/EX_03_RFnoise.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_04_Stationary_multistation(self):
        file = main_files_dir + '/EX_04_Stationary_multistation.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_05_Wake_impedance(self):
        file = main_files_dir + '/EX_05_Wake_impedance.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_06_Preprocess(self):
        file = main_files_dir + '/EX_06_Preprocess.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_07_Ions(self):
        file = main_files_dir + '/EX_07_Ions.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_08_Phase_Loop(self):
        file = main_files_dir + '/EX_08_Phase_Loop.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_09_Radial_Loop(self):
        file = main_files_dir + '/EX_09_Radial_Loop.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_10_Fixed_frequency(self):
        file = main_files_dir + '/EX_10_Fixed_frequency.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_11_comparison_music_fourier_analytical(self):
        file = main_files_dir + '/EX_11_comparison_music_fourier_analytical.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_12_synchrotron_frequency_distribution(self):
        file = main_files_dir + '/EX_12_synchrotron_frequency_distribution.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_13_synchrotron_radiation(self):
        file = main_files_dir + '/EX_13_synchrotron_radiation.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_14_sparse_slicing(self):
        file = main_files_dir + '/EX_14_sparse_slicing.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_15_sparse_multi_bunch(self):
        file = main_files_dir + '/EX_15_sparse_multi_bunch.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_16_impedance_test(self):
        file = main_files_dir + '/EX_16_impedance_test.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_17_multi_turn_wake(self):
        file = main_files_dir + '/EX_17_multi_turn_wake.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_18_robinson_instability(self):
        file = main_files_dir + '/EX_18_robinson_instability.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_19_bunch_generation(self):
        file = main_files_dir + '/EX_19_bunch_generation.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_20_bunch_generation_multibunch(self):
        file = main_files_dir + '/EX_20_bunch_generation_multibunch.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_21_bunch_distribution(self):
        file = main_files_dir + '/EX_21_bunch_distribution.py'
        try:
            ret = subprocess.call(['python', file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    # @parameterized.expand(main_files)
    # def test_example(self, file):

    #     # self.output.write('\nTesting file: ' + file + '\n')
    #     try:
    #         ret = subprocess.call(['python', file], timeout=60)
    #                               # stdout=self.output, stderr=subprocess.STDOUT)
    #         # self.output.flush()
    #         self.assertEqual(ret, 0)
    #     except subprocess.TimeoutExpired as e:
    #         # self.output.write(
    #         #     'Timed out (timeout={}s)\n'.format(e.timeout))
    #         raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))
    #     # self.output.write('Finished Testing.\n\n')


if __name__ == '__main__':

    unittest.main()
