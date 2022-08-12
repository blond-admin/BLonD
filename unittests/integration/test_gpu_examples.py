# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration tests, execute all example GPU main files. 
:Authors: **Konstantinos Iliakis**
"""

import unittest
import pytest
import os
import subprocess

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = os.path.join(this_directory, '../../__EXAMPLES/gpu_main_files')
exec_args = ['python']
timeout = 90    # Timeout in seconds


class TestGPUExamples(unittest.TestCase):

    def _runExample(self, example, timeout=timeout):
        file = os.path.join(main_files_dir, example)
        env = os.environ.copy()
        env['USE_GPU'] = '1'
        try:
            ret = subprocess.run(exec_args + [file], timeout=timeout, env=env)
            self.assertEqual(ret.returncode, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest(
                '[{}] Timed out (timeout={}s)'.format(example, e.timeout))

    # Run before every test

    def setUp(self):
        pytest.importorskip('cupy')

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        example = 'EX_01_Acceleration.py'
        self._runExample(example)

    def test_EX_02_Main_long_ps_booster(self):
        example = 'EX_02_Main_long_ps_booster.py'
        self._runExample(example)

    def test_EX_03_RFnoise(self):
        example = 'EX_03_RFnoise.py'
        self._runExample(example)

    def test_EX_04_Stationary_multistation(self):
        example = 'EX_04_Stationary_multistation.py'
        self._runExample(example)

    def test_EX_05_Wake_impedance(self):
        example = 'EX_05_Wake_impedance.py'
        self._runExample(example)

    def test_EX_07_Ions(self):
        example = 'EX_07_Ions.py'
        self._runExample(example)

    def test_EX_08_Phase_Loop(self):
        example = 'EX_08_Phase_Loop.py'
        self._runExample(example)

    def test_EX_09_Radial_Loop(self):
        example = 'EX_09_Radial_Loop.py'
        self._runExample(example)

    def test_EX_10_Fixed_frequency(self):
        example = 'EX_10_Fixed_frequency.py'
        self._runExample(example)

    def test_EX_16_impedance_test(self):
        example = 'EX_16_impedance_test.py'
        self._runExample(example)

    def test_EX_17_multi_turn_wake(self):
        example = 'EX_17_multi_turn_wake.py'
        self._runExample(example)

    def test_EX_18_robinson_instability(self):
        example = 'EX_18_robinson_instability.py'
        self._runExample(example)


if __name__ == '__main__':

    unittest.main()
