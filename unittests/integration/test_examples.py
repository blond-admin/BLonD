# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration tests, execute all __EXAMPLES main files. 
:Authors: **Konstantinos Iliakis**
"""

import os
import subprocess
import sys
import unittest

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = os.path.join(this_directory, '../../__EXAMPLES/main_files')
exec_args = [sys.executable]
timeout = 60    # Timeout in seconds


class TestExamples(unittest.TestCase):

    def _runExample(self, example, timeout=timeout):
        file = os.path.join(main_files_dir, example)
        try:
            ret = subprocess.run(exec_args + ['-m', 'pip', 'show', 'blond'])
            ret = subprocess.run(exec_args + [file], timeout=timeout)
            self.assertEqual(ret.returncode, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest(
                '[{}] Timed out (timeout={}s)'.format(example, e.timeout))

    # Run before every test

    def setUp(self):
        pass

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

    def test_EX_06_Preprocess(self):
        example = 'EX_06_Preprocess.py'
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

    def test_EX_11_comparison_music_fourier_analytical(self):
        example = 'EX_11_comparison_music_fourier_analytical.py'
        self._runExample(example)

    def test_EX_12_synchrotron_frequency_distribution(self):
        example = 'EX_12_synchrotron_frequency_distribution.py'
        self._runExample(example)

    def test_EX_13_synchrotron_radiation(self):
        example = 'EX_13_synchrotron_radiation.py'
        self._runExample(example)

    def test_EX_14_sparse_slicing(self):
        example = 'EX_14_sparse_slicing.py'
        self._runExample(example)

    def test_EX_15_sparse_multi_bunch(self):
        example = 'EX_15_sparse_multi_bunch.py'
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

    def test_EX_19_bunch_generation(self):
        example = 'EX_19_bunch_generation.py'
        self._runExample(example)

    def test_EX_20_bunch_generation_multibunch(self):
        example = 'EX_20_bunch_generation_multibunch.py'
        self._runExample(example)

    def test_EX_21_bunch_distribution(self):
        example = 'EX_21_bunch_distribution.py'
        self._runExample(example)

    def test_EX_22_Coherent_Radiation(self):
        example = 'EX_22_Coherent_Radiation.py'
        self._runExample(example)


if __name__ == '__main__':

    unittest.main()
