# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Cross check the test files generated by the GPU and non-GPU examples.

:Authors: **Konstantinos Iliakis**
"""

import unittest
import os
import numpy as np

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
out_dir = os.path.join(this_directory, '../../__EXAMPLES/output_files/')
gpu_out_dir = os.path.join(this_directory, '../../__EXAMPLES/gpu_output_files/')


class TestGPUCorrectness(unittest.TestCase):

    def _testFilesClose(self, example, rtol=1e-10, atol=0):
        file = os.path.join(out_dir, '{}_test_data.txt'.format(example))
        gpu_file = os.path.join(gpu_out_dir, '{}_test_data.txt'.format(example))

        # Check if both files exist
        if os.path.isfile(file) and os.path.isfile(gpu_file):
            # Parse the regular blond data
            data = np.genfromtxt(file, dtype=str, delimiter='\t')
            if len(data) == 0:
                raise unittest.SkipTest('[{}] Non-GPU test file empty'.format(example))
            header, data = data[0], np.array(data[1:], dtype=float)

            # Parse the GPU blond data
            datagpu = np.genfromtxt(gpu_file, dtype=str, delimiter='\t')
            if len(datagpu) == 0:
                raise unittest.SkipTest('[{}] GPU test file empty'.format(example))
            headergpu, datagpu = datagpu[0], np.array(datagpu[1:], dtype=float)

            # Make sure the headers agree
            np.testing.assert_equal(header, headergpu,
                                    err_msg='[{}] The headers of the test files disagree'.format(example))

            # Compare the bodies of the text files
            for idx, row, row_gpu in zip(np.arange(len(data)), data, datagpu):
                for h, x, xgpu in zip(header, row, row_gpu):
                    np.testing.assert_allclose([xgpu], [x], atol=atol, rtol=rtol,
                                               err_msg='[{}] Test failed in row {}, column {}'.format(example, idx, h))
        else:
            # otherwise skip the test
            raise unittest.SkipTest(
                '[{}] GPU and/or non-GPU test file missing'.format(example))

    # Run before every test

    def setUp(self):
        pass

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        example = 'EX_01'
        self._testFilesClose(example)

    def test_EX_02_Main_long_ps_booster(self):
        example = 'EX_02'
        self._testFilesClose(example)

    def test_EX_03_RFnoise(self):
        example = 'EX_03'
        self._testFilesClose(example)

    def test_EX_04_Stationary_multistation(self):
        example = 'EX_04'
        self._testFilesClose(example)

    def test_EX_05_Wake_impedance(self):
        example = 'EX_05'
        self._testFilesClose(example)

    def test_EX_07_Ions(self):
        example = 'EX_07'
        self._testFilesClose(example)

    def test_EX_08_Phase_Loop(self):
        example = 'EX_08'
        self._testFilesClose(example)

    def test_EX_09_Radial_Loop(self):
        example = 'EX_09'
        self._testFilesClose(example)

    def test_EX_10_Fixed_frequency(self):
        example = 'EX_10'
        self._testFilesClose(example)

    def test_EX_16_impedance_test(self):
        example = 'EX_16'
        self._testFilesClose(example)

    def test_EX_17_multi_turn_wake(self):
        example = 'EX_17'
        self._testFilesClose(example)

    def test_EX_18_robinson_instability(self):
        example = 'EX_18'
        self._testFilesClose(example)


if __name__ == '__main__':

    unittest.main()
