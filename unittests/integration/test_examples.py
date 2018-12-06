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
import glob
import subprocess
# from functools import wraps
from parameterized import parameterized

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = '__EXAMPLES/main_files'
main_files_pattern = 'EX_*.py'

os.chdir(this_directory + '../../')

main_files = glob.glob(os.path.join(main_files_dir, main_files_pattern))


class TestExamples(unittest.TestCase):

    # Run before every test
    def setUp(self):
        pass
        # self.output = open(this_directory + 'output.txt', 'w')

    # Run after every test
    def tearDown(self):
        pass
        # self.output.close()

    @parameterized.expand(main_files)
    def test_example(self, file):

        # self.output.write('\nTesting file: ' + file + '\n')
        try:
            ret = subprocess.call(['python', file], timeout=60)
                                  # stdout=self.output, stderr=subprocess.STDOUT)
            # self.output.flush()
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            # self.output.write(
            #     'Timed out (timeout={}s)\n'.format(e.timeout))
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))
        # self.output.write('Finished Testing.\n\n')


if __name__ == '__main__':

    unittest.main()
