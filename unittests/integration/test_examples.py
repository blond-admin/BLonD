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
from functools import wraps


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = '__EXAMPLES/main_files'
main_files_pattern = 'EX_*.py'


class TestExamples(unittest.TestCase):

    # Run before every test
    def setUp(self):
        os.chdir(this_directory + '../../')
        self.main_files = glob.glob(os.path.join(
            main_files_dir, main_files_pattern))

        self.output = open(this_directory + 'output.txt', 'w')
    
    # Run after every test
    def tearDown(self):
        self.output.close()


    def test_all_examples(self):

        for file in self.main_files:
            self.output.write('\nTesting file: ' + file + '\n')
            with self.subTest(file=file):
                try:
                    ret = subprocess.call(['python', file], timeout=60,
                                          stdout=self.output, stderr=subprocess.STDOUT)
                    self.output.flush()
                    self.assertEqual(ret, 0)
                except subprocess.TimeoutExpired as e:
                    self.output.write(
                        'Timed out (timeout={}s)\n'.format(e.timeout))
            self.output.write('Finished Testing.\n\n')



if __name__ == '__main__':

    unittest.main()
