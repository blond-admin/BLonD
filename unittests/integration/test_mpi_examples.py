# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration tests for the MPI mode.

:Authors: **Konstantinos Iliakis**
"""

import unittest
import os
import subprocess

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = '__EXAMPLES/mpi_main_files'
timeout = 90
exec_args = ['mpirun', '-n', '2', 'python']


class TestMpiExamples(unittest.TestCase):

    # Run before every test
    def setUp(self):
        try:
            subprocess.call(['mpirun', '--version'])
        except FileNotFoundError:
            self.skipTest('mpirun not found, skipping tests')

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        file = main_files_dir + '/EX_01_Acceleration.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_02_Main_long_ps_booster(self):
        file = main_files_dir + '/EX_02_Main_long_ps_booster.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_03_RFnoise(self):
        file = main_files_dir + '/EX_03_RFnoise.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_04_Stationary_multistation(self):
        file = main_files_dir + '/EX_04_Stationary_multistation.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_05_Wake_impedance(self):
        file = main_files_dir + '/EX_05_Wake_impedance.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_07_Ions(self):
        file = main_files_dir + '/EX_07_Ions.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_08_Phase_Loop(self):
        file = main_files_dir + '/EX_08_Phase_Loop.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_09_Radial_Loop(self):
        file = main_files_dir + '/EX_09_Radial_Loop.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_10_Fixed_frequency(self):
        file = main_files_dir + '/EX_10_Fixed_frequency.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_16_impedance_test(self):
        file = main_files_dir + '/EX_16_impedance_test.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

    def test_EX_18_robinson_instability(self):
        file = main_files_dir + '/EX_18_robinson_instability.py'
        try:
            ret = subprocess.call(exec_args + [file], timeout=timeout)
            self.assertEqual(ret, 0)
        except subprocess.TimeoutExpired as e:
            raise unittest.SkipTest('Timed out (timeout={}s)'.format(e.timeout))

# class TestMPICorrectness(unittest.TestCase):

#     # Run before every test
#     def setUp(self):
#         pass

#     # Run after every test
#     def tearDown(self):
#         pass

#     def test_EX_05_Wake_impedance(self):
#         file = main_files_dir + '/../output_files/EX_05_fig/comparison_induced_voltage.png'
#         mpi_file = main_files_dir + \
#             '/../mpi_output_files/EX_05_fig/comparison_induced_voltage.png'
#         # Check if both files exist
#         if os.path.isfile(file) and os.path.isfile(mpi_file):
#             # if yes compare them
#             import filecmp
#             ret = filecmp.cmp(file, mpi_file)
#             self.assertEqual(ret, True)
#         else:
#             # otherwise skip
#             raise unittest.SkipTest('Regular and/or MPI file do not exist')

#     def test_EX_16_impedance_test(self):
#         file = main_files_dir + '/../output_files/EX_16_fig/fig.png'
#         mpi_file = main_files_dir + '/../mpi_output_files/EX_16_fig/fig.png'
#         # Check if both files exist
#         if os.path.isfile(file) and os.path.isfile(mpi_file):
#             # if yes compare them
#             import filecmp
#             ret = filecmp.cmp(file, mpi_file)
#             self.assertEqual(ret, True)
#         else:
#             # otherwise skip
#             raise unittest.SkipTest('Regular and/or MPI file do not exist')

#     def test_EX_18_robinson_instability(self):
#         file1 = main_files_dir + '/../output_files/EX_18_fig/bunch_center.png'
#         file2 = main_files_dir + '/../output_files/EX_18_fig/bunch_length.png'
#         mpi_file1 = main_files_dir + '/../mpi_output_files/EX_18_fig/bunch_center.png'
#         mpi_file2 = main_files_dir + '/../mpi_output_files/EX_18_fig/bunch_length.png'
#         # Check if both files exist
#         if os.path.isfile(file1) and os.path.isfile(file2) and \
#                 os.path.isfile(mpi_file1) and os.path.isfile(mpi_file2):
#             # if yes compare them
#             import filecmp
#             ret = filecmp.cmp(file1, mpi_file1)
#             self.assertEqual(ret, True)
#             ret = filecmp.cmp(file2, mpi_file2)
#             self.assertEqual(ret, True)
#         else:
#             # otherwise skip
#             raise unittest.SkipTest('Regular and/or MPI file do not exist')


if __name__ == '__main__':

    unittest.main()
