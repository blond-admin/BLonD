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

import importlib.util
import os
import subprocess
import sys
import unittest

import pytest

os.environ["BLOND_EXAMPLES_DRAFT_MODE"] = "1"

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
main_files_dir = os.path.join(this_directory + '../../__EXAMPLES/mpi_main_files')
exec_args = ['mpirun', '-n', '2', sys.executable]


class TestMpiExamples(unittest.TestCase):

    def _runMPIExample(self, example):

        file = os.path.join(main_files_dir, example)

        # Create a module spec from the file
        spec = importlib.util.spec_from_file_location("module.name", file)
        module = importlib.util.module_from_spec(spec)

        # Execute the module
        sys.modules["module.name"] = module
        spec.loader.exec_module(module)

        return module  # Return the module if you need to access its attributes

    # Run before every test

    def setUp(self):
        pytest.importorskip('mpi4py')
        try:
            subprocess.call(['mpirun', '--version'])
        except FileNotFoundError:
            unittest.SkipTest('mpirun not found, skipping tests')

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        example = 'EX_01_Acceleration.py'
        self._runMPIExample(example)

    def test_EX_02_Main_long_ps_booster(self):
        example = 'EX_02_Main_long_ps_booster.py'
        self._runMPIExample(example)

    def test_EX_03_RFnoise(self):
        example = 'EX_03_RFnoise.py'
        self._runMPIExample(example)

    def test_EX_04_Stationary_multistation(self):
        example = 'EX_04_Stationary_multistation.py'
        self._runMPIExample(example)

    def test_EX_05_Wake_impedance(self):
        example = 'EX_05_Wake_impedance.py'
        self._runMPIExample(example)

    def test_EX_07_Ions(self):
        example = 'EX_07_Ions.py'
        self._runMPIExample(example)

    def test_EX_08_Phase_Loop(self):
        example = 'EX_08_Phase_Loop.py'
        self._runMPIExample(example)

    def test_EX_09_Radial_Loop(self):
        example = 'EX_09_Radial_Loop.py'
        self._runMPIExample(example)

    def test_EX_10_Fixed_frequency(self):
        example = 'EX_10_Fixed_frequency.py'
        self._runMPIExample(example)

    def test_EX_16_impedance_test(self):
        example = 'EX_16_impedance_test.py'
        self._runMPIExample(example)

    def test_EX_18_robinson_instability(self):
        example = 'EX_18_robinson_instability.py'
        self._runMPIExample(example)


if __name__ == '__main__':
    unittest.main()
