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

import os
import sys
import unittest

import pytest

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
main_files_dir = os.path.join(
    this_directory, "../../__EXAMPLES/gpu_main_files"
)

os.environ["BLOND_EXAMPLES_DRAFT_MODE"] = "1"


class TestGPUExamples(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("cupy")

        if main_files_dir not in sys.path:
            sys.path.append(main_files_dir)

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        pass  # NOQA, see main_files_dir

    def test_EX_02_Main_long_ps_booster(self):
        pass  # NOQA, see main_files_dir

    def test_EX_03_RFnoise(self):
        pass  # NOQA, see main_files_dir

    def test_EX_04_Stationary_multistation(self):
        pass  # NOQA, see main_files_dir

    def test_EX_05_Wake_impedance(self):
        pass  # NOQA, see main_files_dir

    def test_EX_07_Ions(self):
        pass  # NOQA, see main_files_dir

    def test_EX_08_Phase_Loop(self):
        pass  # NOQA, see main_files_dir

    def test_EX_09_Radial_Loop(self):
        pass  # NOQA, see main_files_dir

    def test_EX_10_Fixed_frequency(self):
        pass  # NOQA, see main_files_dir

    def test_EX_16_impedance_test(self):
        pass  # NOQA, see main_files_dir

    def test_EX_17_multi_turn_wake(self):
        pass  # NOQA, see main_files_dir

    def test_EX_18_robinson_instability(self):
        pass  # NOQA, see main_files_dir


if __name__ == "__main__":
    unittest.main()
