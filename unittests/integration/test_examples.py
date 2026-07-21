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
import sys
import unittest

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
main_files_dir = os.path.join(this_directory, "../../__EXAMPLES/main_files")

os.environ["BLOND_EXAMPLES_DRAFT_MODE"] = "1"


class TestExamples(unittest.TestCase):
    def setUp(self):
        if main_files_dir not in sys.path:
            sys.path.append(main_files_dir)

    # Run after every test
    def tearDown(self):
        pass

    def test_EX_01_Acceleration(self):
        import EX_01_Acceleration  # NOQA

    def test_EX_02_Main_long_ps_booster(self):
        import EX_02_Main_long_ps_booster  # NOQA

    def test_EX_03_RFnoise(self):
        import EX_03_RFnoise  # NOQA

    def test_EX_04_Stationary_multistation(self):
        import EX_04_Stationary_multistation  # NOQA

    def test_EX_05_Wake_impedance(self):
        import EX_05_Wake_impedance  # NOQA

    def test_EX_06_Preprocess(self):
        import EX_06_Preprocess  # NOQA

    def test_EX_07_Ions(self):
        import EX_07_Ions  # NOQA

    def test_EX_08_Phase_Loop(self):
        import EX_08_Phase_Loop  # NOQA

    def test_EX_09_Radial_Loop(self):
        import EX_09_Radial_Loop  # NOQA

    def test_EX_10_Fixed_frequency(self):
        import EX_10_Fixed_frequency  # NOQA

    def test_EX_11_comparison_music_fourier_analytical(self):
        import EX_11_comparison_music_fourier_analytical  # NOQA

    def test_EX_12_synchrotron_frequency_distribution(self):
        import EX_12_synchrotron_frequency_distribution  # NOQA

    def test_EX_13_synchrotron_radiation(self):
        import EX_13_synchrotron_radiation  # NOQA

    def test_EX_14_sparse_slicing(self):
        import EX_14_sparse_slicing  # NOQA

    def test_EX_15_sparse_multi_bunch(self):
        import EX_15_sparse_multi_bunch  # NOQA

    def test_EX_16_impedance_test(self):
        import EX_16_impedance_test  # NOQA

    def test_EX_17_multi_turn_wake(self):
        import EX_17_multi_turn_wake  # NOQA

    def test_EX_18_robinson_instability(self):
        import EX_18_robinson_instability  # NOQA

    def test_EX_19_bunch_generation(self):
        import EX_19_bunch_generation  # NOQA

    @unittest.skip(
        "Implement a faster way of execution, otherwise this test would take several minutes"
    )
    def test_EX_20_bunch_generation_multibunch(self):
        import EX_20_bunch_generation_multibunch  # NOQA

    def test_EX_21_bunch_distribution(self):
        import EX_21_bunch_distribution  # NOQA

    def test_EX_22_Coherent_Radiation(self):
        import EX_22_Coherent_Radiation  # NOQA


if __name__ == "__main__":
    unittest.main()
