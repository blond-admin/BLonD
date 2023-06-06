# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test preprocess.py

'''

import sys
import unittest

import numpy as np

from blond.input_parameters.ring_options import RingOptions


class test_preprocess(unittest.TestCase):

    def setUp(self):

        if int(sys.version[0]) == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """

        standardMsg = "%s is not NaN" % str(value)

        if not np.isnan(value):
            self.fail(self._formatMessage(msg, standardMsg))

    def test_interpolation_type_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'ERROR: Interpolation scheme in PreprocessRamp not recognised. ' +
            'Aborting...',
                msg='No RuntimeError for wrong interpolation scheme!'):

            RingOptions(interpolation='exponential')

    def test_flat_bottom_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'ERROR: flat_bottom value in PreprocessRamp not recognised. ' +
            'Aborting...',
                msg='No RuntimeError for negative flat_bottom!'):

            RingOptions(flat_bottom=-42)

    def test_flat_top_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'ERROR: flat_top value in PreprocessRamp not recognised. ' +
            'Aborting...',
                msg='No RuntimeError for negative flat_top!'):

            RingOptions(flat_top=-42)

    def test_plot_option_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'ERROR: plot value in PreprocessRamp not recognised. ' +
            'Aborting...',
                msg='No RuntimeError for wrong plot option!'):

            RingOptions(plot=42)

    def test_sampling_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'ERROR: sampling value in PreprocessRamp not recognised. ' +
            'Aborting...',
                msg='No RuntimeError for wrong sampling!'):

            RingOptions(sampling=0)


if __name__ == '__main__':

    unittest.main()
