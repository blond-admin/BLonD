# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Unit-tests for the Beam Profile class.
   Run as python testBeamProfileObject.py in console or via travis. **

:Authors: **Danilo Quartullo**
'''

# General imports
# -----------------
from __future__ import division, print_function

import os
import unittest

import numpy as np

import blond.beam.profile as profileModule
# BLonD imports
# --------------
from blond.beam.beam import Beam, Proton
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring

# import matplotlib.pyplot as plt


class testProfileClass(unittest.TestCase):

    # Run before every test
    def setUp(self):
        """
        Slicing of the same Gaussian profile using four distinct settings to
        test different features.
        """

        # Ring parameters
        n_turns = 1
        ring_length = 125
        alpha = 0.001
        momentum = 1e9

        # Ring object initialization
        self.ring = Ring(ring_length, alpha, momentum, Proton(), n_turns)

        # RF object initialization
        self.rf_params = RFStation(Ring=self.ring, harmonic=[1],
                                   voltage=[7e6], phi_rf_d=[0.],
                                   n_rf=1)

        # Beam parameters
        n_macroparticles = 100000
        intensity = 1e10

        # Beam object parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))

        my_beam = Beam(self.ring, n_macroparticles, intensity)
        my_beam.dt = np.load(dir_path + '/dt_coordinates.npz')['arr_0']

        # First profile object initialization and tracking
        self.profile1 = profileModule.Profile(my_beam)
        self.profile1.track()

        # Second profile object initialization and tracking
        n_slices = 200
        CutOptions = profileModule.CutOptions(
            cut_left=0, cut_right=2 * np.pi,
            n_slices=n_slices,
            cuts_unit='rad',
            RFSectionParameters=self.rf_params)

        FitOptions = profileModule.FitOptions(
            fit_option='fwhm',
            fitExtraOptions=None)

        FilterOptions = profileModule.FilterOptions(
            filterMethod=None,
            filterExtraOptions=None)

        OtherSlicesOptions = profileModule.OtherSlicesOptions(
            smooth=False,
            direct_slicing=True)

        self.profile2 = profileModule.Profile(
            my_beam,
            CutOptions=CutOptions,
            FitOptions=FitOptions,
            FilterOptions=FilterOptions,
            OtherSlicesOptions=OtherSlicesOptions)

        # Third profile object initialization and tracking
        n_slices = 150
        CutOptions = profileModule.CutOptions(
            cut_left=0,
            cut_right=self.ring.t_rev[0],
            n_slices=n_slices,
            cuts_unit='s')

        FitOptions = profileModule.FitOptions(
            fit_option='rms',
            fitExtraOptions=None)

        FilterOptions = profileModule.FilterOptions(
            filterMethod=None,
            filterExtraOptions=None)

        OtherSlicesOptions = profileModule.OtherSlicesOptions(
            smooth=True,
            direct_slicing=False)

        self.profile3 = profileModule.Profile(
            my_beam,
            CutOptions=CutOptions,
            FitOptions=FitOptions,
            FilterOptions=FilterOptions,
            OtherSlicesOptions=OtherSlicesOptions)

        self.profile3.track()

        # Fourth profile object initialization and tracking
        n_slices = 100
        CutOptions = profileModule.CutOptions(
            cut_left=0,
            cut_right=self.ring.t_rev[0],
            n_slices=n_slices,
            cuts_unit='s')

        FitOptions = profileModule.FitOptions(
            fit_option='gaussian',
            fitExtraOptions=None)

        filter_option = {'pass_frequency': 1e7,
                         'stop_frequency': 1e8,
                         'gain_pass': 1,
                         'gain_stop': 2,
                         'transfer_function_plot': False}

        FilterOptions = profileModule.FilterOptions(
            filterMethod='chebishev',
            filterExtraOptions=filter_option)

        OtherSlicesOptions = profileModule.OtherSlicesOptions(
            smooth=False,
            direct_slicing=True)

        self.profile4 = profileModule.Profile(
            my_beam,
            CutOptions=CutOptions,
            FitOptions=FitOptions,
            FilterOptions=FilterOptions,
            OtherSlicesOptions=OtherSlicesOptions)

    def test(self):
        rtol = 1e-6             # relative tolerance
        atol = 0                # absolute tolerance
        delta = 1e-14
        self.assertAlmostEqual(self.ring.t_rev[0], 5.71753954209e-07,
                               delta=delta, msg='Ring: t_rev[0] not correct')

        self.assertSequenceEqual(
            self.profile1.n_macroparticles.tolist(),
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 5.0,
             7.0, 10.0, 9.0, 21.0, 20.0, 35.0, 40.0, 59.0, 70.0, 97.0, 146.0,
             164.0, 183.0, 246.0, 324.0, 412.0, 422.0, 523.0, 637.0, 797.0,
             909.0, 1074.0, 1237.0, 1470.0, 1641.0, 1916.0, 1991.0, 2180.0,
             2464.0, 2601.0, 3010.0, 3122.0, 3233.0, 3525.0, 3521.0, 3657.0,
             3650.0, 3782.0, 3845.0, 3770.0, 3763.0, 3655.0, 3403.0, 3433.0,
             3066.0, 3038.0, 2781.0, 2657.0, 2274.0, 2184.0, 1934.0, 1724.0,
             1452.0, 1296.0, 1111.0, 985.0, 803.0, 694.0, 546.0, 490.0, 429.0,
             324.0, 239.0, 216.0, 147.0, 122.0, 87.0, 83.0, 62.0, 48.0, 28.0,
             26.0, 14.0, 18.0, 12.0, 6.0, 7.0, 4.0, 2.0, 2.0, 1.0, 1.0, 2.0,
             0.0, 0.0, 0.0, 0.0],
            msg="Profile1 not correct")

        self.assertSequenceEqual(
            self.profile2.n_macroparticles.tolist(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
             0.0, 0.0, 1.0, 0.0, 4.0, 6.0, 11.0, 13.0, 24.0, 31.0, 50.0, 71.0,
             99.0, 154.0, 212.0, 246.0, 365.0, 502.0, 560.0, 741.0, 942.0,
             1183.0, 1436.0, 1824.0, 2114.0, 2504.0, 2728.0, 3132.0, 3472.0,
             4011.0, 4213.0, 4592.0, 4667.0, 4742.0, 4973.0, 4973.0, 5001.0,
             4737.0, 4470.0, 4245.0, 4031.0, 3633.0, 3234.0, 2877.0, 2540.0,
             2138.0, 1754.0, 1466.0, 1224.0, 966.0, 746.0, 632.0, 468.0, 347.0,
             272.0, 178.0, 122.0, 100.0, 75.0, 49.0, 31.0, 18.0, 20.0, 10.0,
             9.0, 4.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            msg="Profile2 not correct")

        np.testing.assert_allclose(
            self.profile3.n_macroparticles.tolist(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.823322140525,
             0.176677859475, 0.0, -0.255750723657, 1.58255526749, 5.591836814,
             8.52273531218, 26.7796551186, 34.3314288768, 51.0732749092,
             96.1902354197, 157.367580381, 225.376283353, 374.772960616,
             486.894110696, 747.13998452, 949.483971664, 1340.35510048,
             1824.62356727, 2240.79797642, 2904.90319468, 3577.06628686,
             4007.96759936, 4751.84403436, 5592.30491982, 5865.72609993,
             6254.85130914, 6437.0578678, 6667.35522794, 6505.87820056,
             6175.35102937, 5744.45114657, 5166.14734563, 4570.9693115,
             3838.66240227, 3311.19755852, 2643.39729925, 2057.56970401,
             1570.88713405, 1167.90898075, 812.225096261, 674.455899421,
             401.231764382, 254.280874938, 178.332275974, 128.130564109,
             70.1303218524, 41.1867595808, 24.1789148058, 15.6635863931,
             8.08619267781, 2.85660788584, 4.39836119938, 1.71862177506,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rtol=rtol, atol=atol,
            err_msg="Profile3 not correct")

        np.testing.assert_allclose(
            self.profile4.n_macroparticles.tolist(),
            [3.04390342739e-10, 6.42161174461e-10, 1.3547439458e-09,
             2.8580537592e-09, 6.0295314962e-09, 1.27202820964e-08,
             2.68355139555e-08, 5.66139024119e-08, 1.19436279537e-07,
             2.51970351131e-07, 5.31572635175e-07, 1.12143934871e-06,
             2.36585958272e-06, 4.99116744172e-06, 1.05296834238e-05,
             2.22140880466e-05, 4.68642491784e-05, 9.88677926569e-05,
             0.000208577766554, 0.000440028886368, 0.000928312849628,
             0.00195842767028, 0.0041316232359, 0.0087163344465,
             0.0183885320237, 0.0387936135382, 0.0818414677914,
             0.172657951641, 0.364250166442, 0.768445255442, 1.62116085321,
             3.42010376585, 7.06934966864, 14.4070273867, 29.1322663172,
             57.0673409924, 107.367201129, 195.236438646, 343.216047763,
             576.683888858, 925.972171496, 1429.04196501, 2121.53804001,
             3024.03256787, 4107.04688385, 5292.37720345, 6501.56643031,
             7614.78027336, 8457.77133102, 8917.59371854, 8940.7377173,
             8510.50678563, 7697.68236675, 6617.05862601, 5396.53357642,
             4175.00669728, 3071.53152854, 2158.0602225, 1451.53603502,
             931.07854988, 569.13404729, 335.967612417, 192.376031593,
             106.641608202, 57.6152988997, 30.252175918, 15.4392857786,
             7.59503067718, 3.60011486661, 1.70648778177, 0.808891009658,
             0.383421828445, 0.181745496949, 0.0861490484128,
             0.0408354466385, 0.0193563798194, 0.00917510326339,
             0.00434908390304, 0.00206150604006, 0.000977172951353,
             0.000463189027003, 0.000219555887664, 0.000104071523714,
             4.93308658832e-05, 2.33832872042e-05, 1.10838946506e-05,
             5.25386869489e-06, 2.49038240919e-06, 1.18046432148e-06,
             5.59551018822e-07, 2.65232364082e-07, 1.25722596556e-07,
             5.95936750584e-08, 2.82479541435e-08, 1.33897925887e-08,
             6.34688728235e-09, 3.00848674436e-09, 1.42605775497e-09,
             6.75979730414e-10, 3.20452612975e-10],
            rtol=rtol, atol=atol,
            err_msg="Profile4 not correct")

        np.testing.assert_allclose(
            np.array([self.profile2.bunchPosition,
                      self.profile3.bunchPosition,
                      self.profile4.bunchPosition]),
            [2.86004598801e-07, 2.86942707778e-07, 2.86090181555e-07],
            rtol=rtol, atol=atol,
            err_msg='Bunch position values not correct')

        np.testing.assert_allclose(
            np.array([self.profile2.bunchLength,
                      self.profile3.bunchLength,
                      self.profile4.bunchLength]),
            [9.27853156526e-08, 9.24434506817e-08, 9.18544356769e-08],
            rtol=rtol, atol=atol,
            err_msg='Bunch length values not correct')


if __name__ == '__main__':

    unittest.main()
