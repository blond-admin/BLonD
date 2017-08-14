# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Unit-tests for the Beam Profile class.

Run as python testBeamProfileObject.py in console or via travis
'''

# General imports
# -----------------
from __future__ import division, print_function
import unittest
import numpy as np
import matplotlib.pyplot as plt

# BLonD imports
# --------------
from beam.beam import Beam
from input_parameters.ring import Ring
import beam.profile as profileModule


class testProfileClass(unittest.TestCase):
    
    # Run before every test
    def setUp(self):
        
        # Machine parameters
        n_turns = 1
        ring_length = 125
        alpha = 0.001
        momentum = 1e9
        
        # Ring object initialization
        self.ring = Ring(n_turns, ring_length, alpha, momentum)
        
        # Beam parameters
        n_macroparticles = 100000
        intensity = 1e10
        
        my_beam = Beam(self.ring, n_macroparticles, intensity)
        my_beam.dt = np.load('dt_coordinates.npy')
        
        # First profile object initialization and tracking
        self.profile1 = profileModule.Profile(my_beam)
        self.profile1.track()
        
        # Second profile object initialization and tracking
        n_slices = 200
        CutOptions = profileModule.CutOptions(cut_left=0, cut_right=2*np.pi, 
                        n_slices = n_slices, cuts_unit='rad', omega_RF=2*np.pi/self.ring.t_rev[0])
        FitOptions = profileModule.FitOptions(fitMethod='fwhm', fitExtraOptions=None)
        FilterOptions = profileModule.FilterOptions(filterMethod=None, filterExtraOptions=None)
        OtherSlicesOptions = profileModule.OtherSlicesOptions(smooth=False, direct_slicing = True)
        self.profile2 = profileModule.Profile(my_beam, CutOptions = CutOptions,
                         FitOptions= FitOptions,
                         FilterOptions=FilterOptions, 
                         OtherSlicesOptions = OtherSlicesOptions)
        
        # Third profile object initialization and tracking
        n_slices = 150
        CutOptions = profileModule.CutOptions(cut_left=0, cut_right=self.ring.t_rev[0], 
                        n_slices = n_slices, cuts_unit='s')
        FitOptions = profileModule.FitOptions(fitMethod='rms', fitExtraOptions=None)
        FilterOptions = profileModule.FilterOptions(filterMethod=None, filterExtraOptions=None)
        OtherSlicesOptions = profileModule.OtherSlicesOptions(smooth=True, direct_slicing = False)
        self.profile3 = profileModule.Profile(my_beam, CutOptions = CutOptions,
                         FitOptions= FitOptions,
                         FilterOptions=FilterOptions, 
                         OtherSlicesOptions = OtherSlicesOptions)
        self.profile3.track()
        
        # Fourth profile object initialization and tracking
        n_slices = 2000
        CutOptions = profileModule.CutOptions(cut_left=0, cut_right=self.ring.t_rev[0], 
                        n_slices = n_slices, cuts_unit='s')
        FitOptions = profileModule.FitOptions(fitMethod='gaussian', fitExtraOptions=None)
        filter_option = {'pass_frequency':1e8, 
        'stop_frequency':1e9, 'gain_pass':1, 'gain_stop':2, 'transfer_function_plot':True}
        FilterOptions = profileModule.FilterOptions(filterMethod='chebishev', filterExtraOptions=filter_option)
        OtherSlicesOptions = profileModule.OtherSlicesOptions(smooth=False, direct_slicing = True)
        self.profile4 = profileModule.Profile(my_beam, CutOptions = CutOptions,
                         FitOptions= FitOptions,
                         FilterOptions=FilterOptions, 
                         OtherSlicesOptions = OtherSlicesOptions)
        
        print(self.profile2.bunchPosition-self.ring.t_rev[0]/2, sep=", ")
        
        print(self.profile3.bunchPosition-self.ring.t_rev[0]/2, sep=", ")
        
        print(self.profile4.bunchPosition-self.ring.t_rev[0]/2, sep=", ")
        
        print(self.profile2.bunchLength, sep=", ")
        
        print(self.profile3.bunchLength, sep=", ")
        
        print(self.profile4.bunchLength, sep=", ")
        
#         plt.plot(self.profile4.bin_centers, self.profile4.n_macroparticles)
#         plt.show()
    
    def test(self):
        
        self.assertAlmostEqual(self.ring.t_rev[0], 5.71753954209e-07, delta=1e-16,
                               msg='Ring: t_rev[0] not correct')
        
        self.assertSequenceEqual(self.profile1.n_macroparticles.tolist(), 
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 5.0, 7.0, 
             10.0, 9.0, 21.0, 20.0, 35.0, 40.0, 59.0, 70.0, 97.0, 146.0, 164.0, 
             183.0, 246.0, 324.0, 412.0, 422.0, 523.0, 637.0, 797.0, 909.0, 1074.0, 
             1237.0, 1470.0, 1641.0, 1916.0, 1991.0, 2180.0, 2464.0, 2601.0, 3010.0, 
             3122.0, 3233.0, 3525.0, 3521.0, 3657.0, 3650.0, 3782.0, 3845.0, 3770.0, 
             3763.0, 3655.0, 3403.0, 3433.0, 3066.0, 3038.0, 2781.0, 2657.0, 2274.0, 
             2184.0, 1934.0, 1724.0, 1452.0, 1296.0, 1111.0, 985.0, 803.0, 694.0, 
             546.0, 490.0, 429.0, 324.0, 239.0, 216.0, 147.0, 122.0, 87.0, 83.0, 
             62.0, 48.0, 28.0, 26.0, 14.0, 18.0, 12.0, 6.0, 7.0, 4.0, 2.0, 2.0, 
             1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0], 
             msg="Profile1 not correct")
        
        self.assertSequenceEqual(self.profile2.n_macroparticles.tolist(), 
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
        
        np.testing.assert_allclose(self.profile3.n_macroparticles.tolist(), 
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
                    0, 1e-8, err_msg="Profile3 not correct")


if __name__ == '__main__':

    unittest.main()

