# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.filters

:Authors: **Helga Timko**
"""

import unittest
import numpy as np
from beam.beam import Proton
from input_parameters.ring import Ring
from input_parameters.rf_parameters import RFStation
from beam.beam import Beam
from beam.distributions import bigaussian
from beam.profile import Profile, CutOptions
from llrf.signal_processing import rf_beam_current, low_pass_filter


class TestBeamCurrent(unittest.TestCase):
    
    def test(self):
        
        # Set up SPS conditions
        ring = Ring(2*np.pi*1100.009, 1/18**2, 25.92e9,
                    Proton(), 1000)
        RF = RFStation(ring, 4620, 4.5e6, 0)
        beam = Beam(ring, 1e5, 1e11)
        bigaussian(ring, RF, beam, 3.2e-9/4, seed = 1234,
                   reinsertion = True) 
        profile = Profile(beam, CutOptions(cut_left=-1.e-9, cut_right=6.e-9, 
                                           n_slices=100))
        profile.track()
        self.assertEqual(len(beam.dt), np.sum(profile.n_macroparticles), "In" +
            " TestBeamCurrent: particle number mismatch in Beam vs Profile")
        
        # RF current calculation with low-pass filter
        rf_current = rf_beam_current(profile, 2*np.pi*200.222e6, ring.t_rev[0])
        Iref_real = np.array([ -9.4646042539e-12,  -7.9596801534e-10,  
            -2.6993572787e-10,
            2.3790828610e-09,   6.4007063190e-09,   9.5444302650e-09,
            9.6957462918e-09,   6.9944771120e-09,   5.0040512366e-09,
            8.2427583408e-09,   1.6487066238e-08,   2.2178930587e-08,
            1.6497620890e-08,   1.9878201568e-09,  -2.4862807497e-09,
            2.0862096916e-08,   6.6115473293e-08,   1.1218114710e-07,
             1.5428441607e-07,   2.1264254596e-07,   3.1213935713e-07,
             4.6339212948e-07,   6.5039440158e-07,   8.2602190806e-07,
             9.4532001396e-07,   1.0161170159e-06,   1.0795840334e-06,
             1.1306004256e-06,   1.1081141333e-06,   9.7040873320e-07,
             7.1863437325e-07,   3.3833950889e-07,  -2.2273124358e-07,
            -1.0035204008e-06,  -1.9962696992e-06,  -3.1751183137e-06,
            -4.5326227784e-06,  -6.0940850385e-06,  -7.9138578879e-06,
            -9.9867317826e-06,  -1.2114906338e-05,  -1.4055138779e-05,
            -1.5925650405e-05,  -1.8096693885e-05,  -2.0418813156e-05,
            -2.2142865862e-05,  -2.3038234657e-05,  -2.3822481250e-05,
            -2.4891969829e-05,  -2.5543384520e-05,  -2.5196086909e-05,
            -2.4415522211e-05,  -2.3869116251e-05,  -2.3182951665e-05,
            -2.1723128723e-05,  -1.9724625363e-05,  -1.7805112266e-05,
            -1.5981218737e-05,  -1.3906226012e-05,  -1.1635865568e-05,
            -9.5381189596e-06,  -7.7236624815e-06,  -6.0416822483e-06,
            -4.4575806261e-06,  -3.0779237834e-06,  -1.9274519396e-06,
            -9.5699993457e-07,  -1.7840768971e-07,   3.7780452612e-07,
             7.5625231388e-07,   1.0158886027e-06,   1.1538975409e-06,
             1.1677937652e-06,   1.1105424636e-06,   1.0216131672e-06,
             8.8605026541e-07,   7.0783694846e-07,   5.4147914020e-07,
             4.1956457226e-07,   3.2130062098e-07,   2.2762751268e-07,
             1.4923020411e-07,   9.5683463322e-08,   5.8942895620e-08,
             3.0515695233e-08,   1.2444834300e-08,   8.9413517889e-09,
             1.6154761941e-08,   2.3261993674e-08,   2.3057968490e-08,
             1.8354179928e-08,   1.4938991667e-08,   1.2506841004e-08,
             8.1230022648e-09,   3.7428821201e-09,   2.8368110506e-09,
             3.6536247240e-09,   2.8429736524e-09,   1.6640835314e-09,
             2.3960087967e-09])
        I_real = np.around(rf_current.real, 9) # round
        Iref_real = np.around(Iref_real, 9) 
        self.assertSequenceEqual(I_real.tolist(), Iref_real.tolist(),
            msg="In TestBeamCurrent, mismatch in real part of RF current")
        Iref_imag = np.array([ -1.3134886055e-11,   1.0898262206e-09,   
            3.9806900984e-10,
            -3.0007980073e-09,  -7.4404909183e-09,  -9.5619658077e-09,
            -7.9029982105e-09,  -4.5153699012e-09,  -2.8337010673e-09,
            -4.0605999910e-09,  -5.7035811935e-09,  -4.9421561822e-09,
            -2.6226262365e-09,  -1.0904425703e-09,   1.5886725829e-10,
             3.6061564044e-09,   1.2213233410e-08,   3.0717134774e-08,
             6.2263860975e-08,   1.0789908935e-07,   1.8547368321e-07,
             3.3758410599e-07,   5.8319210090e-07,   8.7586115583e-07,
             1.1744525681e-06,   1.5330067491e-06,   2.0257108185e-06,
             2.6290348930e-06,   3.3065045701e-06,   4.1218136471e-06,
             5.1059358251e-06,   6.1421308306e-06,   7.1521192647e-06,
             8.2164613957e-06,   9.3474086978e-06,   1.0368027059e-05,
             1.1176114701e-05,   1.1892303251e-05,   1.2600522466e-05,
             1.3142991032e-05,   1.3286611961e-05,   1.2972067098e-05,
             1.2344251145e-05,   1.1561930031e-05,   1.0577353622e-05,
             9.1838382917e-06,   7.3302333455e-06,   5.2367297732e-06,
             3.1309520147e-06,   1.0396785645e-06,  -1.1104442284e-06,
            -3.3300486963e-06,  -5.5129705406e-06,  -7.4742790081e-06,
            -9.1003715719e-06,  -1.0458342224e-05,  -1.1632423668e-05,
            -1.2513736332e-05,  -1.2942309414e-05,  -1.2975831165e-05,
            -1.2799952495e-05,  -1.2469945465e-05,  -1.1941176358e-05,
            -1.1222986380e-05,  -1.0349594257e-05,  -9.3491445482e-06,
            -8.2956327726e-06,  -7.2394219079e-06,  -6.1539590898e-06,
            -5.0802321519e-06,  -4.1512021086e-06,  -3.3868884793e-06,
            -2.6850344653e-06,  -2.0327038471e-06,  -1.5048854341e-06,
            -1.0965986189e-06,  -7.4914749272e-07,  -4.7128817088e-07,
            -2.9595396024e-07,  -1.9387567373e-07,  -1.1597751838e-07,
            -5.5766761837e-08,  -2.3991059778e-08,  -1.1910924971e-08,
            -4.7797889603e-09,   9.0715301612e-11,   1.5744084129e-09,
             2.8217939283e-09,   5.5919203984e-09,   7.7259433940e-09,
             8.5033504655e-09,   9.1509256107e-09,   8.6746085156e-09,
             5.8909590412e-09,   3.5957212556e-09,   4.3347189168e-09,
             5.3331969589e-09,   3.9322184713e-09,   3.3616434953e-09,
             6.5154351819e-09])
        I_imag = np.around(rf_current.imag, 9) # round
        Iref_imag = np.around(Iref_imag, 9)
        self.assertSequenceEqual(I_imag.tolist(), Iref_imag.tolist(),
            msg="In TestBeamCurrent, mismatch in imaginary part of RF current")


       
if __name__ == '__main__':

    unittest.main()



