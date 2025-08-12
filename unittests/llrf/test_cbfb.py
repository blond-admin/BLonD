# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.coupled_bunch_feedback

:Authors: **Simon Albright**
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest

from blond.llrf import coupled_bunch_feedback as b_cbfb
from blond.beam import profile as b_prof
from blond.beam import beam as b_beam
from blond.input_parameters import ring as b_ring

__HERE = os.path.dirname(os.path.realpath(__file__))

class TestCoupledBunchAnalysis(unittest.TestCase):

    def setUp(self):

        gamma_t = 6.1
        mom_comp = 1 / (gamma_t**2)
        momentum = 15E9
        particle = b_beam.Proton()

        self.ring = b_ring.Ring(2*np.pi*100, mom_comp, momentum, particle)
        self.beam = b_beam.Beam(self.ring, int(1E6), 0)

        cut_opts = b_prof.CutOptions(cut_left=0,
                                     cut_right=self.ring.t_rev[0],
                                     n_slices=2**10)

        self.profile = b_prof.Profile(self.beam, cut_opts)


    def test_init(self):

        n_bunch = 4
        n_samp = 250
        cba = b_cbfb.CoupledBunchAnalysis(n_samp, self.profile, n_bunch)

        self.assertEqual(cba._n_samples, n_samp)

        self.assertTrue(cba._profile is self.profile)

        self.assertEqual(cba._max_n, n_bunch)
        self.assertEqual(cba._bunch_data.shape, (n_bunch, n_samp))

        nptest.assert_array_equal(cba.mode_amplitudes, np.zeros(n_bunch))
        nptest.assert_array_equal(cba.mode_frequencies, np.zeros(n_bunch))
        nptest.assert_array_equal(cba.mode_phases, np.zeros(n_bunch))

    def test_dip_measure(self):

        positions = np.linspace(0.5/21, 1-0.5/21, 21)*2E-6

        self.profile.bunchPosition = positions

        n_samp = 250
        cba = b_cbfb.CoupledBunchAnalysis(n_samp, self.profile, 21,
                                          mode = b_cbfb.CBFBModes.DIPOLAR)

        nptest.assert_array_equal(cba._bunch_data, np.zeros([21, 250]))

        cba._measure(self.profile, cba._bunch_data)

        nptest.assert_array_equal(cba._bunch_data[:,-1], positions)

        for _ in range(n_samp):
            cba._measure(self.profile, cba._bunch_data)

        for i in range(n_samp):
            nptest.assert_array_equal(cba._bunch_data[:,i], positions)


    def test_quad_measure(self):

        lengths = np.array([10E-9]*21)
        self.profile.bunchLength = lengths

        n_samp = 250
        cba = b_cbfb.CoupledBunchAnalysis(n_samp, self.profile, 21,
                                          mode = b_cbfb.CBFBModes.QUADRUPOLAR)

        nptest.assert_array_equal(cba._bunch_data, np.zeros([21, 250]))

        cba._measure(self.profile, cba._bunch_data)

        nptest.assert_array_equal(cba._bunch_data[:,-1], lengths)

        for _ in range(n_samp):
            cba._measure(self.profile, cba._bunch_data)

        for i in range(n_samp):
            nptest.assert_array_equal(cba._bunch_data[:,i], lengths)


    def test_fft(self):

        freqs = [2E-3, 4E-3, 6E-3, 8E-3] # 1/turn
        time = np.arange(2**16)

        cba = b_cbfb.CoupledBunchAnalysis(time.shape[0],
                                          self.profile, 4,
                                          mode = b_cbfb.CBFBModes.QUADRUPOLAR)

        for i, f in enumerate(freqs):
            oscill = np.sin(2*np.pi*f*time)
            cba._bunch_data[i] = oscill

        cba._motion_fft(4)

        for i, f in enumerate(freqs):
            max_pt = np.where(np.abs(cba._fft_matrix[i])
                              == np.max(np.abs(cba._fft_matrix[i])))[0][0]
            self.assertAlmostEqual(f, cba._fft_freqs[max_pt], places = 4)







class TestSupportFuncs(unittest.TestCase):

    def test_calc_amp_freq_phase(self):

        freq_arr = np.arange(1000)
        inds = np.where((freq_arr > 100)*(freq_arr<500))[0]

        data = np.zeros_like(freq_arr, dtype=complex)
        data[300] += 1 + 1j

        amp, freq, phase = b_cbfb.calc_amp_freq_phase(freq_arr, data, inds,
                                                     0, 0, 0)

        self.assertAlmostEqual(amp, np.sqrt(2))
        self.assertEqual(freq, freq_arr[300])
        self.assertAlmostEqual(phase, np.pi/4)

    def test_freq_phase(self):

        freq_arr = np.arange(1000)
        inds = np.where((freq_arr > 100)*(freq_arr<500))[0]

        data = np.ones_like(freq_arr, dtype=complex) + 1j

        freq, phase = b_cbfb.freq_phase(freq_arr, data, inds, 0, 0, 0, 0)

        self.assertEqual(freq, freq_arr[inds[0]])
        self.assertAlmostEqual(phase, np.pi/4)

    def test_linear_correction(self):

        time = np.linspace(0, 3, 100000)
        data = 0.1*np.sin(5*2*np.pi*time)

        correction = b_cbfb._linear_correction(time, [data, data])
        nptest.assert_array_almost_equal(correction, np.zeros_like(correction))

        tilt = np.linspace(1, 1.5, len(time))
        correction = b_cbfb._linear_correction(time, [data+tilt, data+tilt])
        nptest.assert_array_almost_equal(correction, tilt, decimal=2)


    def test_queue(self):

        queue = np.zeros([5, 10])

        for i in range(10):
            measure = np.arange(5) + i
            b_cbfb._populate_queue(measure, queue)
            nptest.assert_array_equal(queue[:,-1], measure)

        nptest.assert_array_equal(queue[:,0], np.arange(5))
        b_cbfb._populate_queue(measure, queue)
        nptest.assert_array_equal(queue[:,-1], queue[:,-2])
        nptest.assert_array_equal(queue[:,0], np.arange(5)+1)




if __name__ == '__main__':
    unittest.main()
