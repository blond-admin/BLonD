# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Unit-tests for the self.SparseSlices  class.**

:Authors: **Markus Schwarz**
'''

# General imports
# -----------------
import unittest

import numpy as np

# BLonD imports
# --------------
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_slices import SparseSlices
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


class testProfileClass(unittest.TestCase):

    # Run before every test
    def setUp(self):
        """
        Slicing of the same Gaussian profile using four distinct settings to
        test different features.
        """

        np.random.seed(1984)

        intensity_pb = 1.0e11
        sigma = 0.2e-9  # Gauss sigma, [s]

        n_macroparticles_pb = int(1e4)
        n_bunches = 2

        # --- Ring and RF ----------------------------------------------
        intensity = n_bunches * intensity_pb     # total intensity SPS
        n_turns = 1
        # Ring parameters SPS
        circumference = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

        gamma_transition = 17.95142852  # Q20 Transition gamma
        momentum_compaction = 1. / gamma_transition**2  # Momentum compaction array

        ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
                    n_turns=n_turns)

        # RF parameters SPS
        harmonic_number = 4620  # harmonic number
        voltage = 3.5e6  # [V]
        phi_offsets = 0

        self.rf_station = RFStation(ring, harmonic_number, voltage, phi_offsets, n_rf=1)
        t_rf = self.rf_station.t_rf[0, 0]

        bunch_spacing = 5  # RF buckets

        n_macroparticles = n_bunches * n_macroparticles_pb
        self.beam = Beam(ring, n_macroparticles, intensity)

        for bunch in range(n_bunches):

            bunchBeam = Beam(ring, n_macroparticles_pb, intensity_pb)
            bigaussian(ring, self.rf_station, bunchBeam, sigma, reinsertion=True, seed=1984 + bunch)

            self.beam.dt[bunch * n_macroparticles_pb: (bunch + 1) * n_macroparticles_pb] \
                = bunchBeam.dt + bunch * bunch_spacing * t_rf
            self.beam.dE[bunch * n_macroparticles_pb: (bunch + 1) * n_macroparticles_pb] = bunchBeam.dE

        self.filling_pattern = np.zeros(bunch_spacing * (n_bunches - 1) + 1)
        self.filling_pattern[::bunch_spacing] = 1

        # uniform profile

        profile_margin = 0 * t_rf

        t_batch_begin = 0 * t_rf
        t_batch_end = (bunch_spacing * (n_bunches - 1) + 1) * t_rf

        self.n_slices_rf = 32  # number of slices per RF-bucket

        cut_left = t_batch_begin - profile_margin
        cut_right = t_batch_end + profile_margin

        # number of rf-buckets of the self.beam
        # + rf-buckets before the self.beam + rf-buckets after the self.beam
        n_slices = self.n_slices_rf * (bunch_spacing * (n_bunches - 1) + 1
                                       + int(np.round((t_batch_begin - cut_left) / t_rf))
                                       + int(np.round((cut_right - t_batch_end) / t_rf)))

        self.uniform_profile = Profile(self.beam,
                                       CutOptions=CutOptions(cut_left=cut_left, n_slices=n_slices,
                                                             cut_right=cut_right))
        self.uniform_profile.track()

    def test_WrongTrackingFunction(self):
        with self.assertRaises(RuntimeError):
            SparseSlices(self.rf_station, self.beam, self.n_slices_rf, self.filling_pattern,
                         tracker='something horribly wrong')

        nonuniform_profile = SparseSlices(self.rf_station, self.beam, self.n_slices_rf,
                                          self.filling_pattern)

        self.assertEqual(nonuniform_profile.bin_centers_array.shape, (2, self.n_slices_rf),
                         msg='Wrong shape of bin_centers_array!')

    def test_onebyone(self):
        rtol = 1e-6             # relative tolerance
        atol = 0                # absolute tolerance

        nonuniform_profile = SparseSlices(self.rf_station, self.beam, self.n_slices_rf,
                                          self.filling_pattern, tracker='onebyone',
                                          direct_slicing=True)

        for bunch in range(2):
            indexes = (self.uniform_profile.bin_centers > nonuniform_profile.cut_left_array[bunch])\
                * (self.uniform_profile.bin_centers < nonuniform_profile.cut_right_array[bunch])

            np.testing.assert_allclose(self.uniform_profile.bin_centers[indexes],
                                       nonuniform_profile.bin_centers_array[bunch],
                                       rtol=rtol, atol=atol,
                                       err_msg=f'Bins for bunch {bunch} do not agree '
                                       + 'for tracker="onebyone"')

            np.testing.assert_allclose(self.uniform_profile.n_macroparticles[indexes],
                                       nonuniform_profile.n_macroparticles_array[bunch],
                                       rtol=rtol, atol=atol,
                                       err_msg=f'Profiles for bunch {bunch} do not agree '
                                       + 'for tracker="onebyone"')

    def test_Ctracker(self):
        rtol = 1e-6             # relative tolerance
        atol = 0                # absolute tolerance

        nonuniform_profile = SparseSlices(self.rf_station, self.beam, self.n_slices_rf,
                                          self.filling_pattern, tracker='C',
                                          direct_slicing=True)

        for bunch in range(2):
            indexes = (self.uniform_profile.bin_centers > nonuniform_profile.cut_left_array[bunch])\
                * (self.uniform_profile.bin_centers < nonuniform_profile.cut_right_array[bunch])

            np.testing.assert_allclose(self.uniform_profile.bin_centers[indexes],
                                       nonuniform_profile.bin_centers_array[bunch],
                                       rtol=rtol, atol=atol,
                                       err_msg=f'Bins for bunch {bunch} do not agree '
                                       + 'for tracker="C"')

            np.testing.assert_allclose(self.uniform_profile.n_macroparticles[indexes],
                                       nonuniform_profile.n_macroparticles_array[bunch],
                                       rtol=rtol, atol=atol,
                                       err_msg=f'Profiles for bunch {bunch} do not agree '
                                       + 'for tracker="C"')


if __name__ == '__main__':

    unittest.main()
