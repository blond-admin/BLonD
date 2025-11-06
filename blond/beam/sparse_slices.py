# coding: utf-8
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute beam slicing for a sparse beam**
**Only valid for cases with constant revolution and RF frequencies**

:Authors: **Juan F. Esteban Mueller**
'''

from __future__ import division, print_function

import copy
from builtins import range

import numpy as np
from .beam import Beam
from ..beam.profile import CutOptions, Profile
from ..utils import bmath as bm


class SparseSlices:
    '''
    *This class instantiates a Profile object for each filled bucket according
    to the provided filling pattern. Each Profile object will be of the size of
    an RF bucket and will have the same number of slices.*
    '''

    def __init__(self, RFStation, Beam, number_of_slices_per_bucket, filling_pattern, tracker='C',
                 bucket_margin :int = 0,
                 direct_slicing=False):

        #: *Import (reference) Beam*
        self.Beam = Beam
        self.energy = Beam.energy
        #: *Import (reference) RFStation*
        self.RFParams = RFStation

        #: *Number of slices per bucket*
        self.bucket_margin = bucket_margin
        self.number_of_slices_per_bucket = number_of_slices_per_bucket * (1 + 2 * self.bucket_margin)

        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        self.filling_pattern = filling_pattern

        # Bunch index for each filled bucket (-1 if empty). Only for C++ track
        self.bunch_indexes = np.cumsum(filling_pattern) * filling_pattern - 1

        #: *Number of buckets to be sliced*
        self.n_filled_buckets = int(np.sum(filling_pattern))

        # Pre-processing the slicing edges
        self.set_cuts(bucket_margin = self.bucket_margin,
                      )

        # Initialize individual slicing objects
        self.profiles_list = []
        # Group n_macroparticles from all objects in a single array
        # (for C++ track).
        self.n_macroparticles_array = np.zeros((self.n_filled_buckets,
                                                self.number_of_slices_per_bucket))

        # Group bin_centers from all objects in a single array (for impedance)
        self.bin_centers_array = np.zeros((self.n_filled_buckets, self.number_of_slices_per_bucket))
        self.edges_array = np.zeros((self.n_filled_buckets, self.number_of_slices_per_bucket + 1))
        for i in range(self.n_filled_buckets):
            # Only valid for cut_edges='edges'

            self.profiles_list.append(Profile(self.Beam, CutOptions(cut_left=self.cut_left_array[i],
                                                               cut_right=self.cut_right_array[i],
                                                               n_slices=self.number_of_slices_per_bucket)))
            self.n_macroparticles_array[i,:] = self.profiles_list[i].n_macroparticles
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.edges_array[i, :] = self.profiles_list[i].edges

        #Total parameters
        self.n_macroparticles = np.concatenate(self.n_macroparticles_array, axis = 0)
        self.n_slices = int(self.number_of_slices_per_bucket * filling_pattern.sum())
        self.bin_centers = np.concatenate(self.bin_centers_array, axis = 0)
        self.bin_size = self.profiles_list[0].bin_size
        # Select the tracker
        if tracker == 'C':
            self.track = self._histogram_c
        elif tracker == 'onebyone':
            self.track = self._histogram_one_by_one
        else:
            # WrongCalcError
            raise RuntimeError(
                'Tracking method not recognized!')

        # Track at initialisation
        if direct_slicing:
            self.track()

    def set_cuts(self, bucket_margin: int = 0):
        '''
        *Method to set the self.cut_left_array and self.cut_right_array
        properties, with the limits being an RF period.
        This is done as a pre-processing.*

        Parameters
        ----------
        bucket_margin
            Extend the scope of the Profile objects to the neighbouring buckets.
        '''
        # RF period
        t_rf = self.RFParams.t_rf[0, self.RFParams.counter[0]]

        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        for i in range(self.n_filled_buckets):
            bucket_index = np.where(self.filling_pattern)[0][i]
            self.cut_left_array[i] = (bucket_index - bucket_margin) * t_rf
            self.cut_right_array[i] = ((bucket_index + 1 + bucket_margin) *
                                       t_rf)

    def _histogram_c(self):
        '''
        *Histrogram generated by calling an optimized C++ function that
        calculates all the profile at once.*
        '''
        bm.sparse_histogram(self.Beam.dt, self.n_macroparticles_array,
                            self.cut_left_array, self.cut_right_array,
                            self.bunch_indexes, self.number_of_slices_per_bucket)

    def _histogram_one_by_one(self):
        '''
        *Histogram generated by calling the tack() method of each Profile
        object*
        '''
        # for i in range(self.n_filled_buckets):
        #     self.profiles_list[i].track()
        for i in range(self.n_filled_buckets):
            self.profiles_list[i].track()
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.n_macroparticles_array[i, :] = self.profiles_list[
                i].n_macroparticles
        self.bin_centers = np.concatenate(self.bin_centers_array, axis = 0)
        self.n_macroparticles = np.concatenate(self.n_macroparticles_array,
                                               axis=0)
    def _set_additional_cuts(self,
                            updated_filling_pattern,
                            ):
        '''
        *Method to update set the self.cut_left_array and
        self.cut_right_array
        properties with additional filled buckets.*
        '''
        # RF period
        t_rf = self.RFParams.t_rf[0, self.RFParams.counter[0]]
        current_filling_pattern = self.filling_pattern

        filled_bunches_current = np.where(current_filling_pattern)[0]
        filled_bunches_new = np.where(updated_filling_pattern)[0]
        additional_filled_buckets = len(filled_bunches_new) - len(filled_bunches_current)

        mask_additional_bunch = copy.deepcopy(current_filling_pattern)
        for i in filled_bunches_new:
            if current_filling_pattern[i] == 0:
                mask_additional_bunch[i] = 1
            else:
                mask_additional_bunch[i] = 0
        #fixme: injected bunch might be between already considered bunches
        updated_cut_left = np.zeros(additional_filled_buckets)
        updated_cut_right = np.zeros(additional_filled_buckets)
        for i in range(additional_filled_buckets):
            bucket_index = np.where(mask_additional_bunch)[0][i]
            updated_cut_left[i] = bucket_index * t_rf
            updated_cut_right[i] = (bucket_index + 1) * t_rf
        self.cut_left_array = np.append(self.cut_left_array, updated_cut_left)
        self.cut_right_array = np.append(self.cut_right_array,
                                        updated_cut_right)
        self.filling_pattern = updated_filling_pattern
        return additional_filled_buckets

    def _update_profile_lists(self,
                            additional_filled_buckets: int,
                            ):
        '''
        *Method to update create individual profiles for the injected
        bunches*
        '''
        # Initialize individual slicing objects
        profiles_list_additional = []
        if (len(self.cut_right_array) != self.n_filled_buckets +
                additional_filled_buckets) or (len(self.cut_left_array) !=
                                               self.n_filled_buckets +
                additional_filled_buckets):
            raise ValueError('Cut arrays have not been updated.')
        if (len(np.where(self.filling_pattern)[0]) != self.n_filled_buckets +
                additional_filled_buckets):
            raise ValueError('Filling pattern has not been updated.')

        for i in range(additional_filled_buckets):
            # Only valid for cut_edges='edges'
            profiles_list_additional.append(
                Profile(self.Beam, CutOptions(cut_left=self.cut_left_array[
                    self.n_filled_buckets + i],
                                         cut_right=self.cut_right_array[self.n_filled_buckets + i],
                                         n_slices=self.number_of_slices_per_bucket)))
            self.n_macroparticles_array = np.insert(
                arr=self.n_macroparticles_array,
                obj = len(self.n_macroparticles_array),
                values = profiles_list_additional[i].n_macroparticles,
                axis=0,
            )
            self.bin_centers_array = np.insert(
                arr=self.bin_centers_array,
                obj = len(self.bin_centers_array),
                values = profiles_list_additional[i].bin_centers,
                axis=0,
            )
            self.edges_array = np.insert(
                arr=self.edges_array,
                obj = len(self.edges_array),
                values = profiles_list_additional[i].edges,
                axis=0,
            )
        self.profiles_list = np.append(self.profiles_list, profiles_list_additional)
        self.n_filled_buckets += additional_filled_buckets
        self._update_general_arrays()


    def _update_general_arrays(self):
        """
        Method to update the general arrays after a profile update.
        """
        # Total parameters
        if len(np.where(self.filling_pattern)[0]) != self.n_filled_buckets:
            raise ValueError(f'Filling pattern has length '
                             f'{len(np.where(self.filling_pattern)[0])}, number of '
                             f'declared filled buckets '
                             f'{self.n_filled_buckets}.')
        self.n_macroparticles = np.concatenate(self.n_macroparticles_array,
                                               axis=0)
        self.n_slices = int(self.number_of_slices_per_bucket * self.filling_pattern.sum())
        self.bunch_indexes = np.cumsum(
            self.filling_pattern) * self.filling_pattern - 1
        self.bin_centers = np.concatenate(self.bin_centers_array, axis=0)
        self.bin_size = self.profiles_list[0].bin_size

    def update_filling_pattern(self, beam: Beam,
                              updated_filling_pattern: list[int],
                              ):
        self.beam = beam
        additional_filled_buckets = self._set_additional_cuts(
            updated_filling_pattern = updated_filling_pattern)
        self._update_profile_lists(additional_filled_buckets = additional_filled_buckets)
