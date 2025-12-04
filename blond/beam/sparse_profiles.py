# coding: utf-8
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to compute beam slicing for a sparse beam**
**Only valid for cases with constant revolution and RF frequencies**

:Authors: **Juan F. Esteban Mueller, Lina Valle (ed. 2025)**
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .profile import CutOptions, Profile
from ..utils import bmath as bm

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import NDArray as NumpyArray

    from .beam import Beam
    from ..input_parameters.rf_parameters import RFStation

    TrackerTypes = Literal["C", "onebyone"]


class _SparseProfileBaseClass:
    "Base class for sparse profiles object."

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        _filling_pattern: NumpyArray,
        _profile_length_in_buckets: int,
        tracker: TrackerTypes = "C",
        initialisation_slicing: bool = False,
    ):
        """
        Common initialization of Sparse objects.

        Parameters
        ----------
        rf_station
            RFStation object
        beam
            Beam object
        number_of_slices_per_profile
            Number of slices per profile
        _filling_pattern
            Filling pattern / Bunch list/ Batch list of the synchrotron
        _profile_length_in_buckets
            Profile lengths in number of RF buckets. Should be greater than 1.
        tracker
            Choice of tracker. Can be "C" or "onebyone". Default is "C".
        initialisation_slicing
            Enables tracking at initialisation. FALSE by default.

        """
        if (len(_filling_pattern) > rf_station.harmonic).any():
            raise ValueError(
                f"The length of filling_pattern exceeds "
                f"the number of RF buckets"
            )

        if (len(_filling_pattern) != rf_station.harmonic).any():
            warnings.warn(
                f"The filling pattern is shorter than the "
                f"total number of RF buckets.",
                UserWarning,
                stacklevel=2,
            )
        if not isinstance(_profile_length_in_buckets, int):
            raise TypeError(
                "The profile length should be an integer number of RF buckets."
            )
        self.beam = beam
        self.rf_station = rf_station

        self.number_of_slices_per_profile = number_of_slices_per_profile
        self._filling_pattern = _filling_pattern
        self._number_of_indices = int(np.sum(_filling_pattern))
        self._profile_length_in_buckets = _profile_length_in_buckets

        # Index of each batch (-1 if empty). Only for C++ track
        self._bucket_indices = (
            np.cumsum(_filling_pattern) * _filling_pattern - 1
        )

        self.profiles_list = None
        self.cut_left_array = None
        self.cut_right_array = None

        self.tracker = tracker
        self.initialisation_slicing = initialisation_slicing

        # Pre-processing the slicing edges
        self._set_cuts(length_in_buckets=_profile_length_in_buckets)
        self._generate_profile_list()

        # Track at initialisation
        if self.initialisation_slicing:
            self.track()

    @property
    def n_macroparticles_array(self):
        return np.array(
            [
                self.profiles_list[i].n_macroparticles
                for i in range(self._number_of_indices)
            ]
        )

    @property
    def bin_centers_array(self):
        return np.array(
            [
                self.profiles_list[i].bin_centers
                for i in range(self._number_of_indices)
            ]
        )

    @property
    def edges_array(self):
        return np.array(
            [
                self.profiles_list[i].edges
                for i in range(self._number_of_indices)
            ]
        )

    @property
    def n_macroparticles(self):
        return self.n_macroparticles_array.flatten()

    @property
    def bin_centers(self):
        return self.bin_centers_array.flatten()

    @property
    def bin_size(self):
        # by construction, all the profiles have the same bin sizes.
        return self.profiles_list[0].bin_size

    @property
    def n_slices(self):
        return self.number_of_slices_per_profile * len(self.profiles_list)

    @property
    def edges(self):
        return self.edges_array.flatten()

    def _set_cuts(self, length_in_buckets: int):
        """
        Internal method to set the self.cut_left_array and self.cut_right_array
        properties.

        The left cut starts at the bucket index considered and the right cut
        is defined by its distance, in number of RF buckets (input
        length_in_buckets), from the bucket index considered.

        This is done as a pre-processing.

        Parameters
        ----------
        length_in_buckets
            Total length, in buckets, of the profile. Should be greater than 1.
        """
        if length_in_buckets < 1:
            raise ValueError(
                "The length of the profile scope should be a "
                "non-zero integer number of RF buckets."
            )
        # RF period
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]
        bucket_indices = np.where(self._filling_pattern != 0)[0]

        self.cut_left_array = bucket_indices * t_rf
        self.cut_right_array = (bucket_indices + length_in_buckets) * t_rf

    def _generate_profile_list(self):
        """
        Internal method which creates a Profile object for each filled
        bucket or list of buckets.

        The created profiles are stored in self.profile_list. The method
        initialises the general arrays and matrices (n_macroparticles,
        bin_size, bin_centers).
        """

        self.profiles_list = [
            Profile(
                self.beam,
                CutOptions(
                    cut_left=self.cut_left_array[i],
                    cut_right=self.cut_right_array[i],
                    n_slices=self.number_of_slices_per_profile,
                ),
            )
            for i in range(self._number_of_indices)
        ]

    def track(self):
        """
        Tracking method of the profile, depending on the tracker choice.
        """
        if self.tracker == "C":
            self._histogram_c()
        elif self.tracker == "onebyone":
            self._histogram_one_by_one()
        else:
            # WrongCalcError
            raise RuntimeError("Tracking method not recognized!")

    def _histogram_c(self):
        """
        Histogram generated by calling an optimized C++ function that
        calculates all the profile at once.
        """
        bm.sparse_histogram(
            self.beam.dt,
            self.n_macroparticles_array,
            self.cut_left_array,
            self.cut_right_array,
            self._bucket_indices,
            self.number_of_slices_per_profile,
        )

    def _histogram_one_by_one(self):
        """
        Histogram generated by calling the tack() method of each Profile
        object
        """

        for i in range(len(self.profiles_list)):
            self.profiles_list[i].track()

    def _set_additional_cuts(
        self,
        _updated_filling_pattern: NumpyArray,
    ):
        """
        Internal method to update the cut array properties of the Sparse
        object with new cut_left | cut_right options around the additional
        indices.

        The left cut starts at the bucket index considered and the right cut
        is defined by its distance, in number of RF buckets (
        self._profile_length_in_buckets), from the bucket index considered.
        This is done as a pre-processing.

        Returns
        ---------
        _additional_indices
            Number of additional indices to consider for an update of the
            profile list.
        """
        if len(self._filling_pattern) != len(_updated_filling_pattern):
            raise ValueError(
                f"The length of the updated filling pattern does not match "
                f"the previously sorted filling pattern lengths: "
                f"{len(_updated_filling_pattern)} != "
                f"{len(self._filling_pattern)}"
            )
        # RF period
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]

        filled_bunches_current = np.where(self._filling_pattern != 0)[0]
        filled_bunches_new = np.where(_updated_filling_pattern != 0)[0]
        _additional_indices = len(filled_bunches_new) - len(
            filled_bunches_current
        )

        mask_additional_bunch = self._filling_pattern.copy()

        for i in filled_bunches_new:
            if self._filling_pattern[i] == 0:
                mask_additional_bunch[i] = 1
            else:
                mask_additional_bunch[i] = 0
        # fixme: injected bunch might be between already considered bunches

        masked_indices = np.where(mask_additional_bunch != 0)[0]

        if len(masked_indices) != _additional_indices:
            raise ValueError(
                "The mask does not reflect the additional indices"
            )

        updated_cut_left = masked_indices * t_rf
        updated_cut_right = (
            masked_indices + self._profile_length_in_buckets
        ) * t_rf
        self.cut_left_array = np.append(self.cut_left_array, updated_cut_left)
        self.cut_right_array = np.append(
            self.cut_right_array, updated_cut_right
        )
        self._filling_pattern = _updated_filling_pattern
        return _additional_indices

    def _update_profile_lists(
        self,
        _additional_indices: int,
    ):
        """
        Internal method to update the profile list with new profile objects
        for to track the additional indices.

        Ths method updates the general arrays and matrices accordingly (
        n_macroparticles, bin_centers).

        Parameters
        ----------
        _additional_indices
            Number of additional indices to consider. Provided by the
            internal method self._set_additional_cuts
        """

        # Initialize individual slicing objects
        profiles_list_additional = []
        _total_number_of_indices = (
            self._number_of_indices + _additional_indices
        )
        if (len(self.cut_right_array) != _total_number_of_indices) or (
            len(self.cut_left_array) != _total_number_of_indices
        ):
            raise ValueError("Cut arrays have not been updated.")
        if (
            len(np.where(self._filling_pattern != 0)[0])
            != _total_number_of_indices
        ):
            raise ValueError("Filling pattern has not been updated.")

        profiles_list_additional = [
            Profile(
                self.beam,
                CutOptions(
                    cut_left=self.cut_left_array[self._number_of_indices + i],
                    cut_right=self.cut_right_array[
                        self._number_of_indices + i
                    ],
                    n_slices=self.number_of_slices_per_profile,
                ),
            )
            for i in range(_additional_indices)
        ]

        self.profiles_list += profiles_list_additional
        self._number_of_indices += _additional_indices
        self._bucket_indices = (
            np.cumsum(self._filling_pattern) * self._filling_pattern - 1
        )


class SparseBucket(_SparseProfileBaseClass):
    """
     This class instantiates a Profile object for each filled bucket according
     to the provided filling pattern or bunch list.
     Each Profile object will be of the size of an RF bucket and will have the same number of slices.

     Parameters
     ----------
     rf_station
         RFStation object
     beam
         Beam object
     number_of_slices_per_profile
         Number of slices per profile
     bunch_list
         Bunch list (or filling pattern) of the synchrotron
     tracker
         Choice of tracker. Can be "C" or "onebyone". Default is "C".
     initialisation_slicing
         Track at initialisation. FALSE by default.


    Example
     --------
     >>> import numpy as np
     >>> from blond.beam.sparse_profiles import SparseBucket
     >>> from blond.beam.beam import Beam
     >>> from blond.input_parameters.rf_parameters import RFStation
     >>> self.ring = Ring(n_turns = 1, ring_length = 100,
     >>> alpha = 0.00001, momentum = 1e9)
     >>> self.beam = Beam(self.ring, 10000, 1e10)
     >>> self.rf_params = RFStation(ring=self.ring, n_rf=1, harmonic=[4620],
     ...                  voltage=[7e6], phi_rf_d=[0.])
     >>> bunch_spacing = 10
     >>> bunch_list = np.zeros(self.rf_params.harmonic[0])
     >>> bunch_list[::bunch_spacing] = 1
     >>> sparse_profile = SparseBucket(rf_station=self.rf_params,
     >>>                                 beam = self.beam,
     >>>number_of_slices_per_profile = 1e4,
     >>>bunch_list = bunch_list)
     >>>
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        bunch_list: NumpyArray,
        tracker: TrackerTypes = "C",
        initialisation_slicing: bool = False,
    ):
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        super().__init__(
            rf_station=rf_station,
            beam=beam,
            number_of_slices_per_profile=number_of_slices_per_profile,
            _filling_pattern=bunch_list,
            _profile_length_in_buckets=1,
            tracker=tracker,
            initialisation_slicing=initialisation_slicing,
        )

    @property
    def bunch_list(self):
        return self._filling_pattern

    @property
    def total_number_of_filled_buckets(self):
        return self._number_of_indices

    @property
    def bunch_indices(self):
        return self._bucket_indices

    def update_bunch_list(
        self,
        updated_bunch_list: list[int],
    ):
        """
        Function to update the SparseBucket object to match the new bunch
        list in the case of newly injected bunches.

        The method creates additional profiles to follow the newly injected
        bunches, and updated the internal arrays and numbering accordingly.

        Parameters
        ----------
        updated_bunch_list
            Updated bunch list. Must be the same length as the stored bunch
            list.
        """
        additional_filled_buckets = self._set_additional_cuts(
            _updated_filling_pattern=updated_bunch_list
        )
        self._update_profile_lists(
            _additional_indices=additional_filled_buckets
        )


class SparseBatch(_SparseProfileBaseClass):
    """
    This class instantiates a Profile object for each batch according
    to the provided batch list.
    Each Profile object will be of the size of
    a batch and will have the same number of slices.

    Parameters
    ----------
    rf_station
        RFStation object
    beam
        Beam object
    number_of_slices_per_profile
        Number of slices per profile
    batch_list
        Batch list (or filling pattern) of the synchrotron
    batch_length
        Batch length in number of RF buckets.
    tracker
        Choice of tracker. Can be "C" or "onebyone". Default is "C".
    initialisation_slicing
        Track at initialisation. FALSE by default.
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        batch_list: NumpyArray,
        batch_length: int = 1,
        tracker: TrackerTypes = "C",
        initialisation_slicing: bool = False,
    ):
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        super().__init__(
            rf_station=rf_station,
            beam=beam,
            number_of_slices_per_profile=number_of_slices_per_profile,
            _filling_pattern=batch_list,
            _profile_length_in_buckets=batch_length,
            tracker=tracker,
            initialisation_slicing=initialisation_slicing,
        )

    @property
    def batch_list(self):
        return self._filling_pattern

    @property
    def number_of_slices_per_bucket(self):
        return self.number_of_slices_per_profile / self.batch_length

    @property
    def total_number_of_batches(self):
        return self._number_of_indices

    @property
    def total_number_of_sliced_buckets(self):
        return int(self._profile_length_in_buckets * self._number_of_indices)

    @property
    def batch_length(self):
        return self._profile_length_in_buckets

    @property
    def batch_indices(self):
        return self._bucket_indices

    def update_batch_list(
        self,
        updated_batch_list: list[int],
    ):
        """
        Function to update the SparseBatch object to match the new batch
        list in the case of newly injected batches.

        The method creates additional profiles to follow the newly injected
        batches, and updated the internal arrays and numbering accordingly.

        Parameters
        ----------
        updated_batch_list
            Updated batch list. Must be the same length as the stored batch
            list.
        """
        additional_batches = self._set_additional_cuts(
            _updated_filling_pattern=updated_batch_list
        )
        self._update_profile_lists(_additional_indices=additional_batches)


def SparseSlices(
    rf_station: RFStation,
    beam: Beam,
    n_slices_bucket: int,
    filling_pattern: NumpyArray,
    tracker: TrackerTypes = "C",
    direct_slicing: bool = False,
):
    """
    Deprecated: please use SparseBucket

    This class instantiates a SparseBucket object.

    Parameters
    ----------
    rf_station
        RFStation object
    beam
        Beam object
    n_slices_bucket
        Number of slices per profile
    filling_pattern
        Bunch list (or filling pattern) of the synchrotron
    tracker
        Choice of tracker. Can be "C" or "onebyone". Default is "C".
    direct_slicing
        Track at initialisation. FALSE by default.
    """
    from warnings import warn

    warn(
        "SparseSlices is deprecated, use SparseBucket",
        DeprecationWarning,
        stacklevel=2,
    )

    return SparseBucket(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=n_slices_bucket,
        bunch_list=filling_pattern,
        tracker=tracker,
        initialisation_slicing=direct_slicing,
    )
