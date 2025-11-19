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

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .profile import CutOptions, Profile
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Literal
    from typing import Optional as LateInit

    from numpy.typing import NDArray as NumpyArray

    from .beam import Beam
    from ..input_parameters.rf_parameters import RFStation

    TrackerTypes = Literal["C", "onebyone"]


class SparseSlices:
    """
    This class instantiates a Profile object for each filled bucket according
    to the provided filling pattern.

    By default, each Profile object has the size of an RF bucket, and the
    same number of slices (number_of_slices_per_bucket).

    Parameters
    ----------
    rf_station
        RFStation object
    beam
        Beam object
    number_of_slices_per_bucket
        Number of slices per bucket
    filling_pattern
        Filling pattern of the synchrotron
    tracker
        Choice of tracker. Can be "C" or "onebyone".
    direct_slicing
        Track at initialisation. FALSE by default.
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_bucket: int,
        filling_pattern: NumpyArray,
        tracker: TrackerTypes = "C",
        direct_slicing: bool = False,
    ):
        if (len(filling_pattern) > rf_station.harmonic).any():
            raise ValueError(
                f"The length of filling_pattern does not match exceeds "
                f"the number of RF buckets"
            )

        if (len(filling_pattern) != rf_station.harmonic).any():
            warnings.warn(
                f"The filling pattern is shorter than the "
                f"total number of RF buckets.",
                UserWarning,
                stacklevel=2,
            )
            #: *Import (reference) Beam*
        self.beam = beam
        self.energy = self.beam.energy
        #: *Import (reference) RFStation*
        self.rf_station = rf_station

        #: *Number of slices per bucket*
        self.number_of_slices_per_bucket = number_of_slices_per_bucket
        self.number_of_slices_per_profile = self.number_of_slices_per_bucket
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        self.filling_pattern = filling_pattern

        # Bunch index for each filled bucket (-1 if empty). Only for C++ track
        self.bunch_indexes = (
            np.cumsum(self.filling_pattern) * self.filling_pattern - 1
        )

        #: *Number of filled buckets in the filling pattern*
        self.n_filled_buckets = int(np.sum(self.filling_pattern))

        # Pre-processing the slicing edges
        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        self.set_cuts()

        # Initialize individual slicing objects
        self.profiles_list = []
        # Group n_macroparticles from all objects in a single array
        # (for C++ track).
        self.n_macroparticles_array = np.zeros(
            (self.n_filled_buckets, self.number_of_slices_per_profile)
        )
        # Group bin_centers from all objects in a single array (for impedance)
        self.bin_centers_array = np.zeros(
            (self.n_filled_buckets, self.number_of_slices_per_profile)
        )
        self.edges_array = np.zeros(
            (self.n_filled_buckets, self.number_of_slices_per_profile + 1)
        )
        for i in range(self.n_filled_buckets):
            # Only valid for cut_edges='edges'

            self.profiles_list.append(
                Profile(
                    beam,
                    CutOptions(
                        cut_left=float(self.cut_left_array[i]),
                        cut_right=float(self.cut_right_array[i]),
                        n_slices=self.number_of_slices_per_profile,
                    ),
                )
            )

            self.n_macroparticles_array[i, :] = self.profiles_list[
                i
            ].n_macroparticles
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.edges_array[i, :] = self.profiles_list[i].edges

        # Total parameters to match the standard Profile object
        self.n_macroparticles = np.concatenate(
            self.n_macroparticles_array, axis=0
        )
        self.n_slices = int(
            self.number_of_slices_per_bucket * filling_pattern.sum()
        )
        self.bin_centers = np.concatenate(self.bin_centers_array, axis=0)
        self.bin_size = self.profiles_list[0].bin_size

        # Select the tracker
        if tracker == "C":
            self.track = self._histogram_c
        elif tracker == "onebyone":
            self.track = self._histogram_one_by_one
        else:
            raise NameError(f"{tracker=}")

        # Track at initialisation
        if direct_slicing:
            self.track()

    @property
    def Beam(self):
        from warnings import warn

        warn("Beam is deprecated, use beam", DeprecationWarning, stacklevel=2)
        return self.beam

    @Beam.setter
    def Beam(self, val):
        from warnings import warn

        warn("Beam is deprecated, use beam", DeprecationWarning, stacklevel=2)
        self.beam = val

    @property
    def RFParams(self):
        from warnings import warn

        warn(
            "RFParams is deprecated, use rf_station",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rf_station

    @RFParams.setter
    def RFParams(self, val):
        from warnings import warn

        warn(
            "RFParams is deprecated, use rf_station",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rf_station = val

    def set_cuts(self):
        """
        *Method to set the self.cut_left_array and self.cut_right_array
        properties, with the limits being an RF period.
        This is done as a pre-processing.*
        """
        # RF period
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]

        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        bucket_indexes = np.where(self.filling_pattern != 0)[0]
        for i in range(self.n_filled_buckets):
            bucket_index = bucket_indexes[i]
            self.cut_left_array[i] = bucket_index * t_rf
            self.cut_right_array[i] = (bucket_index + 1) * t_rf

    def _histogram_c(self):
        """
        *Histogram generated by calling an optimized C++ function that
        calculates all the profile at once.*
        """
        # todo could be any backend, not only C
        bm.sparse_histogram(
            self.beam.dt,
            self.n_macroparticles_array,
            self.cut_left_array,
            self.cut_right_array,
            self.bunch_indexes,
            self.number_of_slices_per_bucket,
        )

    def _histogram_one_by_one(self):
        """
        *Histogram generated by calling the tack() method of each Profile
        object*
        """

        for i in range(self.n_filled_buckets):
            self.profiles_list[i].track()

    def _set_additional_cuts(
        self,
        updated_filling_pattern: NumpyArray,
    ):
        """
        *Method to update total cut array properties with the new cut
        options for the additional filled buckets in the updated filling
        pattern, with the limits being
        an RF period.*
        """
        if len(self.filling_pattern) != len(updated_filling_pattern):
            raise ValueError(
                f"The length of the updated filling pattern does not match "
                f"the previously sorted filling pattern lengths: "
                f"{len(updated_filling_pattern)} != "
                f"{len(self.filling_pattern)}"
            )
        # RF period
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]

        filled_bunches_current = np.where(self.filling_pattern != 0)[0]
        filled_bunches_new = np.where(updated_filling_pattern != 0)[0]
        additional_filled_buckets = len(filled_bunches_new) - len(
            filled_bunches_current
        )

        mask_additional_bunch = self.filling_pattern.copy()

        for i in filled_bunches_new:
            if self.filling_pattern[i] == 0:
                mask_additional_bunch[i] = 1
            else:
                mask_additional_bunch[i] = 0
        # fixme: injected bunch might be between already considered bunches
        updated_cut_left = np.zeros(additional_filled_buckets)
        updated_cut_right = np.zeros(additional_filled_buckets)
        masked_indexes = np.where(mask_additional_bunch != 0)[0]
        for i in range(additional_filled_buckets):
            bucket_index = masked_indexes[i]
            updated_cut_left[i] = bucket_index * t_rf
            updated_cut_right[i] = (bucket_index + 1) * t_rf
        self.cut_left_array = np.append(self.cut_left_array, updated_cut_left)
        self.cut_right_array = np.append(
            self.cut_right_array, updated_cut_right
        )
        self.filling_pattern = updated_filling_pattern
        return additional_filled_buckets

    def _update_profile_lists(
        self,
        additional_filled_buckets: int,
    ):
        """
        *Method to update the total profile lists by creating individual
        profiles for the newly injected bunches*
        """

        # Initialize individual slicing objects
        profiles_list_additional = []
        total_bunches = self.n_filled_buckets + additional_filled_buckets
        if (len(self.cut_right_array) != total_bunches) or (
            len(self.cut_left_array) != total_bunches
        ):
            raise ValueError("Cut arrays have not been updated.")
        if len(np.where(self.filling_pattern != 0)[0]) != total_bunches:
            raise ValueError("Filling pattern has not been updated.")

        for i in range(additional_filled_buckets):
            # Only valid for cut_edges='edges'
            profiles_list_additional.append(
                Profile(
                    self.Beam,
                    CutOptions(
                        cut_left=self.cut_left_array[
                            self.n_filled_buckets + i
                        ],
                        cut_right=self.cut_right_array[
                            self.n_filled_buckets + i
                        ],
                        n_slices=self.number_of_slices_per_profile,
                    ),
                )
            )
            self.n_macroparticles_array = np.insert(
                arr=self.n_macroparticles_array,
                obj=len(self.n_macroparticles_array),
                values=profiles_list_additional[i].n_macroparticles,
                axis=0,
            )
            self.bin_centers_array = np.insert(
                arr=self.bin_centers_array,
                obj=len(self.bin_centers_array),
                values=profiles_list_additional[i].bin_centers,
                axis=0,
            )
            self.edges_array = np.insert(
                arr=self.edges_array,
                obj=len(self.edges_array),
                values=profiles_list_additional[i].edges,
                axis=0,
            )
        self.profiles_list += profiles_list_additional
        self.n_filled_buckets += additional_filled_buckets
        self._update_general_arrays()

    def _update_general_arrays(self):
        """
        Method to update the general arrays after a profile update.
        """
        # Total parameters
        if (
            len(np.where(self.filling_pattern != 0)[0])
            != self.n_filled_buckets
        ):
            raise ValueError(
                f"Filling pattern has length "
                f"{len(np.where(self.filling_pattern)[0])}, number of "
                f"declared filled buckets "
                f"{self.n_filled_buckets}."
            )
        self.n_macroparticles = self.n_macroparticles_array.flatten()
        self.n_slices = int(
            self.number_of_slices_per_bucket * self.filling_pattern.sum()
        )
        self.bunch_indexes = (
            np.cumsum(self.filling_pattern) * self.filling_pattern - 1
        )
        self.bin_centers = self.bin_centers_array.flatten()

    def update_filling_pattern(
        self,
        updated_filling_pattern: NumpyArray,
    ):
        """
        Method to consider an update of filling pattern in case of
        multi-turn injection.

        Parameters
        ----------
        updated_filling_pattern
            Updated filling pattern
        """
        if len(self.filling_pattern) != len(updated_filling_pattern):
            raise ValueError(
                f"The length of the updated filling pattern does not match "
                f"the previously stored filling pattern lengths: "
                f"{len(updated_filling_pattern)} != "
                f"{len(self.filling_pattern)}"
            )
        additional_filled_buckets = self._set_additional_cuts(
            updated_filling_pattern=updated_filling_pattern
        )
        self._update_profile_lists(
            additional_filled_buckets=additional_filled_buckets
        )


class _SparseBaseClass:
    "Internal class for sparse profiles object."

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        _filling_pattern: NumpyArray,
        _profile_length_in_buckets: int,
        tracker="C",
        direct_slicing=False,
    ):
        """
        Common initialization.

        Parameters
        ----------
        rf_station
            RFStation object
        beam
            Beam object
        number_of_slices_per_profile
            Number of slices per bucket
        _filling_pattern
            Filling pattern of the synchrotron
        _profile_length_in_buckets
            Profile lengths in number of RF buckets
        tracker
            Choice of tracker. Can be "C" or "onebyone".
        direct_slicing
            Track at initialisation. FALSE by default.

        """
        self.beam = beam
        self.energy = beam.energy
        self.rf_station = rf_station

        self.number_of_slices_per_profile = number_of_slices_per_profile
        self._filling_pattern = _filling_pattern
        self._number_of_indexes = np.sum(_filling_pattern)
        self._profile_length_in_buckets = _profile_length_in_buckets

        # Index of each batch (-1 if empty). Only for C++ track
        self._bucket_indexes = (
            np.cumsum(_filling_pattern) * _filling_pattern - 1
        )

        self.n_slices = None
        self.profiles_list = None

        # Group n_macroparticles from all objects in a single array
        # (for C++ track).
        self.n_macroparticles_array = None
        self.bin_centers_array = None
        self.edges_array = None
        self.n_macroparticles = None
        self.bin_centers = None
        self.bin_size = None

        self.cut_left_array = None
        self.cut_right_array = None

        self.tracker = tracker
        self.direct_slicing = direct_slicing
        self.track = None

        # Pre-processing the slicing edges
        self.set_cuts(length_in_buckets=_profile_length_in_buckets)
        self.generate_profile_list()
        self.set_tracker()

    def generate_profile_list(self):
        self.profiles_list = []
        self.n_macroparticles_array = np.zeros(
            (self._number_of_indexes, self.number_of_slices_per_profile)
        )
        self.bin_centers_array = np.zeros(
            (self._number_of_indexes, self.number_of_slices_per_profile)
        )
        self.edges_array = np.zeros(
            (self._number_of_indexes, self.number_of_slices_per_profile + 1)
        )

        for i in range(self._number_of_indexes):
            # Only valid for cut_edges='edges'

            self.profiles_list.append(
                Profile(
                    self.beam,
                    CutOptions(
                        cut_left=self.cut_left_array[i],
                        cut_right=self.cut_right_array[i],
                        n_slices=self.number_of_slices_per_profile,
                    ),
                )
            )
            self.n_macroparticles_array[i, :] = self.profiles_list[
                i
            ].n_macroparticles
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.edges_array[i, :] = self.profiles_list[i].edges
        self._init_general_arrays()

    def set_cuts(self, length_in_buckets: int):
        """
        *Method to set the self.cut_left_array and self.cut_right_array
        properties, with the limits being an integer number of RF periods.
        This is done as a pre-processing.*

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

        self.cut_left_array = np.zeros(self._number_of_indexes)
        self.cut_right_array = np.zeros(self._number_of_indexes)
        bucket_indexes = np.where(self._filling_pattern != 0)[0]
        for i in range(self._number_of_indexes):
            bucket_index = bucket_indexes[i]
            self.cut_left_array[i] = bucket_index * t_rf
            self.cut_right_array[i] = (bucket_index + length_in_buckets) * t_rf

    def _init_general_arrays(self):
        """
        Method to update the general arrays after a profile update.
        """
        self.n_macroparticles = self.n_macroparticles_array.flatten()
        self.bin_centers = self.bin_centers_array.flatten()
        self.bin_size = self.profiles_list[0].bin_size
        self.n_slices = self.number_of_slices_per_profile * len(
            self.profiles_list
        )

    def _set_additional_cuts(
        self,
        _updated_filling_pattern: NumpyArray,
    ):
        """
        *Method to update total cut array properties with the new cut
        options for the additional filled buckets in the updated filling
        pattern, with the limits being
        an RF period.*
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
        _additional_indexes = len(filled_bunches_new) - len(
            filled_bunches_current
        )

        mask_additional_bunch = self._filling_pattern.copy()

        for i in filled_bunches_new:
            if self._filling_pattern[i] == 0:
                mask_additional_bunch[i] = 1
            else:
                mask_additional_bunch[i] = 0
        # fixme: injected bunch might be between already considered bunches
        updated_cut_left = np.zeros(_additional_indexes)
        updated_cut_right = np.zeros(_additional_indexes)
        masked_indexes = np.where(mask_additional_bunch != 0)[0]
        for i in range(_additional_indexes):
            bucket_index = masked_indexes[i]
            updated_cut_left[i] = bucket_index * t_rf
            updated_cut_right[i] = (bucket_index + 1) * t_rf
        self.cut_left_array = np.append(self.cut_left_array, updated_cut_left)
        self.cut_right_array = np.append(
            self.cut_right_array, updated_cut_right
        )
        self._filling_pattern = _updated_filling_pattern
        return _additional_indexes

    def _update_profile_lists(
        self,
        _additional_indexes: int,
    ):
        """
        *Method to update the total profile lists by creating individual
        profiles for the newly injected bunches*
        """

        # Initialize individual slicing objects
        profiles_list_additional = []
        _total_number_of_indexes = (
            self._number_of_indexes + _additional_indexes
        )
        if (len(self.cut_right_array) != _total_number_of_indexes) or (
            len(self.cut_left_array) != _total_number_of_indexes
        ):
            raise ValueError("Cut arrays have not been updated.")
        if (
            len(np.where(self._filling_pattern != 0)[0])
            != _total_number_of_indexes
        ):
            raise ValueError("Filling pattern has not been updated.")

        for i in range(_additional_indexes):
            # Only valid for cut_edges='edges'
            profiles_list_additional.append(
                Profile(
                    self.beam,
                    CutOptions(
                        cut_left=self.cut_left_array[
                            self._number_of_indexes + i
                        ],
                        cut_right=self.cut_right_array[
                            self._number_of_indexes + i
                        ],
                        n_slices=self.number_of_slices_per_profile,
                    ),
                )
            )
            self.n_macroparticles_array = np.insert(
                arr=self.n_macroparticles_array,
                obj=len(self.n_macroparticles_array),
                values=profiles_list_additional[i].n_macroparticles,
                axis=0,
            )
            self.bin_centers_array = np.insert(
                arr=self.bin_centers_array,
                obj=len(self.bin_centers_array),
                values=profiles_list_additional[i].bin_centers,
                axis=0,
            )
            self.edges_array = np.insert(
                arr=self.edges_array,
                obj=len(self.edges_array),
                values=profiles_list_additional[i].edges,
                axis=0,
            )
        self.profiles_list += profiles_list_additional
        self._number_of_indexes += _additional_indexes
        self._update_general_arrays()

    def _update_general_arrays(self):
        """
        Method to update the general arrays after a profile update.
        """
        # Total parameters
        if (
            len(np.where(self._filling_pattern != 0)[0])
            != self._number_of_indexes
        ):
            raise ValueError(
                f"Filling pattern has length "
                f"{len(np.where(self._filling_pattern)[0])}, number of "
                f"declared filled buckets "
                f"{self._number_of_indexes}."
            )
        self.n_macroparticles = self.n_macroparticles_array.flatten()
        self.n_slices = int(
            self.number_of_slices_per_profile * self._filling_pattern.sum()
        )
        self._bucket_indexes = (
            np.cumsum(self._filling_pattern) * self._filling_pattern - 1
        )
        self.bin_centers = self.bin_centers_array.flatten()

    def set_tracker(self):
        if self.tracker == "C":
            self.track = self._histogram_c
        elif self.tracker == "onebyone":
            self.track = self._histogram_one_by_one
        else:
            # WrongCalcError
            raise RuntimeError("Tracking method not recognized!")

        # Track at initialisation
        if self.direct_slicing:
            self.track()

    def _histogram_c(self):
        """
        *Histogram generated by calling an optimized C++ function that
        calculates all the profile at once.*
        """
        bm.sparse_histogram(
            self.beam.dt,
            self.n_macroparticles_array,
            self.cut_left_array,
            self.cut_right_array,
            self._bucket_indexes,
            self.number_of_slices_per_profile,
        )

    def _histogram_one_by_one(self):
        """
        *Histogram generated by calling the tack() method of each Profile
        object*
        """

        for i in range(len(self.profiles_list)):
            self.profiles_list[i].track()
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.n_macroparticles_array[i, :] = self.profiles_list[
                i
            ].n_macroparticles
        self.bin_centers = self.bin_centers_array.flatten()
        self.n_macroparticles = self.n_macroparticles_array.flatten()


class SparseBucket(_SparseBaseClass):
    '''
    *This class instantiates a Profile object for each filled bucket according
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
        Choice of tracker. Can be "C" or "onebyone".
    direct_slicing
        Track at initialisation. FALSE by default.
    """
    '''

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        bunch_list: NumpyArray,
        tracker="C",
        direct_slicing: bool = False,
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
            direct_slicing=direct_slicing,
        )

    @property
    def bunch_list(self):
        return self._filling_pattern

    @property
    def total_number_of_filled_buckets(self):
        return self._number_of_indexes

    @property
    def bunch_indexes(self):
        return self._bucket_indexes

    def update_bunch_list(
        self,
        updated_bunch_list: list[int],
    ):
        additional_filled_buckets = self._set_additional_cuts(
            _updated_filling_pattern=updated_bunch_list
        )
        self._update_profile_lists(
            _additional_indexes=additional_filled_buckets
        )


class SparseBatch(_SparseBaseClass):
    """
    *This class instantiates a Profile object for each batch according
    to the provided batch list.
    Each Profile object will be of the size of
    a batch and will have the same number of slices.*
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        batch_list: NumpyArray,
        batch_length: int = 1,
        tracker="C",
        direct_slicing: bool = False,
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
            direct_slicing=direct_slicing,
        )

    @property
    def batch_list(self):
        return self._filling_pattern

    @property
    def number_of_slices_per_bucket(self):
        return self.number_of_slices_per_profile / self.batch_length

    @property
    def total_number_of_batches(self):
        return self._number_of_indexes

    @property
    def total_number_of_sliced_buckets(self):
        return int(self._number_of_indexes * np.sum(self._filling_pattern))

    @property
    def batch_length(self):
        return self._profile_length_in_buckets

    @property
    def batch_indexes(self):
        return self._bucket_indexes

    def update_batch_list(
        self,
        updated_batch_list: list[int],
    ):
        additional_batches = self._set_additional_cuts(
            _updated_filling_pattern=updated_batch_list
        )
        self._update_profile_lists(_additional_indexes=additional_batches)
