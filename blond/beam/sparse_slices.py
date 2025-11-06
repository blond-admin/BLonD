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

from typing import TYPE_CHECKING

import numpy as np

from .profile import CutOptions, Profile
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray as NumpyArray

    from .beam import Beam
    from ..input_parameters.rf_parameters import RFStation

    TrackerTypes = Literal["C", "onebyone"]


class SparseSlices:
    """
    This class instantiates a Profile object for each filled bucket according
    to the provided filling pattern.

    By default, each Profile object has the size of an RF bucket, and the
    same number of slices (number_of_slices_per_bucket). The size of the
    Profile objects can be extended to the neighbouring buckets by inputting
    a "bucket_margin", for the same number of slices per bucket.

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
    bucket_margin
        Extend the scope of the Profile objects to the neighbouring buckets.
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
        bucket_margin: int = 0,
        direct_slicing: bool = False,
    ):
        #: *Import (reference) Beam*
        self.beam = beam
        self.energy = self.beam.energy
        #: *Import (reference) RFStation*
        self.rf_station = rf_station

        #: *Number of slices per bucket*
        self.bucket_margin = bucket_margin
        self.number_of_slices_per_bucket = number_of_slices_per_bucket
        self.number_of_slices_per_profile = (
            self.number_of_slices_per_bucket * (1 + 2 * self.bucket_margin)
        )
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        self.filling_pattern = filling_pattern

        # Bunch index for each filled bucket (-1 if empty). Only for C++ track
        self.bunch_indexes = (
            np.cumsum(self.filling_pattern) * self.filling_pattern - 1
        )

        #: *Number of filled buckets in the filling pattern*
        self.n_filled_buckets = int(np.sum(self.filling_pattern))
        #: *Number of buckets to be sliced (including the bucket_margin)*
        self.n_sliced_buckets = int(np.sum(self.filling_pattern)) * (
            1 + 2 * self.bucket_margin
        )

        # Pre-processing the slicing edges
        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        self.set_cuts(
            bucket_margin=self.bucket_margin,
        )

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

            self.profiles_list[
                i
            ].n_macroparticles = self.n_macroparticles_array[i, :]
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.edges_array[i, :] = self.profiles_list[i].edges
            self.profiles_list[i].bin_centers = self.bin_centers_array[i, :]

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

    def set_cuts(self, bucket_margin: int = 0):
        """
        *Method to set the self.cut_left_array and self.cut_right_array
        properties, with the limits being an RF period.
        This is done as a pre-processing.*

        Parameters
        ----------
        bucket_margin
            Extend the scope of the Profile objects to the neighbouring buckets.
        """
        # RF period
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]

        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        for i in range(self.n_filled_buckets):
            bucket_index = np.where(self.filling_pattern)[0][i]
            self.cut_left_array[i] = (bucket_index - bucket_margin) * t_rf
            self.cut_right_array[i] = (bucket_index + 1 + bucket_margin) * t_rf

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
