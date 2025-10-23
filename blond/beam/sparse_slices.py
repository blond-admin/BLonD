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

:Authors: **Juan F. Esteban Mueller**
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
    *This class instantiates a Profile object for each filled bucket according
    to the provided filling pattern. Each Profile object will be of the size of
    an RF bucket and will have the same number of slices.*
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        n_slices_bucket: int,
        filling_pattern: NumpyArray,
        tracker: TrackerTypes = "C",
        direct_slicing: bool = False,
    ):
        #: *Import (reference) Beam*
        self.beam = beam

        #: *Import (reference) RFStation*
        self.rf_station = rf_station

        #: *Number of slices per bucket*
        self.n_slices_bucket = n_slices_bucket

        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        self.filling_pattern = filling_pattern

        # Bunch index for each filled bucket (-1 if empty). Only for C++ track
        self.bunch_indexes = np.cumsum(filling_pattern) * filling_pattern - 1

        #: *Number of buckets to be sliced*
        self.n_filled_buckets = int(np.sum(filling_pattern))

        # Pre-processing the slicing edges
        self.cut_left_array = np.zeros(self.n_filled_buckets)
        self.cut_right_array = np.zeros(self.n_filled_buckets)
        self.set_cuts()

        # Initialize individual slicing objects
        self.profiles_list = []
        # Group n_macroparticles from all objects in a single array
        # (for C++ track).
        self.n_macroparticles_array = np.zeros(
            (self.n_filled_buckets, n_slices_bucket)
        )
        # Group bin_centers from all objects in a single array (for impedance)
        self.bin_centers_array = np.zeros(
            (self.n_filled_buckets, n_slices_bucket)
        )
        self.edges_array = np.zeros(
            (self.n_filled_buckets, n_slices_bucket + 1)
        )
        for i in range(self.n_filled_buckets):
            # Only valid for cut_edges='edges'

            self.profiles_list.append(
                Profile(
                    beam,
                    CutOptions(
                        cut_left=float(self.cut_left_array[i]),
                        cut_right=float(self.cut_right_array[i]),
                        n_slices=n_slices_bucket,
                    ),
                )
            )

            self.profiles_list[
                i
            ].n_macroparticles = self.n_macroparticles_array[i, :]
            self.bin_centers_array[i, :] = self.profiles_list[i].bin_centers
            self.edges_array[i, :] = self.profiles_list[i].edges
            self.profiles_list[i].bin_centers = self.bin_centers_array[i, :]

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
        for i in range(self.n_filled_buckets):
            bucket_index = np.where(self.filling_pattern)[0][i]
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
            self.n_slices_bucket,
        )

    def _histrogram_one_by_one(self):
        from warnings import warn

        warn(
            "_histrogram_one_by_one is deprecated, use _histogram_one_by_one",
            DeprecationWarning,
            stacklevel=2,
        )
        self._histogram_one_by_one()

    def _histogram_one_by_one(self):
        """
        *Histogram generated by calling the tack() method of each Profile
        object*
        """

        for i in range(self.n_filled_buckets):
            self.profiles_list[i].track()
