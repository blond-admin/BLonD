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
from ..utils.build_wrap_python import sparse_histogram

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import NDArray as NumpyArray

    from .beam import Beam
    from ..input_parameters.rf_parameters import RFStation

    TrackerTypes = Literal["C", "onebyone"]


class SparseProfileBaseClass:
    "Base class for sparse profiles object."

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        _filling_pattern: NumpyArray,
        _profile_length_in_buckets: int,
        tracker_mode: TrackerTypes = "C",
        do_track_on_init: bool = False,
    ):
        """
        Common initialization of Sparse objects.

        This object takes the input _filling_pattern and creates a profile
        object of length _profile_length_in_buckets (in number of buckets)
        per non-zero elements in the _filling_pattern.
        Initialisation happens in the following order:
        - input storage,
        - _set_cuts(): defines the cut left and right options for the
        profile to focus on the filled buckets. The left cut starts at the bucket index considered and
        the right cut is defined by its distance, in number of RF buckets (
        input length_in_buckets), from the bucket index considered.
        - _generate_profile_lists(): creates the profile objects per
        non-zero elements in the _filling_pattern, using the cut_left and
        cut_right properties defined at the previous stage.
        - if do_track_on_init, tracks at initialisation.

        Parameters
        ----------
        rf_station
            RFStation object
        beam
            Beam object
        number_of_slices_per_profile
            Number of slices per profile
        _filling_pattern
            Filling pattern / Bunch list/ Batch list of the synchrotron.
            Defined as a mask arrray of filled buckets.
        _profile_length_in_buckets
            Profile lengths in number of RF buckets. Should be greater than 1.
        tracker_mode
            Choice of tracker. Can be "C" or "onebyone". Default is "C".
        do_track_on_init
            Enables tracking at initialisation. FALSE by default.

        Returns
        --------
        ValueError if the filling pattern length is strictly higher than the
        harmonic number of the main RF station.
        Warning if the filling pattern length is strictly lower than the
        harmonic number of the main RF station.
        TypeError if the number of buckets covered by the profile is not an
        integer.

        Illustration - this class is not intended for direct use
        --------
        >>> import numpy as np
        >>> from blond.beam.sparse_profiles import SparseBatch
        >>> from blond.input_parameters.ring import Ring
        >>> from blond.beam.beam import Beam
        >>> from blond.input_parameters.rf_parameters import RFStation
        >>> ring = Ring(n_turns = 1, ring_length = 100,
        >>> alpha = 0.00001, momentum = 1e9)
        >>> beam = Beam(ring, 10000, 1e10)
        >>> rf_params = RFStation(ring=ring, n_rf=1, harmonic=[4620],
        ...                  voltage=[7e6], phi_rf_d=[0.])
        >>> number_of_bunches_per_batch = 5
        >>> bunch_spacing = 5
        >>> _filling_pattern = np.zeros(rf_params.harmonic[0])
        >>> _filling_pattern[::bunch_spacing] = 1
        >>> _profile_length_in_buckets = 1
        >>> sparse_profile = SparseProfileBaseClass(rf_station=rf_params,
        >>>                                beam = beam,
        >>>                                number_of_slices_per_profile = 1e4,
        >>>                                _filling_pattern = _filling_pattern,
        >>>                                _profile_length_in_buckets = _profile_length_in_buckets)
        """
        if len(_filling_pattern) > rf_station.harmonic[0][0]:
            raise ValueError(
                f"The length of filling_pattern exceeds "
                f"the number of RF buckets"
            )

        if len(_filling_pattern) != rf_station.harmonic[0][0]:
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

        self.tracker_mode = tracker_mode
        self.do_track_on_init = do_track_on_init

        # Pre-processing the slicing edges
        self._set_cuts(length_in_buckets=_profile_length_in_buckets)
        self._generate_profile_list()

        # Track at initialisation
        # if python is enabled, tracking should be done at initialisation to
        # reinit the n_macroparticles aray per profile.
        if tracker_mode == "onebyone":
            self.track()
        else:
            if self.do_track_on_init:
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
        return np.concatenate(
            [profile.n_macroparticles for profile in self.profiles_list]
        )

    @property
    def bin_centers(self):
        return np.concatenate(
            [profile.bin_centers for profile in self.profiles_list]
        )

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
        Internal method to initialise the cut options of each profile.

        Internal method to set the self.cut_left_array and self.cut_right_array
        properties. The left cut starts at the bucket index considered and
        the right cut is defined by its distance, in number of RF buckets (
        input length_in_buckets), from the bucket index considered.

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
        if self.tracker_mode == "C":
            self._histogram_c()
        elif self.tracker_mode == "onebyone":
            self._histogram_one_by_one()
        else:
            # WrongCalcError
            raise RuntimeError("Tracking method not recognized!")

    def _histogram_c(self):
        """
        Histogram generated by calling an optimized C++ function (or its
        numba/python port, depending on the bmath backend) that calculates
        all the profiles at once.

        The routine works on a regular grid of cells of equal length that
        starts at the left edge of its first cell, and expects a lookup
        table giving, for each grid cell, the row of the output histogram
        the cell belongs to (-1 for an empty cell). The cells are therefore
        the RF buckets of all the windows, sorted in time, so that windows
        of several buckets, windows starting anywhere in the turn and
        windows appended out of time order (multi-turn injection) are all
        histogrammed correctly.
        """
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]
        n_buckets = self._profile_length_in_buckets
        n_slices_bucket, remainder = divmod(
            int(self.number_of_slices_per_profile), n_buckets
        )
        if remainder != 0:
            raise ValueError(
                f"The number of slices per profile "
                f"({self.number_of_slices_per_profile}) must be a multiple "
                f"of the profile length in buckets ({n_buckets}) for "
                f'tracker_mode="C".'
            )

        # One row per RF bucket of each window, in time order
        cut_left_rows = np.concatenate(
            [
                cut_left + np.arange(n_buckets) * t_rf
                for cut_left in self.cut_left_array
            ]
        )
        order = np.argsort(cut_left_rows)
        cut_left_rows = cut_left_rows[order]
        cut_right_rows = cut_left_rows + t_rf

        # Grid cell (bucket offset from the first row) -> row, -1 if empty;
        # one spare cell for a particle exactly at the last right edge
        cells = np.rint((cut_left_rows - cut_left_rows[0]) / t_rf).astype(int)
        rows_of_cells = -np.ones(cells[-1] + 2)
        rows_of_cells[cells] = np.arange(len(cells))

        histogram = np.zeros((len(cells), n_slices_bucket))
        bm.sparse_histogram(
            self.beam.dt,
            histogram,
            cut_left_rows,
            cut_right_rows,
            rows_of_cells,
            n_slices_bucket,
        )

        # Back to one row per window, buckets side by side
        nma = np.empty((self._number_of_indices, n_buckets * n_slices_bucket))
        nma.reshape(-1, n_slices_bucket)[order] = histogram

        # Update the n_macroparticle arrays of all internal profiles
        for i in range(self._number_of_indices):
            self.profiles_list[i].n_macroparticles = nma[i, :]

    def _histogram_one_by_one(self):
        """
        Histogram generated by calling the track() method of each Profile
        object
        """

        for i in range(len(self.profiles_list)):
            self.profiles_list[i].track()

    def _histogram_one_by_one_numpy(self):
        """
        Histogram generated by calling the track() method of each Profile
        object
        """

        for profile in self.profiles_list:
            hist_sparse, _ = np.histogram(
                self.beam.dt,
                bins=len(profile.n_macroparticles),
                range=(profile.cut_left, profile.cut_right),
            )
            # same dtype as the C and Profile.track histograms, so that the
            # profile can be tracked by any of the three afterwards
            profile.n_macroparticles = hist_sparse.astype(float)

    def _set_additional_cuts(
        self,
        _updated_filling_pattern: NumpyArray,
        injection_ordered_indices: NumpyArray | None = None,
    ):
        """
        Internal method to update the cut array properties of the Sparse
        object with new cut_left | cut_right options around the additional
        indices.

        The left cut starts at the bucket index considered and the right cut
        is defined by its distance, in number of RF buckets (
        self._profile_length_in_buckets), from the bucket index considered.
        This is done as a pre-processing.

        The appending ORDER matters: the trackers pair profiles_list[i] with
        the beam particle block [i * N: (i + 1) * N], so new profiles must be
        appended in the order the corresponding particles were appended to
        the beam (injection order), which is not in general the bucket-index
        order. A single new index is unambiguous; several new indices at once
        are accepted only together with injection_ordered_indices, which gives the
        injection order explicitly.

        Parameters
        ----------
        _updated_filling_pattern
            Updated filling pattern (0/1 per bucket).
        injection_ordered_indices
            Bucket indices of the new profiles in injection order. Must
            contain exactly the indices that are new in
            _updated_filling_pattern. Optional when only one index is new.

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

        if injection_ordered_indices is not None:
            injection_ordered_indices = np.asarray(
                injection_ordered_indices, dtype=int
            ).ravel()
            if not np.array_equal(
                np.sort(injection_ordered_indices), masked_indices
            ):
                raise ValueError(
                    f"The ordered new indices {injection_ordered_indices} do not "
                    f"match the indices that are new in the updated filling "
                    f"pattern {masked_indices}"
                )
            masked_indices = injection_ordered_indices
        elif _additional_indices > 1:
            raise ValueError(
                f"{_additional_indices} new indices were added in a single "
                f"update call without their injection order. Profiles are "
                f"paired with beam particle blocks by creation order, and "
                f"appending several new profiles in bucket-index order "
                f"silently mispairs (and thereby un-kicks) every batch whose "
                f"injection order differs from its bucket-index order. "
                f"Either update one index per call, in injection order, or "
                f"pass the new bucket indices in injection order via the "
                f"ordered-indices argument."
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


class SparseBucket(SparseProfileBaseClass):
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
    tracker_mode
        Choice of tracker. Can be "C" or "onebyone". Default is "C".
    do_track_on_init
        Track at initialisation. FALSE by default.


    Example
    --------
    >>> import numpy as np
    >>> from blond.beam.sparse_profiles import SparseBucket
    >>> from blond.input_parameters.ring import Ring
    >>> from blond.beam.beam import Beam
    >>> from blond.input_parameters.rf_parameters import RFStation
    >>> ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> beam = Beam(ring, 10000, 1e10)
    >>> rf_params = RFStation(ring=ring, n_rf=1, harmonic=[4620],
    ...                  voltage=[7e6], phi_rf_d=[0.])
    >>> bunch_spacing = 10
    >>> bunch_list = np.zeros(rf_params.harmonic[0])
    >>> bunch_list[::bunch_spacing] = 1
    >>> sparse_profile = SparseBucket(rf_station=rf_params,
    >>>                                beam = beam,
    >>>                                number_of_slices_per_profile = 1e4,
    >>>                                bunch_list = bunch_list)
    >>>
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        bunch_list: NumpyArray,
        tracker_mode: TrackerTypes = "C",
        do_track_on_init: bool = False,
    ):
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        super().__init__(
            rf_station=rf_station,
            beam=beam,
            number_of_slices_per_profile=number_of_slices_per_profile,
            _filling_pattern=bunch_list,
            _profile_length_in_buckets=1,
            tracker_mode=tracker_mode,
            do_track_on_init=do_track_on_init,
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
        new_bunch_indices: list[int] | None = None,
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
        new_bunch_indices
            Bucket indices of the newly injected bunches in INJECTION order.
            Required when more than one bunch is new in this call: profiles
            are paired with beam particle blocks by creation order, so the
            new profiles must be appended in the order the bunches were
            appended to the beam.
        """
        additional_filled_buckets = self._set_additional_cuts(
            _updated_filling_pattern=updated_bunch_list,
            injection_ordered_indices=new_bunch_indices,
        )
        self._update_profile_lists(
            _additional_indices=additional_filled_buckets
        )


class SparseBatch(SparseProfileBaseClass):
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
    tracker_mode
        Choice of tracker. Can be "C" or "onebyone". Default is "C".
    do_track_on_init
        Track at initialisation. FALSE by default.
    Example
    --------
    >>> import numpy as np
    >>> from blond.beam.sparse_profiles import SparseBatch
    >>> from blond.beam.beam import Beam
    >>> from blond.input_parameters.ring import Ring
    >>> from blond.input_parameters.rf_parameters import RFStation
    >>> ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> beam = Beam(ring, 10000, 1e10)
    >>> rf_params = RFStation(ring=ring, n_rf=1, harmonic=[4620],
    ...                  voltage=[7e6], phi_rf_d=[0.])
    >>> number_of_bunches_per_batch = 5
    >>> bunch_spacing = 5
    >>> batch_spacing = 50
    >>> batch_length = number_of_bunches_per_batch * bunch_spacing
    >>> batch_list = np.zeros(rf_params.harmonic[0])
    >>> batch_list[::batch_spacing] = 1
    >>> sparse_profile = SparseBatch(rf_station=rf_params,
    >>>                                beam = beam,
    >>>                                number_of_slices_per_profile = 1e4,
    >>>                                batch_list = batch_list,
    >>>                                batch_length = batch_length)
    >>>
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int,
        batch_list: NumpyArray,
        batch_length: int = 1,
        tracker_mode: TrackerTypes = "C",
        do_track_on_init: bool = False,
    ):
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        super().__init__(
            rf_station=rf_station,
            beam=beam,
            number_of_slices_per_profile=number_of_slices_per_profile,
            _filling_pattern=batch_list,
            _profile_length_in_buckets=batch_length,
            tracker_mode=tracker_mode,
            do_track_on_init=do_track_on_init,
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
        new_batch_indices: list[int] | None = None,
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
        new_batch_indices
            Bucket indices of the newly injected batches in INJECTION order.
            Required when more than one batch is new in this call: profiles
            are paired with beam particle blocks by creation order, so the
            new profiles must be appended in the order the batches were
            appended to the beam.
        """
        additional_batches = self._set_additional_cuts(
            _updated_filling_pattern=updated_batch_list,
            injection_ordered_indices=new_batch_indices,
        )
        self._update_profile_lists(_additional_indices=additional_batches)

def SparseSlices(
    rf_station: RFStation,
    beam: Beam,
    n_slices_bucket: int,
    filling_pattern: NumpyArray,
    tracker_mode: TrackerTypes = "C",
    do_track_on_init: bool = False,
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
    tracker_mode
        Choice of tracker. Can be "C" or "onebyone". Default is "C".
    do_track_on_init
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
        tracker_mode=tracker_mode,
        do_track_on_init=do_track_on_init,
    )

class SparseBunch(SparseProfileBaseClass):
    """
    This class instantiates a Profile object for each filled bucket according
    to the provided filling pattern or bunch list, and narrows it down to
    the beam coordinates: one window per bunch, centred on the bunch, all
    windows of the same width (2 * length_in_sigma times the rms length of
    the longest bunch) and of the same bin size, so that the one-pass kick
    and the fine-grid cavity model, which assume one bin size, apply.

    Two ways to set the grid:

    - number_of_slices_per_bucket: the bin size is the one of the RF
      bucket grid (t_rf / number_of_slices_per_bucket, as SparseBucket),
      the windows are an integer number of bins wide and sit on the global
      bin lattice (edges at integer multiples of the bin size);
    - number_of_slices_per_profile: that many bins per window, the bin size
      following from the window width.

    With recenter_every, after every recenter_every-th track() each window
    is moved onto the centroid of its own histogram by a whole number of
    bins, so that the windows follow a drifting bunch. After every track()
    the macroparticles outside all windows (neither histogrammed nor
    kicked) are counted in n_macroparticles_outside, with a warning when
    their fraction exceeds warn_outside_fraction.

    Parameters
    ----------
    rf_station
        RFStation object
    beam
        Beam object
    number_of_slices_per_profile
        Number of slices per window (give this or
        number_of_slices_per_bucket)
    bunch_list
        Bunch list (or filling pattern) of the synchrotron
    length_in_sigma
        Half-width of the windows in rms lengths of the longest bunch.
    tracker_mode
        Only "onebyone": the windows are not aligned on the RF buckets, which
        the one-pass "C" histogram requires.
    do_track_on_init
        Track at initialisation. FALSE by default.
    number_of_slices_per_bucket
        Bins per RF bucket of the grid the windows are built on (give this
        or number_of_slices_per_profile)
    recenter_every
        Re-centre the windows on their bunches every that many track()
        calls; None (default) keeps the windows where they were set.
    warn_outside_fraction
        Warn when more than this fraction of the macroparticles is outside
        all windows (default 0: any); None disables the warning.
    resize_windows
        With recenter_every: at every re-centring, if the outer tenth of
        the bins on either side of any window holds more than
        resize_tail_fraction of its particles (a bunch that got longer than
        the margin), all windows are widened by resize_factor (whole bins,
        same lattice, re-centred and histogrammed again), repeatedly until
        the tails fit. Default True. The windows never shrink.
    resize_tail_fraction
        Fraction of the particles of a window in its outer bins above which
        the windows are widened (default 1e-3; a Gaussian bunch 2 x 5 rms
        lengths wide has 6e-5 beyond 4 rms lengths).
    resize_factor
        Growth factor of the window width at each widening (default 1.5).


    Example
    --------
    >>> import numpy as np
    >>> from blond.beam.sparse_profiles import SparseBucket
    >>> from blond.input_parameters.ring import Ring
    >>> from blond.beam.beam import Beam
    >>> from blond.input_parameters.rf_parameters import RFStation
    >>> ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> beam = Beam(ring, 10000, 1e10)
    >>> rf_params = RFStation(ring=ring, n_rf=1, harmonic=[4620],
    ...                  voltage=[7e6], phi_rf_d=[0.])
    >>> bunch_spacing = 10
    >>> bunch_list = np.zeros(rf_params.harmonic[0])
    >>> bunch_list[::bunch_spacing] = 1
    >>> sparse_profile = SparseBucket(rf_station=rf_params,
    >>>                                beam = beam,
    >>>                                number_of_slices_per_profile = 1e4,
    >>>                                bunch_list = bunch_list)
    >>>
    """

    def __init__(
        self,
        rf_station: RFStation,
        beam: Beam,
        number_of_slices_per_profile: int | None = None,
        bunch_list: NumpyArray | None = None,
        length_in_sigma: float = 5,
        tracker_mode: TrackerTypes = "C",
        do_track_on_init: bool = False,
        number_of_slices_per_bucket: int | None = None,
        recenter_every: int | None = None,
        warn_outside_fraction: float | None = 0.0,
        resize_windows: bool = True,
        resize_tail_fraction: float = 1e-3,
        resize_factor: float = 1.5,
    ):
        #: *Filling pattern as a boolean array where True (1) means filled
        # bucket*
        if tracker_mode == "C" and number_of_slices_per_bucket is None:
            raise ValueError(
                'tracker_mode="C" (one pass over the particles for all '
                "windows) needs the windows on the bin lattice of the RF "
                "bucket grid: give number_of_slices_per_bucket, or use "
                'tracker_mode="onebyone".'
            )
        if length_in_sigma <= 0:
            raise ValueError(
                "The margin of the profile scope should be a "
                "positive number of bunch lengths."
            )
        if bunch_list is None:
            raise ValueError("bunch_list is required.")
        if (number_of_slices_per_profile is None) == (
            number_of_slices_per_bucket is None
        ):
            raise ValueError(
                "Give either number_of_slices_per_bucket (bin size of the "
                "RF bucket grid, windows an integer number of bins wide on "
                "that grid) or number_of_slices_per_profile (bins per "
                "window), not both."
            )
        if recenter_every is not None and recenter_every < 1:
            raise ValueError("recenter_every must be a positive number of turns.")
        if resize_factor <= 1:
            raise ValueError("resize_factor must be larger than 1.")
        self.resize_windows = bool(resize_windows)
        self.resize_tail_fraction = float(resize_tail_fraction)
        self.resize_factor = float(resize_factor)
        # read by _set_cuts, which the base-class initialisation calls
        self.length_in_sigma = float(length_in_sigma)
        self.number_of_slices_per_bucket = (
            None if number_of_slices_per_bucket is None
            else int(number_of_slices_per_bucket)
        )
        self.recenter_every = (
            None if recenter_every is None else int(recenter_every)
        )
        self.warn_outside_fraction = warn_outside_fraction
        #: *Common width [s] and bin size [s] of the windows (set by _set_cuts)*
        self.window_width = None
        self.window_bin_size = None
        #: *Macroparticles outside all windows after the last track()*
        self.n_macroparticles_outside = 0
        self._turns_tracked = 0
        super().__init__(
            rf_station=rf_station,
            beam=beam,
            number_of_slices_per_profile=number_of_slices_per_profile,
            _filling_pattern=bunch_list,
            _profile_length_in_buckets=1,
            tracker_mode=tracker_mode,
            do_track_on_init=do_track_on_init,
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

    def _bunch_statistics(self, bucket_indices, dt=None):
        """
        Centroid and rms length [s] of the bunch of each of the given RF
        buckets, from the particles found in the bucket (one pass over the
        particles: bucket index per particle, then sums per bucket of the
        coordinate relative to the bucket centre, so that the variance does
        not suffer from cancellation). `dt` restricts the particles
        considered (default: the whole beam). A bucket without particles
        gets its centre and a zero length, with a warning.
        """
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]
        dt = np.asarray(self.beam.dt if dt is None else dt, dtype=float)
        bucket_indices = np.asarray(bucket_indices, dtype=int)
        bucket_of = np.floor(dt / t_rf).astype(np.int64)
        valid = bucket_of >= 0
        bucket_of = bucket_of[valid]
        relative = dt[valid] - (bucket_of + 0.5) * t_rf
        n_buckets = int(max(bucket_of.max(initial=0),
                            bucket_indices.max(initial=0))) + 1
        counts = np.bincount(bucket_of, minlength=n_buckets)
        sums = np.bincount(bucket_of, weights=relative, minlength=n_buckets)
        squares = np.bincount(bucket_of, weights=relative**2,
                              minlength=n_buckets)
        n = counts[bucket_indices]
        centers = (bucket_indices + 0.5) * t_rf
        sigmas = np.zeros(len(bucket_indices))
        filled = n > 0
        if filled.any():
            mean = sums[bucket_indices[filled]] / n[filled]
            centers[filled] += mean
            sigmas[filled] = np.sqrt(np.maximum(
                squares[bucket_indices[filled]] / n[filled] - mean**2, 0.0))
        if not filled.all():
            warnings.warn(
                f"{np.count_nonzero(~filled)} SparseBunch window(s) have no "
                "particles in their RF bucket: the bucket centre is used as "
                "window centre.",
                UserWarning,
                stacklevel=3,
            )
        return centers, sigmas

    def _block_statistics(self, blocks):
        """
        Centroid and rms length [s] of given particle blocks (slices or
        index arrays of beam.dt), one window each.
        """
        centers, sigmas = [], []
        for block in blocks:
            dt = self.beam.dt[block]
            centers.append(float(np.mean(dt)))
            sigmas.append(float(np.std(dt)))
        return np.array(centers, dtype=float), np.array(sigmas, dtype=float)

    def _windows_around(self, centers):
        """
        Edges of windows of the common width centred on `centers`; on the
        bucket grid the left edges are moved to the nearest lattice point,
        so the centre is within half a bin of the centroid.
        """
        cut_left = np.asarray(centers, dtype=float) - 0.5 * self.window_width
        if self.number_of_slices_per_bucket is not None:
            cut_left = np.rint(cut_left / self.window_bin_size) * self.window_bin_size
        return cut_left, cut_left + self.window_width

    def _bunch_windows(self, bucket_indices, new_particles=None):
        """
        Windows of the common width around the bunches of the given RF
        buckets (see _windows_around), for bunches added after the
        initialisation. With new_particles, the statistics of the new
        bunches come from the injected particles only: one block (slice or
        index array of beam.dt) per new bunch, in the order of
        bucket_indices, or a single block that is split by RF bucket. A
        bunch longer than the windows is reported.

        Returns
        -------
        cut_left, cut_right
            Arrays of the window edges [s], in the order of bucket_indices.
        """
        if new_particles is None:
            centers, sigmas = self._bunch_statistics(bucket_indices)
        elif isinstance(new_particles, (list, tuple)):
            if len(new_particles) != len(bucket_indices):
                raise ValueError(
                    f"{len(new_particles)} particle blocks for "
                    f"{len(bucket_indices)} new bunches."
                )
            centers, sigmas = self._block_statistics(new_particles)
        else:
            centers, sigmas = self._bunch_statistics(
                bucket_indices, dt=self.beam.dt[new_particles]
            )
        if np.any(2 * self.length_in_sigma * sigmas > self.window_width):
            warnings.warn(
                "An injected bunch is longer than the SparseBunch windows "
                f"({self.window_width:.3e} s, set from the bunches present at "
                "the initialisation).",
                UserWarning,
                stacklevel=2,
            )
        return self._windows_around(centers)

    def _set_cuts(self, length_in_buckets: int = 1):
        """
        Internal method to initialise the cut options of each profile: one
        window per filled bucket, centred on the bunch it holds. The common
        width is 2 * length_in_sigma rms lengths of the longest bunch,
        rounded up to a whole number of bins of the bucket grid when
        number_of_slices_per_bucket is given; otherwise the bin size is the
        width over number_of_slices_per_profile.

        Parameters
        ----------
        length_in_buckets
            Argument of the base-class method, not used: the windows follow
            the bunches, not the buckets.
        """
        t_rf = self.rf_station.t_rf[0, self.rf_station.counter[0]]
        bucket_indices = np.where(self._filling_pattern != 0)[0]
        centers, sigmas = self._bunch_statistics(bucket_indices)
        width = 2 * self.length_in_sigma * np.max(sigmas) if sigmas.size else 0.0
        if width <= 0:
            # no particles yet: one RF bucket per window
            width = t_rf
        if self.number_of_slices_per_bucket is not None:
            self.window_bin_size = t_rf / self.number_of_slices_per_bucket
            n_bins = int(np.ceil(width / self.window_bin_size - 1e-9))
            self.number_of_slices_per_profile = n_bins
        else:
            n_bins = int(self.number_of_slices_per_profile)
            self.window_bin_size = width / n_bins
        self.window_width = n_bins * self.window_bin_size
        self.cut_left_array, self.cut_right_array = self._windows_around(
            centers
        )

    def track(self):
        """
        Histogram the windows (see the base class), then re-centre them on
        their bunches when due (recenter_every) and count the
        macroparticles outside all windows.
        """
        super().track()
        self._turns_tracked += 1
        if (
            self.recenter_every is not None
            and self._turns_tracked % self.recenter_every == 0
        ):
            self.recenter_windows()
            if self.resize_windows:
                self.resize_windows_if_needed()
        self._check_coverage()

    def _tail_fractions(self):
        """Fraction of the particles of every window found in the outer
        tenth of its bins on either side (nan for an empty window)."""
        histogram = self.n_macroparticles_array
        n_bins = histogram.shape[1]
        edge = max(1, int(round(0.1 * n_bins)))
        totals = histogram.sum(axis=1)
        tails = histogram[:, :edge].sum(axis=1) + histogram[:, -edge:].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(totals > 0, tails / totals, np.nan)

    def resize_windows_if_needed(self, max_iterations: int = 5):
        """
        Widen all windows by resize_factor while the outer bins of any
        window hold more than resize_tail_fraction of its particles, i.e.
        while a bunch is longer than the margin (at most max_iterations
        times). Each widening keeps the bin size and the lattice, centres
        every window on its centroid and histograms the windows again.

        Returns
        -------
        widened
            Number of widenings applied.
        """
        widened = 0
        for _ in range(max_iterations):
            tails = self._tail_fractions()
            if not np.any(tails > self.resize_tail_fraction):
                break
            n_bins = int(np.ceil(self.number_of_slices_per_profile * self.resize_factor))
            # never wider than the distance between neighbouring windows
            centers = np.sort(0.5 * (np.asarray(self.cut_left_array)
                                     + np.asarray(self.cut_right_array)))
            if len(centers) > 1:
                n_bins = min(n_bins, int(np.floor(
                    np.min(np.diff(centers)) / self.window_bin_size)))
            if n_bins <= self.number_of_slices_per_profile:
                warnings.warn(
                    "SparseBunch windows cannot be widened further without "
                    "overlapping neighbouring windows; particles beyond "
                    f"{self.window_width:.3e} s stay outside.",
                    UserWarning,
                    stacklevel=2,
                )
                break
            self._set_window_bins(n_bins)
            widened += 1
        if widened:
            warnings.warn(
                f"SparseBunch windows widened to {self.window_width:.3e} s "
                f"({self.number_of_slices_per_profile} bins): a bunch got "
                "longer than the margin.",
                UserWarning,
                stacklevel=2,
            )
        return widened

    def _set_window_bins(self, n_bins: int):
        """
        Give every window n_bins bins of the current bin size, centred on
        the centroid of its current histogram (the window centre when it is
        empty), on the lattice; the Profile objects are rebuilt and the
        windows histogrammed again.
        """
        n_bins = int(n_bins)
        centers = np.empty(len(self.profiles_list))
        for i, profile in enumerate(self.profiles_list):
            weights = profile.n_macroparticles
            total = float(np.sum(weights))
            centers[i] = (
                float(np.dot(profile.bin_centers, weights)) / total
                if total > 0
                else 0.5 * (profile.cut_left + profile.cut_right)
            )
        self.number_of_slices_per_profile = n_bins
        self.window_width = n_bins * self.window_bin_size
        self.cut_left_array, self.cut_right_array = self._windows_around(centers)
        self.profiles_list = [
            Profile(
                self.beam,
                CutOptions(
                    cut_left=self.cut_left_array[i],
                    cut_right=self.cut_right_array[i],
                    n_slices=n_bins,
                ),
            )
            for i in range(len(self.profiles_list))
        ]
        self._histogram_windows(list(range(len(self.profiles_list))))

    def recenter_windows(self, max_iterations: int = 5):
        """
        Move every window onto the centroid of its own histogram, by a
        whole number of bins (the grid stays aligned), and histogram the
        moved windows again. The centroid of a bunch that drifted partly
        out of its window is biased towards the window, so the step is
        repeated on the moved windows until none moves (at most
        max_iterations times). Windows without particles stay.

        Returns
        -------
        moved
            Indices (in profiles_list) of the windows that were moved.
        """
        moved_at_all = set()
        candidates = range(len(self.profiles_list))
        for _ in range(max_iterations):
            moved = []
            for i in candidates:
                profile = self.profiles_list[i]
                weights = profile.n_macroparticles
                total = float(np.sum(weights))
                if total <= 0:
                    continue
                centroid = float(np.dot(profile.bin_centers, weights)) / total
                shift = centroid - 0.5 * (profile.cut_left + profile.cut_right)
                n_bins = int(np.rint(shift / self.window_bin_size))
                if n_bins == 0:
                    continue
                delta = n_bins * self.window_bin_size
                profile.cut_left += delta
                profile.cut_right += delta
                profile.edges += delta
                profile.bin_centers += delta
                self.cut_left_array[i] += delta
                self.cut_right_array[i] += delta
                moved.append(i)
            self._histogram_windows(moved)
            moved_at_all.update(moved)
            if not moved:
                break
            candidates = moved
        return sorted(moved_at_all)

    def _histogram_windows(self, indices):
        """Histogram the given windows again: one pass for all windows in
        "C" mode, window by window otherwise."""
        if not indices:
            return
        if self.tracker_mode == "C":
            self._histogram_c()
        else:
            for i in indices:
                self.profiles_list[i].track()

    def _histogram_c(self):
        """
        One pass over the particles for all windows
        (bm.sparse_bunch_histogram): the windows sit on the bin lattice of
        the RF bucket grid, so the bin index of a particle locates its
        window by bisection among the first bins of the windows, sorted in
        time. Particles outside every window are skipped.
        """
        n_bins = int(self.number_of_slices_per_profile)
        first_bins = np.rint(
            np.asarray(self.cut_left_array, dtype=float) / self.window_bin_size
        ).astype(np.int64)
        order = np.argsort(first_bins, kind="stable").astype(np.int64)
        histogram = np.zeros((len(first_bins), n_bins))
        bm.sparse_bunch_histogram(
            self.beam.dt,
            histogram,
            np.ascontiguousarray(first_bins[order]),
            np.ascontiguousarray(order),
            n_bins,
            1.0 / self.window_bin_size,
        )
        for i, profile in enumerate(self.profiles_list):
            profile.n_macroparticles = histogram[i]

    def _check_coverage(self):
        """
        Count the macroparticles outside all windows (neither histogrammed
        nor kicked) and warn when their fraction exceeds
        warn_outside_fraction.
        """
        inside = float(np.sum(self.n_macroparticles))
        self.n_macroparticles_outside = max(
            int(round(self.beam.n_macroparticles - inside)), 0
        )
        if (
            self.warn_outside_fraction is not None
            and self.n_macroparticles_outside
            > self.warn_outside_fraction * self.beam.n_macroparticles
        ):
            warnings.warn(
                "Macroparticles outside the SparseBunch windows are neither "
                "histogrammed nor kicked (count in n_macroparticles_outside): "
                "enlarge length_in_sigma or set recenter_every.",
                UserWarning,
                stacklevel=2,
            )

    def _set_additional_cuts(
            self,
            _updated_filling_pattern: NumpyArray,
            injection_ordered_indices: NumpyArray | None = None,
            new_particles=None,
    ):
        """
        Internal method to update the cut array properties of the Sparse
        object with new cut_left | cut_right options around the additional
        indices.

        The left cut starts at the bucket index considered and the right cut
        is defined by its distance, in number of RF buckets (
        self._profile_length_in_buckets), from the bucket index considered.
        This is done as a pre-processing.

        The appending ORDER matters: the trackers pair profiles_list[i] with
        the beam particle block [i * N: (i + 1) * N], so new profiles must be
        appended in the order the corresponding particles were appended to
        the beam (injection order), which is not in general the bucket-index
        order. A single new index is unambiguous; several new indices at once
        are accepted only together with injection_ordered_indices, which gives the
        injection order explicitly.

        Parameters
        ----------
        _updated_filling_pattern
            Updated filling pattern (0/1 per bucket).
        injection_ordered_indices
            Bucket indices of the new profiles in injection order. Must
            contain exactly the indices that are new in
            _updated_filling_pattern. Optional when only one index is new.

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

        if injection_ordered_indices is not None:
            injection_ordered_indices = np.asarray(
                injection_ordered_indices, dtype=int
            ).ravel()
            if not np.array_equal(
                    np.sort(injection_ordered_indices), masked_indices
            ):
                raise ValueError(
                    f"The ordered new indices {injection_ordered_indices} do not "
                    f"match the indices that are new in the updated filling "
                    f"pattern {masked_indices}"
                )
            masked_indices = injection_ordered_indices
        elif _additional_indices > 1:
            raise ValueError(
                f"{_additional_indices} new indices were added in a single "
                f"update call without their injection order. Profiles are "
                f"paired with beam particle blocks by creation order, and "
                f"appending several new profiles in bucket-index order "
                f"silently mispairs (and thereby un-kicks) every batch whose "
                f"injection order differs from its bucket-index order. "
                f"Either update one index per call, in injection order, or "
                f"pass the new bucket indices in injection order via the "
                f"ordered-indices argument."
            )
        updated_cut_left, updated_cut_right = self._bunch_windows(
            masked_indices, new_particles=new_particles
        )

        self.cut_left_array = np.append(self.cut_left_array, updated_cut_left)
        self.cut_right_array = np.append(
            self.cut_right_array, updated_cut_right
        )
        self._filling_pattern = _updated_filling_pattern
        return _additional_indices

    def update_bunch_list(
        self,
        updated_bunch_list: list[int],
        new_bunch_indices: list[int] | None = None,
        new_particles=None,
    ):
        """
        Function to update the SparseBunch object to match the new bunch
        list in the case of newly injected bunches.

        The method creates additional windows around the newly injected
        bunches, and updates the internal arrays and numbering accordingly.

        Parameters
        ----------
        updated_bunch_list
            Updated bunch list. Must be the same length as the stored bunch
            list.
        new_bunch_indices
            Bucket indices of the newly injected bunches in INJECTION order.
            Required when more than one bunch is new in this call: profiles
            are paired with beam particle blocks by creation order, so the
            new profiles must be appended in the order the bunches were
            appended to the beam.
        new_particles
            The injected particles, so that the new windows are centred on
            them and not on whatever else sits in their RF buckets (the
            tails of neighbouring bunches): one block (slice or index array
            of beam.dt) per new bunch, in injection order, or one block for
            all of them, split by RF bucket. Default: all the particles of
            the buckets.
        """
        additional_filled_buckets = self._set_additional_cuts(
            _updated_filling_pattern=updated_bunch_list,
            injection_ordered_indices=new_bunch_indices,
            new_particles=new_particles,
        )
        self._update_profile_lists(
            _additional_indices=additional_filled_buckets
        )