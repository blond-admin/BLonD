# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holder the `PotentialWellHelper`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class PotentialWellHelper:
    """
    Helper class to analyze and visualize a potential well.

    Helper class to analyze and visualize potential wells
    from a voltage waveform sampled over time.

    The class identifies buckets by locating local maxima
    in the voltage signal and finding surrounding points at the same
    or higher voltage level.

    Parameters
    ----------
    time_axis
        1D array representing the time coordinate.
    voltage_axis
        1D array representing the voltage values corresponding to
        `time_axis`.

    Attributes
    ----------
    time_axis
        Time axis as a NumPy array.
    voltage_axis
        Voltage axis as a NumPy array.
    bucket_list
        Array of shape (N, 2) containing `(start_time, stop_time)` for
        each detected bucket.

    Examples
    --------
    >>> xs = np.linspace(0.4, 6 * np.pi - 0.3, 1000)
    >>> ys = np.cos(xs)
    >>> pwh = PotentialWellHelper(xs, ys)
    >>> pwh.plot()
    >>> plt.show()
    """

    def __init__(self, time_axis: NumpyArray, voltage_axis: NumpyArray):
        self.time_axis = np.array(time_axis)
        self.voltage_axis = np.array(voltage_axis)
        self.bucket_list = self._analyze_buckets()

    def _analyze_buckets(self) -> NumpyArray:
        """
        Detect buckets in the voltage waveform.

        Local maxima of the voltage are used as anchors. For each local
        maximum, the method searches left and right until the voltage
        reaches the same or a higher level, defining the bucket limits.

        Returns
        -------
        buckets
            Array of shape (N, 2) containing `(start_time, stop_time)`
            tuples for each detected bucket.
        """
        y = self.voltage_axis
        x = self.time_axis

        maxima_indices, _ = find_peaks(y)

        if len(maxima_indices) == 0:
            buckets = self._find_single_partial_bucket(x, y)
        else:
            buckets = self._find_n_complete_buckets(maxima_indices, x, y)
            buckets = self._handle_border(maxima_indices, x, y, buckets)
        return np.array(buckets)

    def _find_single_partial_bucket(
        self, x: NumpyArray, y: NumpyArray
    ) -> list[tuple[float, float]]:
        """
        Identify a single partial bucket.

        Partial buckets appear, when there is no maximum in the data
        or when there is a single maximum, but no normal bucket can be detected
        because all values are below the maximum.
        This will happen for partial single buckets, two incomplete buckets,
        or the leftmost and rightmost bucket in a multi-bucket dataset.

        Parameters
        ----------
        x : NumpyArray
            Array of x-coordinates corresponding to the `y` values.
        y : NumpyArray
            Array of y-coordinates representing the signal or data.

        Returns
        -------
        list of tuple of float
            A list containing at most one tuple `(x_start, x_end)` representing
            the partial bucket. Returns an empty list if no partial bucket is detected.

        Notes
        -----
        - The threshold is set to the smaller of the first or last y-value.
        - Local minima are found using `find_peaks(-y)`; the bucket is created
          only if exactly one minimum exists.
        - The bucket spans the region where the signal is below the threshold
          at the boundary.
        """
        buckets = []
        minima_indices, _ = find_peaks(-y)

        thershold = min(float(y[0]), float(y[-1]))
        mask = y <= thershold
        if len(minima_indices) == 1:
            start = int(np.argmax(mask))
            stop = int(len(x) - np.argmax(mask[::-1])) - 1
            buckets.append(
                (
                    x[min(start, stop)],
                    x[max(start, stop)],
                )
            )
        return buckets

    def _find_n_complete_buckets(
        self, maxima_indices: NumpyArray, x: NumpyArray, y: NumpyArray
    ) -> list[tuple[float, float]]:
        """
        Identify buckets around local maxima in the `y` data.

        For each local maximum specified by `maxima_indices`, this function
        searches to the left and right of the maximum to find the extent
        where the values remain close to the maximum within a small epsilon.
        Each such contiguous region is returned as a tuple of `(x_start, x_end)`
        coordinates corresponding to the boundaries of the region.

        Parameters
        ----------
        maxima_indices : NumpyArray
            Array of indices corresponding to local maxima in `y`.
        x : NumpyArray
            Array of x-coordinates corresponding to the `y` values.
        y : NumpyArray
            Array of y-coordinates representing the signal or data from which
            maxima are identified.

        Returns
        -------
        list of tuple of float
            A list of tuples, where each tuple contains the start and end
            `x` coordinates defining the region (bucket) around a local maximum.

        Notes
        -----
        - An epsilon tolerance is used to define the region around the maximum,
          calculated as `0.1%` of the total `y` range.
        - Buckets are identified separately for each maximum and are inclusive
          of the maximum's x-coordinate.
        - The function assumes `x` and `y` are 1-dimensional arrays of the same length.
        """
        buckets = []
        epsilon = 0.1 / 100 * (np.max(y) - np.min(y))

        for nth_maximum in range(len(maxima_indices)):  # type: ignore
            max_idx: int = maxima_indices[nth_maximum]  # type: ignore

            threshold_y = float(y[max_idx])
            for direction, range_args in zip(
                (1, -1),
                ((max_idx + 1, len(y) - 1, +1), (max_idx + -1, 1, -1)),
                strict=False,
            ):
                inside_local_region = True

                # Search to the left of the maximum
                for j in range(range_args[0], range_args[1], range_args[2]):
                    current_y = y[j]
                    next_y = y[j + direction]
                    if (current_y > (threshold_y + epsilon)) or (
                        current_y < (threshold_y - epsilon)
                    ):
                        inside_local_region = False  # once false stays false
                    above_threshold = current_y >= (threshold_y - epsilon)
                    next_falling = next_y <= current_y
                    next_above = next_y > threshold_y
                    if (
                        not inside_local_region
                        and above_threshold
                        and (next_falling)
                    ) or (not inside_local_region and next_above):
                        second_anchor_index = j
                        buckets.append(
                            (
                                x[min(max_idx, second_anchor_index)],
                                x[max(max_idx, second_anchor_index)],
                            )
                        )
                        break
        return buckets

    def _handle_border(
        self,
        maxima_indices: NumpyArray,
        x: NumpyArray,
        y: NumpyArray,
        buckets: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Extend buckets to include partial regions at the borders of the data.

        This function checks for potential regions (buckets) at the beginning
        and end of the `x` and `y` arrays that are not captured by the
        identified local maxima. It uses `_find_single_partial_bucket` to
        detect these border regions and appends or prepends them to the
        existing list of complete buckets.

        Parameters
        ----------
        maxima_indices : NumpyArray
            Array of indices corresponding to local maxima in `y`.
        x : NumpyArray
            Array of x-coordinates corresponding to the `y` values.
        y : NumpyArray
            Array of y-coordinates representing the signal or data from which
            maxima are identified.
        buckets : list of tuple of float
            Existing list of buckets (start and end x-coordinates) around maxima.

        Returns
        -------
        list of tuple of float
            Updated list of buckets including potential partial regions at
            the left and right borders of the data.

        Notes
        -----
        - The left border region is checked from the start of `x` to the
          first maximum.
        - The right border region is checked from the last maximum to the
          end of `x`.
        """
        # left
        start = 0
        stop = maxima_indices[0] + 1
        sel = slice(start, stop)
        bucket = self._find_single_partial_bucket(x[sel], y[sel])
        if len(bucket) > 0:
            # prepend
            buckets = bucket + buckets

        # right
        start = maxima_indices[-1]
        stop = len(x)
        sel = slice(start, stop)
        bucket = self._find_single_partial_bucket(x[sel], y[sel])
        if len(bucket) > 0:
            b = bucket[0]
            bucket = [
                (
                    b[0],
                    b[1],
                )
            ]
            # append
            buckets = buckets + bucket
        return buckets

    def plot(self) -> None:
        """
        Plot the voltage waveform and highlight detected buckets.

        Each bucket is visualized as a shaded vertical region spanning
        the full voltage range.
        """
        y = self.voltage_axis
        x = self.time_axis
        plt.plot(x, y)
        plt.ylim(*plt.ylim())
        for _i, bucket in enumerate(self.bucket_list):  # type: ignore
            x1 = bucket[0]
            x2 = bucket[1]
            y1 = y[np.argmin(np.abs(x - x1))]
            y2 = y[np.argmin(np.abs(x - x2))]
            plt.plot([x1, x2], [y1, y2])

    def get_in_bucket_mask(self) -> NumpyArray:
        """
        Compute a boolean mask indicating time points inside any bucket.

        Returns
        -------
        mask
            Boolean array with the same length as `time_axis`, where
            `True` indicates the time point lies within a bucket.
        """
        mask = np.zeros(len(self.time_axis), dtype=bool)

        for start, stop in self.bucket_list:  # type: ignore
            sel = (self.time_axis >= start) & (self.time_axis <= stop)
            mask |= sel

        return mask

    def get_principal_bucket_slices(self) -> list[slice]:
        """
        Return index slices corresponding to contiguous bucket regions.

        The slices can be used for in-place operations on arrays indexed
        by the time axis.

        Returns
        -------
        slices
            List of Python `slice` objects, each representing a
            contiguous bucket region.
        """
        mask = self.get_in_bucket_mask()
        diff_mask = np.diff(mask.astype(int))

        starts = np.where(diff_mask == 1)[0] + 1
        stops = np.where(diff_mask == -1)[0] + 1

        if mask[0]:
            starts = np.concatenate(([0], starts))
        if mask[-1]:
            stops = np.append(stops, len(mask))

        slices = []
        for start, stop in zip(starts, stops, strict=False):
            slices.append(slice(int(start), int(stop)))

        return slices
