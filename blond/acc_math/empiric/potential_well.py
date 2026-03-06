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

        epsilon = .1 / 100 * (np.max(y) - np.min(y))

        maxima_indices, _ = find_peaks(y)
        buckets = []

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
                    above_threshold = (
                            current_y >= (
                            threshold_y - epsilon)
                    )
                    next_falling = next_y <= current_y
                    next_above = next_y > threshold_y
                    if (
                            not inside_local_region
                            and above_threshold
                            and (next_falling)
                    ) or (
                            not inside_local_region
                            and next_above
                    ):
                        second_anchor_index = j
                        buckets.append(
                            (
                                x[min(max_idx, second_anchor_index)],
                                x[max(max_idx, second_anchor_index)],
                            )
                        )
                        break

        return np.array(buckets)

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
        n = len(self.bucket_list)
        for i, bucket in enumerate(self.bucket_list):  # type: ignore
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
