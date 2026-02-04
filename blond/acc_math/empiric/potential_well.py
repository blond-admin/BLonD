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
    time_axis : array_like
        1D array representing the time coordinate.
    voltage_axis : array_like
        1D array representing the voltage values corresponding to
        `time_axis`.

    Attributes
    ----------
    time_axis : numpy.ndarray
        Time axis as a NumPy array.
    voltage_axis : numpy.ndarray
        Voltage axis as a NumPy array.
    bucket_list : numpy.ndarray
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

        from scipy.signal import find_peaks

        maxima_indices, _ = find_peaks(y)
        buckets = []

        for i in range(len(maxima_indices)):  # type: ignore
            index_anchor: int = maxima_indices[i]  # type: ignore
            threshold_y = y[index_anchor]

            # Search to the left
            for j in range(index_anchor - 1, 0, -1):
                if y[j] >= threshold_y:
                    if (j + 1) <= len(y):
                        second_anchor_index = j + 1
                    else:
                        second_anchor_index = j
                    buckets.append(
                        (
                            x[min(index_anchor, second_anchor_index)],
                            x[max(index_anchor, second_anchor_index)],
                        )
                    )
                    break

            # Search to the right
            for j in range(index_anchor + 1, len(y)):
                if y[j] >= threshold_y:
                    if (j - 1) >= 0:
                        second_anchor_index = j - 1
                    else:
                        second_anchor_index = j
                    buckets.append(
                        (
                            x[min(index_anchor, second_anchor_index)],
                            x[max(index_anchor, second_anchor_index)],
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
        plt.plot(self.time_axis, self.voltage_axis)
        plt.ylim(*plt.ylim())

        for bucket in self.bucket_list:  # type: ignore
            plt.fill_betweenx(
                y=np.linspace(*plt.ylim(), 10),
                x1=bucket[0],
                x2=bucket[1],
                alpha=0.1,
            )
            plt.draw()
            plt.pause(0.5)

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
