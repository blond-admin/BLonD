# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np
from matplotlib import pyplot as plt


class PotentialWellHelper:
    def __init__(self, time_axis, voltage_axis):
        self.time_axis = np.array(time_axis)
        self.voltage_axis = np.array(voltage_axis)
        self.bucket_list = self.analyze_buckets()
        # self.principal_bucket_list = self.filter_buckets(self.bucket_list)

    def analyze_buckets(self):
        """
        Returns a list of (start_time, stop_time) tuples
        corresponding to potential wells (buckets).
        """
        y = self.voltage_axis
        x = self.time_axis

        # First derivative
        from scipy.signal import find_peaks

        # Local maxima: + slope to - slope
        maxima_indices, _ = find_peaks(y)

        buckets = []

        for i in range(len(maxima_indices)):
            index_anchor: int = maxima_indices[i]
            threshold_y = y[index_anchor]
            # go to left
            for j in range(index_anchor - 1, 0, -1):
                if y[j] >= threshold_y:
                    second_anchor_index = j
                    buckets.append(
                        (
                            x[min(index_anchor, second_anchor_index)],
                            x[max(index_anchor, second_anchor_index)],
                        )
                    )
                    break

            # go to right
            for j in range(index_anchor + 1, len(y)):
                if y[j] >= threshold_y:
                    second_anchor_index = j
                    buckets.append(
                        (
                            x[min(index_anchor, second_anchor_index)],
                            x[max(index_anchor, second_anchor_index)],
                        )
                    )
                    break
        print(buckets)
        return np.array(buckets)

    def plot(self):
        plt.plot(self.time_axis, self.voltage_axis)
        plt.ylim(*plt.ylim())
        buckets = self.bucket_list
        for i, bucket in enumerate(buckets):
            plt.fill_betweenx(
                y=np.linspace(*plt.ylim(), 10),
                x1=buckets[i, 0],
                x2=buckets[i, 1],
                alpha=0.1,
            )
            plt.draw()
            plt.pause(0.5)

    @staticmethod
    def filter_buckets(buckets):
        keep = []

        for i in range(len(buckets)):
            a_start, a_stop = buckets[i]

            contained = False
            for j in range(len(buckets)):
                if i == j:
                    continue

                b_start, b_stop = buckets[j]

                if b_start <= a_start and a_stop <= b_stop:
                    contained = True
                    break

            if not contained:
                keep.append(buckets[i])

        return np.array(keep)

    def get_in_bucket_mask(self):
        mask = np.zeros(len(self.time_axis), dtype=bool)

        for bucket in self.bucket_list:
            start = bucket[0]
            stop = bucket[1]
            assert start <= stop
            sel = (self.time_axis >= start) & (self.time_axis <= stop)
            mask = mask | sel
        return mask
