// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// The binning rule of the histograms, shared by the eager `histogram`
// (histogram.cpp) and the deferred executor (`bin_chunk`), so both count
// every particle in the same bin.

#pragma once

#include <cmath>

#include "blond_common.h"

// Bin of `value` in a histogram of `n_bins` over [start, stop], as a
// double: the caller drops anything outside [0, n_bins) before
// converting it to an integer, since converting an out-of-range double
// is undefined behaviour.
inline double histogram_bin_position(const real_t value, const real_t start,
                                     const real_t stop,
                                     const real_t inv_bin_width,
                                     const index_t n_bins) {
  double position = std::floor((value - start) * inv_bin_width);
  // Scaling is not exact: a value at or just below `stop` can land on
  // n_bins. It belongs in the last bin, as in np.histogram.
  if (position >= static_cast<double>(n_bins) && value <= stop) {
    position = static_cast<double>(n_bins - 1);
  }
  return position;
}
