// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

/**
C++ Math library
@Author: Leonard Thiele
@Date: 27.04.2026
*/
#include "blond_common.h"
#include "openmp.h"

extern "C" real_t sum_1d_array(const real_t *__restrict__ array_1,
                               const index_t n) {
  real_t acc = 0.0;

#pragma omp parallel for reduction(+ : acc)
  for (index_t idx = 0; idx < n; ++idx) {
    acc += array_1[idx];
  }

  return acc;
}

extern "C" real_t dot_product_1d_array(const real_t *__restrict__ array_1,
                                       const real_t *__restrict__ array_2,
                                       const index_t n) {
  real_t acc = 0.0;

#pragma omp parallel for reduction(+ : acc)
  for (index_t idx = 0; idx < n; ++idx) {
    acc += array_1[idx] * array_2[idx];
  }

  return acc;
}

extern "C" real_t
centred_dot_product_1d_array(const real_t *__restrict__ array_1,
                             const real_t offset_1,
                             const real_t *__restrict__ array_2,
                             const real_t offset_2, const index_t n) {
  real_t acc = 0.0;

  // Centre per element instead of in a temporary array: no copy of a
  // beam-sized input, and no cancellation as in dot(x, y) - n*mx*my.
#pragma omp parallel for reduction(+ : acc)
  for (index_t idx = 0; idx < n; ++idx) {
    acc += (array_1[idx] - offset_1) * (array_2[idx] - offset_2);
  }

  return acc;
}

extern "C" index_t count_flag(const int *__restrict__ flags, const int flag,
                              const index_t n) {
  index_t count = 0;

#pragma omp parallel for reduction(+ : count)
  for (index_t idx = 0; idx < n; ++idx) {
    count += static_cast<index_t>(flags[idx] == flag);
  }

  return count;
}
