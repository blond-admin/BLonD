// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
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
                    const int n) {
real_t acc = 0.0;

#pragma omp parallel for reduction(+:acc)
  for (int idx = 0; idx < n; ++idx) {
      acc += array_1[idx];
  }

  return acc;
}

extern "C" real_t dot_product_1d_array(const real_t *__restrict__ array_1,
                    const real_t *__restrict__ array_2,
                    const int n) {
real_t acc = 0.0;

#pragma omp parallel for reduction(+:acc)
  for (int idx = 0; idx < n; ++idx) {
      acc += array_1[idx] * array_2[idx];
  }

  return acc;
}
