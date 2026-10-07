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

#include <vector>

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

// The five phase-space sums of a beam in ONE pass over dt and dE:
// sums = {sum(dt), sum(dE), sum(dt^2), sum(dE^2), sum(dt * dE)}.
// Every first- and second-order statistic (means, RMS sizes, RMS
// emittance) is a function of these, so fusing them reads each particle
// array once instead of once per sum.
//
// Reproducible: each thread sums a fixed contiguous share of the particles
// (schedule(static)) into its own slot, and the slots are added in thread
// order afterwards. An OpenMP `reduction` clause combines the threads'
// partial sums in an unspecified order instead, so repeated calls on the
// same beam differed in the last bits -- and the emittance, a difference
// of large sums (mean / sigma ~ 70 for a bunch), jittered by ~4e-12.
extern "C" void phase_space_sums(const real_t *__restrict__ dt,
                                 const real_t *__restrict__ dE, const index_t n,
                                 real_t *__restrict__ sums) {
  constexpr int n_sums = 5;
  const int n_threads = omp_get_max_threads();
  std::vector<real_t> thread_sums(static_cast<size_t>(n_sums) * n_threads, 0.0);

#pragma omp parallel
  {
    real_t dt_sum = 0.0;
    real_t dE_sum = 0.0;
    real_t dt_dt_sum = 0.0;
    real_t dE_dE_sum = 0.0;
    real_t dt_dE_sum = 0.0;

#pragma omp for schedule(static) nowait
    for (index_t idx = 0; idx < n; ++idx) {
      const real_t dt_i = dt[idx];
      const real_t dE_i = dE[idx];
      dt_sum += dt_i;
      dE_sum += dE_i;
      dt_dt_sum += dt_i * dt_i;
      dE_dE_sum += dE_i * dE_i;
      dt_dE_sum += dt_i * dE_i;
    }

    real_t *slot =
        &thread_sums[static_cast<size_t>(n_sums) * omp_get_thread_num()];
    slot[0] = dt_sum;
    slot[1] = dE_sum;
    slot[2] = dt_dt_sum;
    slot[3] = dE_dE_sum;
    slot[4] = dt_dE_sum;
  }

  for (int k = 0; k < n_sums; ++k) {
    real_t total = 0.0;
    for (int thread = 0; thread < n_threads; ++thread)
      total += thread_sums[static_cast<size_t>(n_sums) * thread + k];
    sums[k] = total;
  }
}
